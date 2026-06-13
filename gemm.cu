#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

using namespace nvcuda;

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Naive GEMM: C = A * B, A[M x K], B[K x N], C[M x N]
__global__ void gemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled GEMM with shared memory
#define TILE 16

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// WMMA GEMM using Tensor Cores
// Each warp computes a 16x16 output tile
// Block: 128 threads = 4 warps, laid out 2x2 over a 32x32 block tile
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_TILE 32  // block tile = 2x2 warp tiles

__global__ void gemm_wmma(const half* A, const half* B, float* C,
                           int M, int N, int K) {
    int warpId = threadIdx.x / 32;
    int warpRow = warpId / 2;
    int warpCol = warpId % 2;

    int block_row = blockIdx.y * WARP_TILE + warpRow * WMMA_M;
    int block_col = blockIdx.x * WARP_TILE + warpCol * WMMA_N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int t = 0; t < K; t += WMMA_K) {
        if (block_row < M && t < K)
            wmma::load_matrix_sync(a_frag, A + block_row * K + t, K);
        if (t < K && block_col < N)
            wmma::load_matrix_sync(b_frag, B + t * N + block_col, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (block_row < M && block_col < N)
        wmma::store_matrix_sync(C + block_row * N + block_col, c_frag, N, wmma::mem_row_major);
}

// WMMA + cp.async: async global→shared, then shared→fragment→Tensor Core
// Double buffering to overlap load of next tile with compute of current tile
#define ASYNC_TILE 32  // block tile
#define ASYNC_WMMA 16  // wmma tile

// Helper: async load a tile from global to shared, zero-padding OOB
// Uses 4-byte (2 half) copies as required by __pipeline_memcpy_async
__device__ void async_load_tile(half* smem, const half* gmem,
                                 int smem_rows, int smem_cols, int smem_stride,
                                 int grow_base, int gcol_base,
                                 int M_bound, int N_bound, int gmem_stride) {
    // Total uint32_t copies = (rows * cols) / 2
    int total_pairs = (smem_rows * smem_cols) / 2;
    for (int i = threadIdx.x; i < total_pairs; i += blockDim.x) {
        // Map linear index to (row, col_pair) in shared memory
        int pairs_per_row = smem_cols / 2;
        int r = i / pairs_per_row;
        int cp = i % pairs_per_row;
        int c = cp * 2;

        int gr = grow_base + r;
        int gc = gcol_base + c;

        half* dst = &smem[r * smem_stride + c];
        const half* src = &gmem[gr * gmem_stride + gc];

        if (gr < M_bound && gc + 1 < N_bound) {
            __pipeline_memcpy_async(dst, src, sizeof(uint32_t));
        } else {
            // Zero-pad out of bounds
            dst[0] = __float2half(0.0f);
            dst[1] = __float2half(0.0f);
        }
    }
}

__global__ void gemm_wmma_async(const half* A, const half* B, float* C,
                                 int M, int N, int K) {
    // Double buffer: two slots for A and B tiles in shared memory
    __shared__ half As[2][ASYNC_TILE * ASYNC_WMMA];  // [buf][32*16], row-major
    __shared__ half Bs[2][ASYNC_WMMA * ASYNC_TILE];  // [buf][16*32], row-major

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / 2;
    int warpCol = warpId % 2;

    int block_row = blockIdx.y * ASYNC_TILE;
    int block_col = blockIdx.x * ASYNC_TILE;

    wmma::fragment<wmma::matrix_a, ASYNC_WMMA, ASYNC_WMMA, ASYNC_WMMA, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, ASYNC_WMMA, ASYNC_WMMA, ASYNC_WMMA, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, ASYNC_WMMA, ASYNC_WMMA, ASYNC_WMMA, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int numTiles = (K + ASYNC_WMMA - 1) / ASYNC_WMMA;

    // Prefetch tile 0 into buffer 0
    async_load_tile(As[0], A, ASYNC_TILE, ASYNC_WMMA, ASYNC_WMMA,
                    block_row, 0, M, K, K);
    async_load_tile(Bs[0], B, ASYNC_WMMA, ASYNC_TILE, ASYNC_TILE,
                    0, block_col, K, N, N);
    __pipeline_commit();

    for (int t = 0; t < numTiles; t++) {
        int cur = t % 2;
        int nxt = 1 - cur;

        // Prefetch next tile into other buffer
        if (t + 1 < numTiles) {
            int nt = t + 1;
            async_load_tile(As[nxt], A, ASYNC_TILE, ASYNC_WMMA, ASYNC_WMMA,
                            block_row, nt * ASYNC_WMMA, M, K, K);
            async_load_tile(Bs[nxt], B, ASYNC_WMMA, ASYNC_TILE, ASYNC_TILE,
                            nt * ASYNC_WMMA, block_col, K, N, N);
            __pipeline_commit();
        }

        // Wait for current tile's data
        // If we just prefetched next tile: 2 pending, wait until ≤1 (current done)
        // Last iteration: no prefetch, 1 pending, must wait for all
        if (t + 1 < numTiles)
            __pipeline_wait_prior(1);
        else
            __pipeline_wait_prior(0);
        __syncthreads();

        // WMMA from shared memory
        wmma::load_matrix_sync(a_frag, &As[cur][warpRow * ASYNC_WMMA * ASYNC_WMMA], ASYNC_WMMA);
        wmma::load_matrix_sync(b_frag, &Bs[cur][warpCol * ASYNC_WMMA], ASYNC_TILE);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    int out_row = block_row + warpRow * ASYNC_WMMA;
    int out_col = block_col + warpCol * ASYNC_WMMA;
    if (out_row < M && out_col < N)
        wmma::store_matrix_sync(C + out_row * N + out_col, c_frag, N, wmma::mem_row_major);
}

// WMMA + register tiling: each warp computes 2x2 grid of 16x16 tiles
// 64x64 block = 2x2 warps, each warp covers 32x32 output
#define RT_BLOCK 64
#define RT_WARP_M 2
#define RT_WARP_N 2

__global__ void gemm_wmma_regtile(const half* A, const half* B, float* C,
                                    int M, int N, int K) {
    constexpr int WM = 16;
    constexpr int WARPS_M = RT_BLOCK / (RT_WARP_M * WM);
    constexpr int WARPS_N = RT_BLOCK / (RT_WARP_N * WM);

    __shared__ half As[RT_BLOCK * WM];
    __shared__ half Bs[WM * RT_BLOCK];

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / WARPS_N;
    int warpCol = warpId % WARPS_N;
    int block_row = blockIdx.y * RT_BLOCK;
    int block_col = blockIdx.x * RT_BLOCK;

    wmma::fragment<wmma::accumulator, WM, WM, WM, float> c_frag[RT_WARP_M][RT_WARP_N];
    for (int i = 0; i < RT_WARP_M; i++)
        for (int j = 0; j < RT_WARP_N; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    for (int t = 0; t < K; t += WM) {
        for (int idx = threadIdx.x; idx < RT_BLOCK * WM; idx += blockDim.x) {
            int r = idx / WM, c = idx % WM;
            int gr = block_row + r, gc = t + c;
            As[idx] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        for (int idx = threadIdx.x; idx < WM * RT_BLOCK; idx += blockDim.x) {
            int r = idx / RT_BLOCK, c = idx % RT_BLOCK;
            int gr = t + r, gc = block_col + c;
            Bs[idx] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WM, WM, WM, half, wmma::row_major> a_frag[RT_WARP_M];
        wmma::fragment<wmma::matrix_b, WM, WM, WM, half, wmma::row_major> b_frag[RT_WARP_N];

        for (int i = 0; i < RT_WARP_M; i++) {
            int a_row = (warpRow * RT_WARP_M + i) * WM;
            wmma::load_matrix_sync(a_frag[i], &As[a_row * WM], WM);
        }
        for (int j = 0; j < RT_WARP_N; j++) {
            int b_col = (warpCol * RT_WARP_N + j) * WM;
            wmma::load_matrix_sync(b_frag[j], &Bs[b_col], RT_BLOCK);
        }

        for (int i = 0; i < RT_WARP_M; i++)
            for (int j = 0; j < RT_WARP_N; j++)
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);

        __syncthreads();
    }

    for (int i = 0; i < RT_WARP_M; i++) {
        for (int j = 0; j < RT_WARP_N; j++) {
            int out_row = block_row + (warpRow * RT_WARP_M + i) * WM;
            int out_col = block_col + (warpCol * RT_WARP_N + j) * WM;
            if (out_row < M && out_col < N)
                wmma::store_matrix_sync(C + out_row * N + out_col, c_frag[i][j], N,
                                        wmma::mem_row_major);
        }
    }
}

// CPU reference
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

float max_diff(const float* ref, const float* out, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; i++)
        d = fmaxf(d, fabsf(ref[i] - out[i]));
    return d;
}

int main() {
    const int M = 2048, N = 2048, K = 2048;

    float* h_A   = (float*)malloc(M * K * sizeof(float));
    float* h_B   = (float*)malloc(K * N * sizeof(float));
    float* h_C   = (float*)malloc(M * N * sizeof(float));
    float* h_ref = (float*)malloc(M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    gemm_cpu(h_A, h_B, h_ref, M, N, K);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double cpu_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double cpu_tflops = 2.0 * M * N * K / cpu_s / 1e12;
    printf("gemm_cpu:        M=%d N=%d K=%d, %.3f s, %.4f TFLOPS\n", M, N, K, cpu_s, cpu_tflops);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Prepare half-precision inputs for WMMA
    half* h_A_half = (half*)malloc(M * K * sizeof(half));
    half* h_B_half = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++) h_B_half[i] = __float2half(h_B[i]);

    half *d_A_half, *d_B_half;
    CHECK_CUDA(cudaMalloc(&d_A_half, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_half, K * N * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(d_A_half, h_A_half, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_half, h_B_half, K * N * sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    // --- naive ---
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("gemm_naive:      M=%d N=%d K=%d, max_err=%.6f — %s\n",
           M, N, K, max_diff(h_ref, h_C, M * N),
           max_diff(h_ref, h_C, M * N) < 1e-2f ? "PASS" : "FAIL");

    // --- tiled ---
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("gemm_tiled:      M=%d N=%d K=%d, max_err=%.6f — %s\n",
           M, N, K, max_diff(h_ref, h_C, M * N),
           max_diff(h_ref, h_C, M * N) < 1e-2f ? "PASS" : "FAIL");

    // --- wmma ---
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
    dim3 wmma_block(128);
    dim3 wmma_grid((N + WARP_TILE - 1) / WARP_TILE, (M + WARP_TILE - 1) / WARP_TILE);
    gemm_wmma<<<wmma_grid, wmma_block>>>(d_A_half, d_B_half, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float wmma_err = max_diff(h_ref, h_C, M * N);
    printf("gemm_wmma:       M=%d N=%d K=%d, max_err=%.6f — %s\n",
           M, N, K, wmma_err,
           wmma_err < 5e-1f ? "PASS" : "FAIL");

    // --- wmma + cp.async ---
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
    dim3 async_block(128);
    dim3 async_grid((N + ASYNC_TILE - 1) / ASYNC_TILE, (M + ASYNC_TILE - 1) / ASYNC_TILE);
    gemm_wmma_async<<<async_grid, async_block>>>(d_A_half, d_B_half, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float async_err = max_diff(h_ref, h_C, M * N);
    printf("gemm_wmma_async: M=%d N=%d K=%d, max_err=%.6f — %s\n",
           M, N, K, async_err,
           async_err < 5e-1f ? "PASS" : "FAIL");

    // --- wmma + register tiling ---
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
    dim3 rt_block(128);
    dim3 rt_grid((N + RT_BLOCK - 1) / RT_BLOCK, (M + RT_BLOCK - 1) / RT_BLOCK);
    gemm_wmma_regtile<<<rt_grid, rt_block>>>(d_A_half, d_B_half, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float rt_err = max_diff(h_ref, h_C, M * N);
    printf("gemm_regtile:    M=%d N=%d K=%d, max_err=%.6f — %s\n",
           M, N, K, rt_err,
           rt_err < 5e-1f ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_half); cudaFree(d_B_half);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    free(h_A_half); free(h_B_half);
    return 0;
}
