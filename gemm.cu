#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

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
    // 2x2 warp layout within block
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
    const int M = 256, N = 256, K = 256;

    float* h_A   = (float*)malloc(M * K * sizeof(float));
    float* h_B   = (float*)malloc(K * N * sizeof(float));
    float* h_C   = (float*)malloc(M * N * sizeof(float));
    float* h_ref = (float*)malloc(M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    gemm_cpu(h_A, h_B, h_ref, M, N, K);

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
    printf("gemm_naive:  M=%d N=%d K=%d, max_err=%.6f — %s\n",
           M, N, K, max_diff(h_ref, h_C, M * N),
           max_diff(h_ref, h_C, M * N) < 1e-3f ? "PASS" : "FAIL");

    // --- tiled ---
    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("gemm_tiled:  M=%d N=%d K=%d, max_err=%.6f — %s\n",
           M, N, K, max_diff(h_ref, h_C, M * N),
           max_diff(h_ref, h_C, M * N) < 1e-3f ? "PASS" : "FAIL");

    // --- wmma ---
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
    dim3 wmma_block(128);  // 4 warps
    dim3 wmma_grid((N + WARP_TILE - 1) / WARP_TILE, (M + WARP_TILE - 1) / WARP_TILE);
    gemm_wmma<<<wmma_grid, wmma_block>>>(d_A_half, d_B_half, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float wmma_err = max_diff(h_ref, h_C, M * N);
    printf("gemm_wmma:   M=%d N=%d K=%d, max_err=%.6f — %s\n",
           M, N, K, wmma_err,
           wmma_err < 5e-2f ? "PASS" : "FAIL");  // fp16 accumulation, looser tolerance

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_half); cudaFree(d_B_half);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    free(h_A_half); free(h_B_half);
    return 0;
}
