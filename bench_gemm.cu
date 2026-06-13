#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>

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

// ============================================================
// Kernel 1: WMMA, 32x32 block tile (4 warps, 2x2)
// ============================================================
template <int BLOCK_TILE>
__global__ void gemm_wmma_t(const half* A, const half* B, float* C,
                             int M, int N, int K) {
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int WARPS_M = BLOCK_TILE / WM;
    constexpr int WARPS_N = BLOCK_TILE / WN;

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / WARPS_N;
    int warpCol = warpId % WARPS_N;

    int block_row = blockIdx.y * BLOCK_TILE + warpRow * WM;
    int block_col = blockIdx.x * BLOCK_TILE + warpCol * WN;

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int t = 0; t < K; t += WK) {
        wmma::load_matrix_sync(a_frag, A + block_row * K + t, K);
        wmma::load_matrix_sync(b_frag, B + t * N + block_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (block_row < M && block_col < N)
        wmma::store_matrix_sync(C + block_row * N + block_col, c_frag, N, wmma::mem_row_major);
}

// ============================================================
// Kernel 2: WMMA + cp.async + double buffer
// ============================================================
template <int BLOCK_TILE>
__global__ void gemm_wmma_async_t(const half* A, const half* B, float* C,
                                    int M, int N, int K) {
    constexpr int WM = 16;
    constexpr int WARPS_M = BLOCK_TILE / WM;
    constexpr int WARPS_N = BLOCK_TILE / WM;
    constexpr int A_TILE_ELEMS = BLOCK_TILE * WM;      // rows * K_tile
    constexpr int B_TILE_ELEMS = WM * BLOCK_TILE;      // K_tile * cols
    constexpr int A_PAIRS = A_TILE_ELEMS / 2;
    constexpr int B_PAIRS = B_TILE_ELEMS / 2;

    __shared__ half As[2][A_TILE_ELEMS];
    __shared__ half Bs[2][B_TILE_ELEMS];

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / WARPS_N;
    int warpCol = warpId % WARPS_N;
    int block_row = blockIdx.y * BLOCK_TILE;
    int block_col = blockIdx.x * BLOCK_TILE;
    int nWarps = WARPS_M * WARPS_N;
    int nThreads = nWarps * 32;

    wmma::fragment<wmma::matrix_a, WM, WM, WM, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WM, WM, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WM, WM, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int numTiles = K / WM;

    // Prefetch tile 0
    for (int i = threadIdx.x; i < A_PAIRS; i += nThreads) {
        int r = (i * 2) / WM;
        int c = (i * 2) % WM;
        __pipeline_memcpy_async(&As[0][r * WM + c],
                                &A[(block_row + r) * K + c],
                                sizeof(uint32_t));
    }
    for (int i = threadIdx.x; i < B_PAIRS; i += nThreads) {
        int r = (i * 2) / BLOCK_TILE;
        int c = (i * 2) % BLOCK_TILE;
        __pipeline_memcpy_async(&Bs[0][r * BLOCK_TILE + c],
                                &B[r * N + block_col + c],
                                sizeof(uint32_t));
    }
    __pipeline_commit();

    for (int t = 0; t < numTiles; t++) {
        int cur = t % 2;
        int nxt = 1 - cur;

        if (t + 1 < numTiles) {
            int kt = (t + 1) * WM;
            for (int i = threadIdx.x; i < A_PAIRS; i += nThreads) {
                int r = (i * 2) / WM;
                int c = (i * 2) % WM;
                __pipeline_memcpy_async(&As[nxt][r * WM + c],
                                        &A[(block_row + r) * K + kt + c],
                                        sizeof(uint32_t));
            }
            for (int i = threadIdx.x; i < B_PAIRS; i += nThreads) {
                int r = (i * 2) / BLOCK_TILE;
                int c = (i * 2) % BLOCK_TILE;
                __pipeline_memcpy_async(&Bs[nxt][r * BLOCK_TILE + c],
                                        &B[(kt + r) * N + block_col + c],
                                        sizeof(uint32_t));
            }
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &As[cur][warpRow * WM * WM], WM);
        wmma::load_matrix_sync(b_frag, &Bs[cur][warpCol * WM], BLOCK_TILE);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    int out_row = block_row + warpRow * WM;
    int out_col = block_col + warpCol * WM;
    if (out_row < M && out_col < N)
        wmma::store_matrix_sync(C + out_row * N + out_col, c_frag, N, wmma::mem_row_major);
}

// ============================================================
// Benchmark harness
// ============================================================
typedef void (*kernel_fn)(const half*, const half*, float*, int, int, int);

struct KernelConfig {
    const char* name;
    kernel_fn fn;
    int block_tile;
    int threads;
};

double bench(KernelConfig& cfg, const half* dA, const half* dB, float* dC,
             int M, int N, int K, int warmup, int iters) {
    dim3 block(cfg.threads);
    dim3 grid((N + cfg.block_tile - 1) / cfg.block_tile,
              (M + cfg.block_tile - 1) / cfg.block_tile);

    for (int i = 0; i < warmup; i++)
        cfg.fn<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        cfg.fn<<<grid, block>>>(dA, dB, dC, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double avg_s = (ms / 1000.0) / iters;
    double flops = 2.0 * M * N * K;
    return flops / avg_s / 1e12;  // TFLOPS
}

int main() {
    int sizes[] = {1024, 2048, 4096, 8192};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    // Register all kernel configs
    KernelConfig configs[] = {
        {"wmma_32",        gemm_wmma_t<32>,        32,  128},  // 4 warps
        {"wmma_64",        gemm_wmma_t<64>,        64,  512},  // 16 warps
        {"wmma_async_32",  gemm_wmma_async_t<32>,  32,  128},
        {"wmma_async_64",  gemm_wmma_async_t<64>,  64,  512},
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);

    // Print header
    printf("%-16s", "Kernel");
    for (int s = 0; s < nsizes; s++)
        printf("  %5dx%-5d", sizes[s], sizes[s]);
    printf("\n");
    for (int i = 0; i < 16 + nsizes * 13; i++) printf("-");
    printf("\n");

    for (int ci = 0; ci < nconfigs; ci++) {
        printf("%-16s", configs[ci].name);
        for (int si = 0; si < nsizes; si++) {
            int M = sizes[si], N = sizes[si], K = sizes[si];

            half *dA, *dB;
            float *dC;
            CHECK_CUDA(cudaMalloc(&dA, (size_t)M * K * sizeof(half)));
            CHECK_CUDA(cudaMalloc(&dB, (size_t)K * N * sizeof(half)));
            CHECK_CUDA(cudaMalloc(&dC, (size_t)M * N * sizeof(float)));
            CHECK_CUDA(cudaMemset(dA, 0, (size_t)M * K * sizeof(half)));
            CHECK_CUDA(cudaMemset(dB, 0, (size_t)K * N * sizeof(half)));

            double tflops = bench(configs[ci], dA, dB, dC, M, N, K, 5, 20);
            printf("  %9.2f TF", tflops);

            cudaFree(dA); cudaFree(dB); cudaFree(dC);
        }
        printf("\n");
    }

    return 0;
}
