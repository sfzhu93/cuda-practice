#include "kernels.cuh"

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
    return flops / avg_s / 1e12;
}

int main() {
    int sizes[] = {1024, 2048, 4096, 8192};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    KernelConfig configs[] = {
        {"wmma_32",        gemm_wmma_t<32>,                       32,  128},
        {"wmma_64",        gemm_wmma_t<64>,                       64,  512},
        {"wmma_async_32",  gemm_wmma_async_t<32>,                 32,  128},
        {"wmma_async_64",  gemm_wmma_async_t<64>,                 64,  512},
        {"rt_64_2x2",      gemm_wmma_regtile_t<64, 64, 2, 2>,    64,  128},
        {"rt_128_2x2",     gemm_wmma_regtile_t<128, 128, 2, 2>,  128, 512},
        {"rt_128_2x4",     gemm_wmma_regtile_t<128, 128, 2, 4>,  128, 256},
        {"rt_128_4x4",     gemm_wmma_regtile_t<128, 128, 4, 4>,  128, 128},
    };
    int nconfigs = sizeof(configs) / sizeof(configs[0]);

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
