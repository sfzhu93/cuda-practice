#include "kernels.cuh"
#include <cmath>
#include <ctime>

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
    printf("gemm_cpu:        M=%d N=%d K=%d, %.3f s, %.4f TFLOPS\n",
           M, N, K, cpu_s, cpu_tflops);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    half* h_A_half = (half*)malloc(M * K * sizeof(half));
    half* h_B_half = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) h_A_half[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; i++) h_B_half[i] = __float2half(h_B[i]);

    half *d_A_half, *d_B_half;
    CHECK_CUDA(cudaMalloc(&d_A_half, M * K * sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B_half, K * N * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(d_A_half, h_A_half, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_half, h_B_half, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // --- naive ---
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
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
    constexpr int WMMA_TILE = 32;
    dim3 wmma_block(128);
    dim3 wmma_grid((N + WMMA_TILE - 1) / WMMA_TILE, (M + WMMA_TILE - 1) / WMMA_TILE);
    gemm_wmma_t<WMMA_TILE><<<wmma_grid, wmma_block>>>(d_A_half, d_B_half, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float wmma_err = max_diff(h_ref, h_C, M * N);
    printf("gemm_wmma:       M=%d N=%d K=%d, max_err=%.6f — %s\n",
           M, N, K, wmma_err, wmma_err < 5e-1f ? "PASS" : "FAIL");

    // --- wmma + cp.async ---
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
    gemm_wmma_async_t<WMMA_TILE><<<wmma_grid, wmma_block>>>(d_A_half, d_B_half, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float async_err = max_diff(h_ref, h_C, M * N);
    printf("gemm_wmma_async: M=%d N=%d K=%d, max_err=%.6f — %s\n",
           M, N, K, async_err, async_err < 5e-1f ? "PASS" : "FAIL");

    // --- wmma + register tiling ---
    CHECK_CUDA(cudaMemset(d_C, 0, M * N * sizeof(float)));
    constexpr int RT_TILE = 128;
    constexpr int RT_WARPS = 4;  // 2x2 warps, each 4x4 tiles
    dim3 rt_block(RT_WARPS * 32);
    dim3 rt_grid((N + RT_TILE - 1) / RT_TILE, (M + RT_TILE - 1) / RT_TILE);
    gemm_wmma_regtile_t<RT_TILE, RT_TILE, 4, 4><<<rt_grid, rt_block>>>(
        d_A_half, d_B_half, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    float rt_err = max_diff(h_ref, h_C, M * N);
    printf("gemm_regtile:    M=%d N=%d K=%d, max_err=%.6f — %s\n",
           M, N, K, rt_err, rt_err < 5e-1f ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_half); cudaFree(d_B_half);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    free(h_A_half); free(h_B_half);
    return 0;
}
