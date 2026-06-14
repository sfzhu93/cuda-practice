#pragma once

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
// Naive GEMM (fp32): one thread per output element
// ============================================================
constexpr int TILE = 16;

__global__ void gemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// ============================================================
// Tiled GEMM with shared memory (fp32)
// ============================================================
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
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ============================================================
// WMMA GEMM: Tensor Core, 1 MMA per warp per K step
// ============================================================
template <int BLOCK_TILE>
__global__ void gemm_wmma_t(const half* A, const half* B, float* C,
                             int M, int N, int K) {
    constexpr int WM = 16;
    constexpr int WARPS_N = BLOCK_TILE / WM;

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / WARPS_N;
    int warpCol = warpId % WARPS_N;

    int block_row = blockIdx.y * BLOCK_TILE + warpRow * WM;
    int block_col = blockIdx.x * BLOCK_TILE + warpCol * WM;

    wmma::fragment<wmma::matrix_a, WM, WM, WM, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WM, WM, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WM, WM, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int t = 0; t < K; t += WM) {
        wmma::load_matrix_sync(a_frag, A + block_row * K + t, K);
        wmma::load_matrix_sync(b_frag, B + t * N + block_col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    if (block_row < M && block_col < N)
        wmma::store_matrix_sync(C + block_row * N + block_col, c_frag, N, wmma::mem_row_major);
}

// ============================================================
// WMMA + cp.async double buffer: async global->shared copy
// ============================================================
template <int BLOCK_TILE>
__global__ void gemm_wmma_async_t(const half* A, const half* B, float* C,
                                    int M, int N, int K) {
    constexpr int WM = 16;
    constexpr int WARPS_N = BLOCK_TILE / WM;
    constexpr int A_PAIRS = (BLOCK_TILE * WM) / 2;
    constexpr int B_PAIRS = (WM * BLOCK_TILE) / 2;

    __shared__ half As[2][BLOCK_TILE * WM];
    __shared__ half Bs[2][WM * BLOCK_TILE];

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / WARPS_N;
    int warpCol = warpId % WARPS_N;
    int block_row = blockIdx.y * BLOCK_TILE;
    int block_col = blockIdx.x * BLOCK_TILE;
    int nThreads = blockDim.x;

    wmma::fragment<wmma::matrix_a, WM, WM, WM, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WM, WM, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WM, WM, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int numTiles = K / WM;

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
// WMMA + cp.async + double buffer + register tiling
// Each warp computes WARP_TILE_M x WARP_TILE_N grid of 16x16 tiles
// ============================================================
template <int BLOCK_M, int BLOCK_N, int WARP_TILE_M, int WARP_TILE_N>
__global__ void gemm_wmma_regtile_t(const half* A, const half* B, float* C,
                                     int M, int N, int K) {
    constexpr int WM = 16;
    constexpr int WARPS_M = BLOCK_M / (WARP_TILE_M * WM);
    constexpr int WARPS_N = BLOCK_N / (WARP_TILE_N * WM);
    constexpr int NUM_WARPS = WARPS_M * WARPS_N;
    constexpr int NUM_THREADS = NUM_WARPS * 32;
    constexpr int A_PAIRS = (BLOCK_M * WM) / 2;
    constexpr int B_PAIRS = (WM * BLOCK_N) / 2;

    __shared__ half As[2][BLOCK_M * WM];
    __shared__ half Bs[2][WM * BLOCK_N];

    int warpId = threadIdx.x / 32;
    int warpRow = warpId / WARPS_N;
    int warpCol = warpId % WARPS_N;
    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    wmma::fragment<wmma::accumulator, WM, WM, WM, float> c_frag[WARP_TILE_M][WARP_TILE_N];
    for (int i = 0; i < WARP_TILE_M; i++)
        for (int j = 0; j < WARP_TILE_N; j++)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    int numTiles = K / WM;

    for (int i = threadIdx.x; i < A_PAIRS; i += NUM_THREADS) {
        int r = (i * 2) / WM;
        int c = (i * 2) % WM;
        __pipeline_memcpy_async(&As[0][r * WM + c],
                                &A[(block_row + r) * K + c],
                                sizeof(uint32_t));
    }
    for (int i = threadIdx.x; i < B_PAIRS; i += NUM_THREADS) {
        int r = (i * 2) / BLOCK_N;
        int c = (i * 2) % BLOCK_N;
        __pipeline_memcpy_async(&Bs[0][r * BLOCK_N + c],
                                &B[r * N + block_col + c],
                                sizeof(uint32_t));
    }
    __pipeline_commit();

    for (int t = 0; t < numTiles; t++) {
        int cur = t % 2;
        int nxt = 1 - cur;

        if (t + 1 < numTiles) {
            int kt = (t + 1) * WM;
            for (int i = threadIdx.x; i < A_PAIRS; i += NUM_THREADS) {
                int r = (i * 2) / WM;
                int c = (i * 2) % WM;
                __pipeline_memcpy_async(&As[nxt][r * WM + c],
                                        &A[(block_row + r) * K + kt + c],
                                        sizeof(uint32_t));
            }
            for (int i = threadIdx.x; i < B_PAIRS; i += NUM_THREADS) {
                int r = (i * 2) / BLOCK_N;
                int c = (i * 2) % BLOCK_N;
                __pipeline_memcpy_async(&Bs[nxt][r * BLOCK_N + c],
                                        &B[(kt + r) * N + block_col + c],
                                        sizeof(uint32_t));
            }
            __pipeline_commit();
            __pipeline_wait_prior(1);
        } else {
            __pipeline_wait_prior(0);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WM, WM, WM, half, wmma::row_major> a_frag[WARP_TILE_M];
        wmma::fragment<wmma::matrix_b, WM, WM, WM, half, wmma::row_major> b_frag[WARP_TILE_N];

        for (int i = 0; i < WARP_TILE_M; i++) {
            int a_row = (warpRow * WARP_TILE_M + i) * WM;
            wmma::load_matrix_sync(a_frag[i], &As[cur][a_row * WM], WM);
        }
        for (int j = 0; j < WARP_TILE_N; j++) {
            int b_col = (warpCol * WARP_TILE_N + j) * WM;
            wmma::load_matrix_sync(b_frag[j], &Bs[cur][b_col], BLOCK_N);
        }

        for (int i = 0; i < WARP_TILE_M; i++)
            for (int j = 0; j < WARP_TILE_N; j++)
                wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);

        __syncthreads();
    }

    for (int i = 0; i < WARP_TILE_M; i++) {
        for (int j = 0; j < WARP_TILE_N; j++) {
            int out_row = block_row + (warpRow * WARP_TILE_M + i) * WM;
            int out_col = block_col + (warpCol * WARP_TILE_N + j) * WM;
            if (out_row < M && out_col < N)
                wmma::store_matrix_sync(C + out_row * N + out_col, c_frag[i][j], N,
                                        wmma::mem_row_major);
        }
    }
}
