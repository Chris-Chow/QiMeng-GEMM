
#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// transfer float4
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])



template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int TM, const int TN>
__global__ void gemm(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
    __shared__ float As[2][BK][BM];  // Double buffer, transposed
    __shared__ float Bs[2][BK][BN];  // Double buffer

    const int thread_num = BM * BN / (WM * WN) * 32;
    const int hightA = thread_num / (BK / 4);
    const int hightB = thread_num / (BN / 4);

    // Calculate the load indices for shared memory
    int load_a_smem_m = threadIdx.x / (BK / 4);
    int load_a_smem_k = threadIdx.x % (BK / 4) * 4;
    int load_b_smem_k = threadIdx.x / (BN / 4);
    int load_b_smem_n = threadIdx.x % (BN / 4) * 4;

    const int wid = threadIdx.x / 32;
    const int tid = threadIdx.x % 32;
    const int Wrow = wid / (BN / WN);
    const int Wcol = wid % (BN / WN);
    const int Trow = tid / (WNITER / TN);
    const int Tcol = tid % (WNITER / TN);

    const int A_offset = blockIdx.y * BM * K;
    const int B_offset = blockIdx.x * BN;

    A += A_offset;
    B += B_offset;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    float results[WM / WMITER * TM][WN / WNITER * TN] = {0};
    float regM[TM] = {0};
    float regN[TN] = {0};

    const int A_row_stride = hightA;
    const int B_row_stride = hightB;

    // Moving numbers phase 0
    for (int loadOffset = 0; loadOffset < BM; loadOffset += A_row_stride) {
        float4 tmp = FLOAT4(A[(load_a_smem_m + loadOffset) * K + load_a_smem_k]);
        As[0][load_a_smem_k][load_a_smem_m+loadOffset] = tmp.x;
        As[0][load_a_smem_k+1][load_a_smem_m+loadOffset] = tmp.y;
        As[0][load_a_smem_k+2][load_a_smem_m+loadOffset] = tmp.z;
        As[0][load_a_smem_k+3][load_a_smem_m+loadOffset] = tmp.w;
    }
    for (int loadOffset = 0; loadOffset < BK; loadOffset += B_row_stride) {
        FLOAT4(Bs[0][load_b_smem_k + loadOffset][load_b_smem_n]) = FLOAT4(B[(load_b_smem_k + loadOffset) * N + load_b_smem_n]);
    }
    A += BK;
    B += BK*BN;

    for (int bkIdx = 1; bkIdx < (K + BK - 1) / BK; bkIdx++) {
        __syncthreads();
        int comp_flag = (bkIdx - 1) & 1;
        int mem_flag = bkIdx & 1;

        // Computation phase n-1
        for (int k = 0; k < BK; k++) {
            for (int wm = 0; wm < WM / WMITER; ++wm) {
                for (int wn = 0; wn < WN / WNITER; ++wn) {
                    for (int tm = 0; tm < TM; tm++) {
                        regM[tm] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + tm];
                    }
                    for (int tn = 0; tn < TN; tn++) {
                        regN[tn] = Bs[comp_flag][k][Wcol * WN + wn * WNITER + Tcol * TN + tn];
                    }
                    for (int tm = 0; tm < TM; tm++) {
                        for (int tn = 0; tn < TN; tn++) {
                            results[wm * TM + tm][wn * TN + tn] += regM[tm] * regN[tn];
                        }
                    }
                }
            }
        }

        // Moving numbers phase n
        #pragma unroll
        for (int loadOffset = 0; loadOffset < BM; loadOffset += A_row_stride) {
            float4 tmp = FLOAT4(A[(load_a_smem_m + loadOffset) * K + load_a_smem_k]);
            As[mem_flag][load_a_smem_k][load_a_smem_m+loadOffset] = tmp.x;
            As[mem_flag][load_a_smem_k+1][load_a_smem_m+loadOffset] = tmp.y;
            As[mem_flag][load_a_smem_k+2][load_a_smem_m+loadOffset] = tmp.z;
            As[mem_flag][load_a_smem_k+3][load_a_smem_m+loadOffset] = tmp.w;
        }
        #pragma unroll
        for (int loadOffset = 0; loadOffset < BK; loadOffset += B_row_stride) {
            FLOAT4(Bs[mem_flag][load_b_smem_k + loadOffset][load_b_smem_n]) = FLOAT4(B[(load_b_smem_k + loadOffset) * N + load_b_smem_n]);
        }
        A += BK;
        B += BK*BN;
    }

    // Computation phase n
    int comp_flag = 1;
    for (int k = 0; k < BK; k++) {
        for (int wm = 0; wm < WM / WMITER; ++wm) {
            for (int wn = 0; wn < WN / WNITER; ++wn) {
                for (int tm = 0; tm < TM; tm++) {
                    regM[tm] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + tm];
                }
                for (int tn = 0; tn < TN; tn++) {
                    regN[tn] = Bs[comp_flag][k][Wcol * WN + wn * WNITER + Tcol * TN + tn];
                }
                for (int tm = 0; tm < TM; tm++) {
                    for (int tn = 0; tn < TN; tn++) {
                        results[wm * TM + tm][wn * TN + tn] += regM[tm] * regN[tn];
                    }
                }
            }
        }
    }

    // Store results
    for (int wm = 0; wm < WM / WMITER; ++wm) {
        for (int wn = 0; wn < WN / WNITER; ++wn) {
            for (int tm = 0; tm < TM; tm++) {
                for (int tn = 0; tn < TN; tn++) {
                    int row = Wrow * WM + wm * WMITER + Trow * TM + tm;
                    int col = Wcol * WN + wn * WNITER + Tcol * TN + tn;
                    C[row * N + col] = alpha * results[wm * TM + tm][wn * TN + tn] + beta * C[row * N + col];
                }
            }
        }
    }
}
