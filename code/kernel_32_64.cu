#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include "cuda.cuh"


// cal offset from row col && ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// transfer float4
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void cuda_gemm_32_64(int M, int N, int K, float alpha, float *A, float *B,
                 float beta, float *C) {
    static const int BM = 32;
    static const int BN = 64;
    static const int BK = 16;
    static const int WM = 16;
    static const int WN = 32;
    static const int WMITER = 8;
    static const int WNITER = 32;
    static const int TM = 2;
    static const int TN = 4;

    dim3 threadsPerBlock((BM*BN)/(WM*WN)*32);
    dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    gemm<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>
        <<<blocksPerGrid, threadsPerBlock>>>(M, N, K, alpha, A, B, beta, C);      
}