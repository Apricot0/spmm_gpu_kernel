#ifndef SPARSE_KERNELS_H
#define SPARSE_KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>

//vector length for alignment and coalescing
#define VEC_LEN 32

// vectorized loads
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])

// nmSparse baseline 
void nmsparse_baseline(
    float* A,           // Dense matrix A (M×K), column-major
    float* B,           // Sparse values B (W×N), row-major
    int* B_idx,         // Sparse indices (W×N), row-major
    float* C,           // Output matrix C (M×N), column-major
    int M, int N, int K, int W  // Dimensions, W = K * (1 - sparsity)
);

// nmSparse with double buffering 
void nmsparse_double_buffer(
    float* A,           // Dense matrix A (M×K), column-major
    float* B,           // Sparse values B (W×N), row-major
    int* B_idx,         // Sparse indices (W×N), row-major
    float* C,           // Output matrix C (M×N), column-major
    int M, int N, int K, int W  // Dimensions, W = K * (1 - sparsity)
);

// nmSparse with single-stage prefetch 
void nmsparse_prefetch(
    float* A,           // Dense matrix A (M×K), column-major
    float* B,           // Sparse values B (W×N), row-major
    int* B_idx,         // Sparse indices (W×N), row-major
    float* C,           // Output matrix C (M×N), column-major
    int M, int N, int K, int W  // Dimensions, W = K * (1 - sparsity)
);

// N:M SpMM kernel 
void nmspmm_kernel_optimized(
    float* A,           // Dense matrix A (M×K), column-major
    float* B,           // Sparse values of B (W×N), row-major
    int* D,             // Indices for sparse B (W×Q), row-major, Q = N/VEC_LEN
    float* C,           // Output matrix C (M×N), row-major
    int M, int N, int K, int W  // Dimensions, W = K * (1 - sparsity)
);

// cuBLAS dense GEMM for comparison
void cublas_dense_gemm(
    float* A,           // Dense matrix A (M×K), column-major
    float* B,           // Dense matrix B (K×N), row-major
    float* C,           // Output matrix C (M×N), row-major
    int M, int N, int K
);

// cuBLAS dense GEMM with benchmarking
double cublas_dense_gemm_benchmark(
    float* A,           // Dense matrix A (M×K), column-major
    float* B,           // Dense matrix B (K×N), row-major
    float* C,           // Output matrix C (M×N), row-major
    int M, int N, int K,
    int warmup_iters,
    int test_iters
);

void init_data_50_sparsity(
    float* A, float* B, int* B_idx, 
    float* C, int M, int N, int K
);

void cpu_reference(
    float* A, float* B, int* B_idx, 
    float* C, int M, int N, int K, int W
);

bool verify_result(float* C_gpu, float* C_cpu, int M, int N);

#endif // SPARSE_KERNELS_H
