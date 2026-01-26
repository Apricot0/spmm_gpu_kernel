#include "sparse_kernels.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <iostream>

#define CHECK_CUBLAS(Expr)                                         \
    {                                                              \
        int err = (Expr);                                          \
        if (err != 0) {                                            \
            printf("cuBLAS error %d at line %d\n", err, __LINE__); \
        }                                                          \
    }

// Internal GEMM function wrapper
static void gemm_internal(cublasHandle_t handle,
    int m,
    int n,
    int k,
    const void* alpha,
    const void* beta,
    cudaDataType_t input_type,
    const void* A,
    const void* B,
    cudaDataType_t output_type,
    void* C,
#if __CUDACC_VER_MAJOR__ >= 11
    cublasComputeType_t compute_type,
#else
    cudaDataType_t compute_type,
#endif
    int algo)
{
    // cuBLAS uses column-major: C = alpha*A*B + beta*C
    // Parameters: (handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    // For C(M×N) = A(M×K) * B(K×N), all column-major:
    //   m=M (rows of A and C), n=N (cols of B and C), k=K (cols of A, rows of B)
    //   A: M×K with leading dimension M
    //   B: K×N with leading dimension K
    //   C: M×N with leading dimension M
    cublasStatus_t res = cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
        alpha, A, input_type, m, B, input_type, k,
        beta, C, output_type, m, compute_type, static_cast<cublasGemmAlgo_t>(algo));
    CHECK_CUBLAS(res);
}

// Public API: cuBLAS dense GEMM for comparison with sparse kernels
// Note: This matches the original cuBLAS layout from the benchmark
void cublas_dense_gemm(
    float* A,           // Dense matrix A (M×K), column-major
    float* B,           // Dense matrix B (K×N), column-major (need to convert from row-major)
    float* C,           // Output matrix C (M×N), column-major (will be in column-major, need to convert)
    int M, int N, int K
)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaDataType_t input_type = CUDA_R_32F;
    cudaDataType_t output_type = CUDA_R_32F;
#if __CUDACC_VER_MAJOR__ >= 11
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#else
    cudaDataType_t compute_type = CUDA_R_32F;
#endif

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Use default algorithm for simplicity
    int algo = CUBLAS_GEMM_DEFAULT;

    // Original cuBLAS call uses (m, n, k) -> (N, M, K) with B, A order
    // This computes C = A * B where A is M×K column-major, B is K×N column-major
    // The gemm_internal expects (n, m, k) and (B, A) for the column-major layout
    gemm_internal(handle, M, N, K, &alpha, &beta, input_type, A, B,
        output_type, C, compute_type, algo);

    cublasDestroy(handle);
}

// Benchmarking version with timing and algorithm selection
double cublas_dense_gemm_benchmark(
    float* A,           // Dense matrix A (M×K), column-major
    float* B,           // Dense matrix B (K×N), row-major
    float* C,           // Output matrix C (M×N), row-major
    int M, int N, int K,
    int warmup_iters,
    int test_iters
)
{
    float alpha = 1.0f;
    float beta = 0.0f;

    cudaDataType_t input_type = CUDA_R_32F;
    cudaDataType_t output_type = CUDA_R_32F;
#if __CUDACC_VER_MAJOR__ >= 11
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#else
    cudaDataType_t compute_type = CUDA_R_32F;
#endif

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Search for best algorithm
    double best_time_ms = 1e10;
    int start_algo = CUBLAS_GEMM_DEFAULT;
    int end_algo = CUBLAS_GEMM_DEFAULT;
    
    for (int algo = start_algo; algo <= end_algo; ++algo) {
        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            gemm_internal(handle, M, N, K, &alpha, &beta, input_type, A, B,
                output_type, C, compute_type, algo);
        }
        
        cudaEventRecord(start);
        for (int i = 0; i < test_iters; ++i) {
            gemm_internal(handle, M, N, K, &alpha, &beta, input_type, A, B,
                output_type, C, compute_type, algo);
        }
        cudaEventRecord(stop);

        float time_ms = 0.f;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_ms, start, stop);
        
        double avg_time = (double)time_ms / test_iters;
        if (avg_time < best_time_ms) {
            best_time_ms = avg_time;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);

    return best_time_ms;
}
