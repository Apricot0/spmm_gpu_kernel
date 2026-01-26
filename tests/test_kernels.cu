#include "sparse_kernels.h"
#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

struct KernelConfig {
    bool test_cpu;
    bool test_cublas;
    bool test_nmsparse_baseline;
    bool test_nmsparse_double_buffer;
    bool test_nmsparse_prefetch;
    bool test_nmspmm;
    
    // test all kernels
    KernelConfig() : 
        test_cpu(true),
        test_cublas(true),
        test_nmsparse_baseline(true),
        test_nmsparse_double_buffer(true),
        test_nmsparse_prefetch(true),
        test_nmspmm(true) {}
};

struct BenchmarkResult {
    const char* name;
    float time_ms;
    double tflops;
    bool verified;
    bool ran;
    
    BenchmarkResult() : name(""), time_ms(0), tflops(0), verified(false), ran(false) {}
    BenchmarkResult(const char* n) : name(n), time_ms(0), tflops(0), verified(false), ran(false) {}
};

#define LARGE_MATRIX_THRESHOLD 4096

// Transpose matrix in-place (for layout conversion)
void transpose_matrix(float* mat, int rows, int cols) {
    float* temp = (float*)malloc(sizeof(float) * rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            temp[j * rows + i] = mat[i * cols + j];
        }
    }
    memcpy(mat, temp, sizeof(float) * rows * cols);
    free(temp);
}


void benchmark_cpu_reference(
    float* h_A, float* h_B_dense, float* h_C_cpu,
    int M, int N, int K,
    BenchmarkResult& result,
    bool use_cublas_ref)
{
    printf("\n=== 1. Sequential (CPU) ===\n");
    
    if (use_cublas_ref) {
        printf("Matrix too large (>=%d) - skipping CPU computation.\n", LARGE_MATRIX_THRESHOLD);
        printf("cuBLAS will be used as reference for verification.\n");
        result.ran = false;
        result.verified = false;
        return;
    }
    
    printf("Computing CPU reference (dense: A × B_dense)...\n");
    printf("  Note: All GPU kernels should produce this same result\n");
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Dense × dense computation
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i + k * M] * h_B_dense[k * N + j];
            }
            h_C_cpu[i * N + j] = sum;
        }
    }
    
    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    result.time_ms = duration.count();
    
    // TFLOPS
    double dense_flops = 2.0 * M * N * K;
    result.tflops = (dense_flops / 1e12) / (result.time_ms / 1e3);
    
    printf("  Time: %.3f ms\n", result.time_ms);
    printf("  Performance: %.2f GFLOPS\n", result.tflops * 1000);
    
    result.ran = true;
    result.verified = true;  // CPU is the reference
    printf("CPU reference computed.\n");
}

void benchmark_cublas(
    float* d_A, float* d_B_dense, float* d_C_col_major,
    float* h_C_col_major, float* h_C_cpu,
    int M, int N, int K, int warmup, int iterations,
    BenchmarkResult& result,
    bool use_cublas_ref)
{
    printf("\n=== 2. cuBLAS Dense GEMM ===\n");
    CUDA_CHECK(cudaMemset(d_C_col_major, 0, sizeof(float) * M * N));
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        cublas_dense_gemm(d_A, d_B_dense, d_C_col_major, M, N, K);
        CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        cublas_dense_gemm(d_A, d_B_dense, d_C_col_major, M, N, K);
        CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&result.time_ms, start, stop));
    result.time_ms /= iterations;
    
    double dense_flops = 2.0 * M * N * K;
    result.tflops = (dense_flops / 1e12) / (result.time_ms / 1e3);
    
    printf("  Time: %.3f ms\n", result.time_ms);
    printf("  Performance: %.2f TFLOPS\n", result.tflops);
    
    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_col_major, d_C_col_major, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    transpose_matrix(h_C_col_major, M, N);
    
    if (use_cublas_ref) {
        // For large matrices, cuBLAS IS the reference
        printf("  Using cuBLAS as reference (matrix size >= %d)\n", LARGE_MATRIX_THRESHOLD);
        memcpy(h_C_cpu, h_C_col_major, sizeof(float) * M * N);
        result.verified = true;  // cuBLAS is the reference
    } else {
        // For small matrices, verify against CPU
        printf("  Verification: ");
        result.verified = verify_result(h_C_col_major, h_C_cpu, M, N);
    }
    result.ran = true;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void benchmark_nmsparse_baseline(
    float* d_A, float* d_B, int* d_B_idx, float* d_C_col_major,
    float* h_C_col_major, float* h_C_cpu,
    int M, int N, int K, int W, int warmup, int iterations,
    BenchmarkResult& result)
{
    printf("\n=== 3. nmSparse (simple baseline) ===\n");
    CUDA_CHECK(cudaMemset(d_C_col_major, 0, sizeof(float) * M * N));
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        nmsparse_baseline(d_A, d_B, d_B_idx, d_C_col_major, M, N, K, W);
        CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        nmsparse_baseline(d_A, d_B, d_B_idx, d_C_col_major, M, N, K, W);
        CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&result.time_ms, start, stop));
    result.time_ms /= iterations;
    
    double sparse_flops = 2.0 * M * N * W;
    result.tflops = (sparse_flops / 1e12) / (result.time_ms / 1e3);
    
    printf("  Time: %.3f ms\n", result.time_ms);
    printf("  Performance: %.2f TFLOPS (effective)\n", result.tflops);
    
    // Verify result (transpose from column-major to row-major)
    CUDA_CHECK(cudaMemcpy(h_C_col_major, d_C_col_major, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    transpose_matrix(h_C_col_major, M, N);
    printf("  Verification: ");
    result.verified = verify_result(h_C_col_major, h_C_cpu, M, N);
    result.ran = true;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void benchmark_nmsparse_double_buffer(
    float* d_A, float* d_B, int* d_B_idx, float* d_C_col_major,
    float* h_C_col_major, float* h_C_cpu,
    int M, int N, int K, int W, int warmup, int iterations,
    BenchmarkResult& result)
{
    printf("\n=== 4. nmSparse Double Buffer ===\n");
    
    // Warmup - NOTE: double buffer uses SPLIT_K with atomicAdd, must zero before EACH call
    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaMemset(d_C_col_major, 0, sizeof(float) * M * N));
        nmsparse_double_buffer(d_A, d_B, d_B_idx, d_C_col_major, M, N, K, W);
        CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CUDA_CHECK(cudaMemset(d_C_col_major, 0, sizeof(float) * M * N));
        nmsparse_double_buffer(d_A, d_B, d_B_idx, d_C_col_major, M, N, K, W);
        CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&result.time_ms, start, stop));
    result.time_ms /= iterations;
    
    double sparse_flops = 2.0 * M * N * W;
    result.tflops = (sparse_flops / 1e12) / (result.time_ms / 1e3);
    
    printf("  Time: %.3f ms\n", result.time_ms);
    printf("  Performance: %.2f TFLOPS (effective)\n", result.tflops);
    
    // Verify result (transpose from column-major to row-major)
    CUDA_CHECK(cudaMemcpy(h_C_col_major, d_C_col_major, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    transpose_matrix(h_C_col_major, M, N);
    printf("  Verification: ");
    result.verified = verify_result(h_C_col_major, h_C_cpu, M, N);
    result.ran = true;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void benchmark_nmsparse_prefetch(
    float* d_A, float* d_B, int* d_B_idx, float* d_C_col_major,
    float* h_C_col_major, float* h_C_cpu,
    int M, int N, int K, int W, int warmup, int iterations,
    BenchmarkResult& result)
{
    printf("\n=== 5. nmSparse Prefetch ===\n");
    CUDA_CHECK(cudaMemset(d_C_col_major, 0, sizeof(float) * M * N));
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        nmsparse_prefetch(d_A, d_B, d_B_idx, d_C_col_major, M, N, K, W);
        CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        nmsparse_prefetch(d_A, d_B, d_B_idx, d_C_col_major, M, N, K, W);
        CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&result.time_ms, start, stop));
    result.time_ms /= iterations;
    
    double sparse_flops = 2.0 * M * N * W;
    result.tflops = (sparse_flops / 1e12) / (result.time_ms / 1e3);
    
    printf("  Time: %.3f ms\n", result.time_ms);
    printf("  Performance: %.2f TFLOPS (effective)\n", result.tflops);
    
    // Verify result (transpose from column-major to row-major)
    CUDA_CHECK(cudaMemcpy(h_C_col_major, d_C_col_major, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    transpose_matrix(h_C_col_major, M, N);
    printf("  Verification: ");
    result.verified = verify_result(h_C_col_major, h_C_cpu, M, N);
    result.ran = true;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void benchmark_nmspmm(
    float* d_A, float* d_B, int* d_D, float* d_C,
    float* h_C_gpu, float* h_C_cpu,
    int M, int N, int K, int W, int warmup, int iterations,
    BenchmarkResult& result)
{
    printf("\n=== 6. NM-SpMM (optimized) ===\n");
    CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * M * N));
    
    // Warmup
    for (int i = 0; i < warmup; i++) {
        nmspmm_kernel_optimized(d_A, d_B, d_D, d_C, M, N, K, W);
        CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        nmspmm_kernel_optimized(d_A, d_B, d_D, d_C, M, N, K, W);
        CUDA_CHECK(cudaGetLastError());  // Check for launch errors
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&result.time_ms, start, stop));
    result.time_ms /= iterations;
    
    double sparse_flops = 2.0 * M * N * W;
    result.tflops = (sparse_flops / 1e12) / (result.time_ms / 1e3);
    
    printf("  Time: %.3f ms\n", result.time_ms);
    printf("  Performance: %.2f TFLOPS (effective)\n", result.tflops);
    
    // Verify result
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    printf("  Verification: ");
    result.verified = verify_result(h_C_gpu, h_C_cpu, M, N);
    result.ran = true;
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void print_performance_summary(
    const KernelConfig& config,
    const BenchmarkResult& cpu,
    const BenchmarkResult& cublas,
    const BenchmarkResult& nmsparse,
    const BenchmarkResult& nmsparse_db,
    const BenchmarkResult& nmsparse_pf,
    const BenchmarkResult& nmspmm,
    int warmup, int iterations)
{
    printf("\n=== Performance Summary ===\n");
    printf("Warmup iterations: %d, Test iterations: %d\n\n", warmup, iterations);
    
    printf("Implementation              Time(ms)  TFLOPS   Speedup(CPU)  Speedup(cuBLAS)\n");
    printf("--------------------------------------------------------------------------------\n");
    if (cpu.ran) {
        printf("1. Sequential (CPU)         %.3f     %.2f     1.00x         ", 
               cpu.time_ms, cpu.tflops);
        if (cublas.ran) {
            printf("%.2fx\n", cpu.time_ms / cublas.time_ms);
        } else {
            printf("N/A\n");
        }
    } else {
        printf("1. Sequential (CPU)         N/A       N/A      baseline      baseline\n");
    }
    
    if (cublas.ran) {
        printf("2. cuBLAS (dense)           %.3f     %.2f     ", 
               cublas.time_ms, cublas.tflops);
        if (cpu.ran) {
            printf("%.2fx         ", cpu.time_ms / cublas.time_ms);
        } else {
            printf("N/A           ");
        }
        printf("1.00x\n");
    } else {
        printf("2. cuBLAS (dense)           SKIPPED\n");
    }
    
    if (nmsparse.ran) {
        printf("3. nmSparse (simple)        %.3f     %.2f     ", 
               nmsparse.time_ms, nmsparse.tflops);
        if (cpu.ran) {
            printf("%.2fx         ", cpu.time_ms / nmsparse.time_ms);
        } else {
            printf("N/A           ");
        }
        if (cublas.ran) {
            printf("%.2fx\n", cublas.time_ms / nmsparse.time_ms);
        } else {
            printf("N/A\n");
        }
    } else {
        printf("3. nmSparse (simple)        SKIPPED\n");
    }
    
    if (nmsparse_db.ran) {
        printf("4. nmSparse DblBuf          %.3f     %.2f     ", 
               nmsparse_db.time_ms, nmsparse_db.tflops);
        if (cpu.ran) {
            printf("%.2fx         ", cpu.time_ms / nmsparse_db.time_ms);
        } else {
            printf("N/A           ");
        }
        if (cublas.ran) {
            printf("%.2fx\n", cublas.time_ms / nmsparse_db.time_ms);
        } else {
            printf("N/A\n");
        }
    } else {
        printf("4. nmSparse DblBuf          SKIPPED\n");
    }
    
    if (nmsparse_pf.ran) {
        printf("5. nmSparse Prefetch        %.3f     %.2f     ", 
               nmsparse_pf.time_ms, nmsparse_pf.tflops);
        if (cpu.ran) {
            printf("%.2fx         ", cpu.time_ms / nmsparse_pf.time_ms);
        } else {
            printf("N/A           ");
        }
        if (cublas.ran) {
            printf("%.2fx\n", cublas.time_ms / nmsparse_pf.time_ms);
        } else {
            printf("N/A\n");
        }
    } else {
        printf("5. nmSparse Prefetch        SKIPPED\n");
    }
    
    if (nmspmm.ran) {
        printf("6. NM-SpMM (optimized)      %.3f     %.2f     ", 
               nmspmm.time_ms, nmspmm.tflops);
        if (cpu.ran) {
            printf("%.2fx         ", cpu.time_ms / nmspmm.time_ms);
        } else {
            printf("N/A           ");
        }
        if (cublas.ran) {
            printf("%.2fx\n", cublas.time_ms / nmspmm.time_ms);
        } else {
            printf("N/A\n");
        }
    } else {
        printf("6. NM-SpMM (optimized)      SKIPPED\n");
    }
    
    // Sparse kernels comparison
    if (nmsparse.ran && (nmsparse_db.ran || nmsparse_pf.ran || nmspmm.ran)) {
        printf("\nSparse vs Sparse:\n");
        if (nmsparse_db.ran) {
            printf("  nmSparse DblBuf vs nmSparse:   %.2fx %s\n", 
                   nmsparse.time_ms / nmsparse_db.time_ms,
                   (nmsparse_db.time_ms < nmsparse.time_ms) ? "faster" : "slower");
        }
        if (nmsparse_pf.ran) {
            printf("  nmSparse Prefetch vs nmSparse: %.2fx %s\n", 
                   nmsparse.time_ms / nmsparse_pf.time_ms,
                   (nmsparse_pf.time_ms < nmsparse.time_ms) ? "faster" : "slower");
        }
        if (nmspmm.ran) {
            printf("  NM-SpMM vs nmSparse:           %.2fx %s\n", 
                   nmsparse.time_ms / nmspmm.time_ms,
                   (nmspmm.time_ms < nmsparse.time_ms) ? "faster" : "slower");
        }
    }
    
    // Efficiency analysis (vs cuBLAS accounting for 50% sparsity)
    if (cublas.ran) {
        printf("\nEfficiency (vs cuBLAS accounting for 50%% sparsity):\n");
        if (nmsparse.ran) {
            printf("  nmSparse:          %.1f%%\n", 
                   (nmsparse.tflops / (cublas.tflops * 0.5)) * 100);
        }
        if (nmsparse_db.ran) {
            printf("  nmSparse DblBuf:   %.1f%%\n", 
                   (nmsparse_db.tflops / (cublas.tflops * 0.5)) * 100);
        }
        if (nmsparse_pf.ran) {
            printf("  nmSparse Prefetch: %.1f%%\n", 
                   (nmsparse_pf.tflops / (cublas.tflops * 0.5)) * 100);
        }
        if (nmspmm.ran) {
            printf("  NM-SpMM:           %.1f%%\n", 
                   (nmspmm.tflops / (cublas.tflops * 0.5)) * 100);
        }
    }
}

int main(int argc, char** argv) {

    // CONFIGURATION: Edit this section to enable/disable kernels

    KernelConfig config;
    // config.test_cpu = true;                    // Always compute CPU reference
    // config.test_cublas = true;                 // cuBLAS dense baseline
    // config.test_nmsparse_baseline = true;      // Simple nmSparse
    // config.test_nmsparse_double_buffer = true; // nmSparse with double buffering
    // config.test_nmsparse_prefetch = true;      // nmSparse with prefetch
    // config.test_nmspmm = true;                 // Optimized NM-SpMM
    
    // Uncomment to test only specific kernels, for example:
    // config.test_cublas = false;
    // config.test_nmsparse_double_buffer = false;
    config.test_nmspmm = false;
    config.test_cpu = true;
    
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int warmup = 10;
    int iterations = 100;
    
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 6) {
        warmup = atoi(argv[4]);
        iterations = atoi(argv[5]);
    }
    
    const int W = K / 2;  // 50% sparsity
    const int Q = (N + VEC_LEN - 1) / VEC_LEN;
    const float sparsity = 0.5f;
    
    printf("=== N:M Sparse Matrix Multiplication Test ===\n");
    printf("Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Sparsity: %.0f%% (W=%d, Q=%d)\n", sparsity * 100, W, Q);
    printf("Target GPU: sm_89 (RTX 4090)\n\n");
    
    printf("Enabled tests:\n");
    if (config.test_cpu) printf("  ✓ 1. Sequential (CPU)\n");
    if (config.test_cublas) printf("  ✓ 2. cuBLAS (dense baseline)\n");
    if (config.test_nmsparse_baseline) printf("  ✓ 3. nmSparse (simple sparse baseline)\n");
    if (config.test_nmsparse_double_buffer) printf("  ✓ 4. nmSparse Double Buffer\n");
    if (config.test_nmsparse_prefetch) printf("  ✓ 5. nmSparse Prefetch\n");  
    if (config.test_nmspmm) printf("  ✓ 6. NM-SpMM (optimized sparse)\n");
    printf("\n");
    
    size_t A_bytes = sizeof(float) * M * K;
    size_t B_bytes = sizeof(float) * W * N;
    size_t B_idx_bytes = sizeof(int) * W * N;
    size_t D_bytes = sizeof(int) * W * Q;
    size_t C_bytes = sizeof(float) * M * N;
   
    float* h_A = (float*)malloc(A_bytes);
    float* h_B = (float*)malloc(B_bytes);
    int* h_B_idx = (int*)malloc(B_idx_bytes);
    int* h_D = (int*)malloc(D_bytes);
    float* h_C_cpu = (float*)malloc(C_bytes);
    float* h_C_gpu = (float*)malloc(C_bytes);
    float* h_C_col_major = (float*)malloc(C_bytes);
    
    printf("Initializing data...\n");
    init_data_50_sparsity(h_A, h_B, h_D, h_C_cpu, M, N, K);
    
    // Convert D to B_idx 
    const int Ns = 32;
    const int Qs = Ns / VEC_LEN;
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < N; j++) {
            int q = j / VEC_LEN;
            int q_block = q / Qs;
            int q_offset = q % Qs;
            h_B_idx[i * N + j] = h_D[q_block * W * Qs + i * Qs + q_offset];
        }
    }
    
    float* h_B_dense = (float*)malloc(sizeof(float) * K * N);
    memset(h_B_dense, 0, sizeof(float) * K * N);
    
    // Create row-major dense B from sparse representation
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < N; j++) {
            int dense_row = h_B_idx[i * N + j];
            if (dense_row >= 0 && dense_row < K) {
                h_B_dense[dense_row * N + j] = h_B[i * N + j];
            }
        }
    }
    
    float *d_A, *d_B, *d_B_dense, *d_C, *d_C_col_major;
    int *d_B_idx, *d_D;
    CUDA_CHECK(cudaMalloc(&d_A, A_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, B_bytes));
    CUDA_CHECK(cudaMalloc(&d_B_dense, sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc(&d_B_idx, B_idx_bytes));
    CUDA_CHECK(cudaMalloc(&d_D, D_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, C_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_col_major, C_bytes));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, A_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, B_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_idx, h_B_idx, B_idx_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D, h_D, D_bytes, cudaMemcpyHostToDevice));
    
    // Create column-major version of dense B for cuBLAS
    
    float* h_B_dense_col = (float*)malloc(sizeof(float) * K * N);
    
    // Transpose row-major h_B_dense to column-major for cuBLAS
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B_dense_col[i + j * K] = h_B_dense[i * N + j];
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_B_dense, h_B_dense_col, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    
    BenchmarkResult result_cpu("CPU");
    BenchmarkResult result_cublas("cuBLAS");
    BenchmarkResult result_nmsparse("nmSparse");
    BenchmarkResult result_nmsparse_db("nmSparse DblBuf");
    BenchmarkResult result_nmsparse_pf("nmSparse Prefetch");
    BenchmarkResult result_nmspmm("NM-SpMM");
    
    //determine if matrix is too large for CPU reference
    bool use_cublas_ref = (M >= LARGE_MATRIX_THRESHOLD || 
                           N >= LARGE_MATRIX_THRESHOLD || 
                           K >= LARGE_MATRIX_THRESHOLD);
    
    if (use_cublas_ref) {
        printf("\n=== LARGE MATRIX DETECTED ===\n");
        printf("Matrix dimensions exceed threshold (%d)\n", LARGE_MATRIX_THRESHOLD);
        printf("Using cuBLAS result as reference for correctness checking.\n");
        printf("=====================================\n");
    }
    
    // CPU reference (skip for large matrices)
    if (config.test_cpu) {
        benchmark_cpu_reference(h_A, h_B_dense, h_C_cpu, M, N, K, result_cpu, use_cublas_ref);
    }
    
    // GPU benchmarks 
    if (config.test_cublas) {
        benchmark_cublas(d_A, d_B_dense, d_C_col_major, h_C_col_major, h_C_cpu,
                        M, N, K, warmup, iterations, result_cublas, use_cublas_ref);
    } else if (use_cublas_ref) {
        printf("\n!!! ERROR: cuBLAS must be enabled for large matrices (serves as reference) !!!\n");
        printf("Please enable config.test_cublas = true\n");
        return 1;
    }
    
    if (config.test_nmsparse_baseline) {
        benchmark_nmsparse_baseline(d_A, d_B, d_B_idx, d_C_col_major, h_C_col_major, h_C_cpu,
                                   M, N, K, W, warmup, iterations, result_nmsparse);
    }
    
    if (config.test_nmsparse_double_buffer) {
        benchmark_nmsparse_double_buffer(d_A, d_B, d_B_idx, d_C_col_major, h_C_col_major, h_C_cpu,
                                        M, N, K, W, warmup, iterations, result_nmsparse_db);
    }
    
    if (config.test_nmsparse_prefetch) {
        benchmark_nmsparse_prefetch(d_A, d_B, d_B_idx, d_C_col_major, h_C_col_major, h_C_cpu,
                                   M, N, K, W, warmup, iterations, result_nmsparse_pf);
    }
    
    if (config.test_nmspmm) {
        benchmark_nmspmm(d_A, d_B, d_D, d_C, h_C_gpu, h_C_cpu,
                        M, N, K, W, warmup, iterations, result_nmspmm);
    }
    
    print_performance_summary(config, result_cpu, result_cublas, result_nmsparse, 
                            result_nmsparse_db, result_nmsparse_pf, result_nmspmm,
                            warmup, iterations);
    
    free(h_A);
    free(h_B);
    free(h_B_idx);
    free(h_D);
    free(h_B_dense);
    free(h_B_dense_col);
    free(h_C_cpu);
    free(h_C_gpu);
    free(h_C_col_major);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_B_dense));
    CUDA_CHECK(cudaFree(d_B_idx));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_col_major));
    
    printf("\n=== Test completed successfully! ===\n");
    return 0;
}
