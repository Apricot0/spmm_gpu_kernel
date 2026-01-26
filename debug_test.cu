#include "sparse_kernels.h"
#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstring>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void print_matrix(const char* name, float* mat, int rows, int cols, const char* layout) {
    printf("\n%s (%dx%d, %s):\n", name, rows, cols, layout);
    for (int i = 0; i < rows && i < 8; i++) {
        for (int j = 0; j < cols && j < 8; j++) {
            if (strcmp(layout, "row-major") == 0) {
                printf("%7.3f ", mat[i * cols + j]);
            } else { // column-major
                printf("%7.3f ", mat[i + j * rows]);
            }
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
    if (rows > 8) printf("...\n");
}

void print_int_matrix(const char* name, int* mat, int rows, int cols) {
    printf("\n%s (%dx%d, row-major):\n", name, rows, cols);
    for (int i = 0; i < rows && i < 8; i++) {
        for (int j = 0; j < cols && j < 8; j++) {
            printf("%4d ", mat[i * cols + j]);
        }
        if (cols > 8) printf("...");
        printf("\n");
    }
    if (rows > 8) printf("...\n");
}

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

int main() {
    int M = 1024, N = 1024, K = 1024;
    const int W = K / 2;
    const int Q = (N + VEC_LEN - 1) / VEC_LEN;
    
    printf("=== Debug Test (64x64 matrices) ===\n");
    printf("M=%d, N=%d, K=%d, W=%d, Q=%d\n\n", M, N, K, W, Q);
    
    // Allocate memory
    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(W * N * sizeof(float));
    int* h_D = (int*)malloc(W * Q * sizeof(int));
    int* h_B_idx = (int*)malloc(W * N * sizeof(int));
    float* h_C_cpu = (float*)malloc(M * N * sizeof(float));
    
    // Initialize
    printf("Initializing data...\n");
    init_data_50_sparsity(h_A, h_B, h_D, h_C_cpu, M, N, K);
    
    print_matrix("A (column-major)", h_A, M, K, "column-major");
    print_matrix("B sparse values (row-major)", h_B, W, N, "row-major");
    
    // Convert D to B_idx
    // D is in blocked layout: [Q/Qs blocks][W][Qs] where Qs = Ns/VEC_LEN
    // Need to transform to row-major [W][N] for B_idx
    const int Ns = 32;
    const int Qs = Ns / VEC_LEN;
    
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < N; j++) {
            int q = j / VEC_LEN;
            int q_block = q / Qs;
            int q_offset = q % Qs;
            
            // D layout: D[q_block * W * Qs + i * Qs + q_offset]
            h_B_idx[i * N + j] = h_D[q_block * W * Qs + i * Qs + q_offset];
        }
    }
    
    print_int_matrix("B_idx (indices, row-major)", h_B_idx, W, N);
    
    // Reconstruct dense B
    float* h_B_dense = (float*)malloc(K * N * sizeof(float));
    memset(h_B_dense, 0, K * N * sizeof(float));
    
    printf("\nReconstructing dense B from sparse...\n");
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < N; j++) {
            int dense_row = h_B_idx[i * N + j];
            if (dense_row >= 0 && dense_row < K) {
                h_B_dense[dense_row * N + j] = h_B[i * N + j];
                if (i < 4 && j < 4) {
                    printf("  B[%d,%d]=%.3f -> B_dense[%d,%d]\n", 
                           i, j, h_B[i * N + j], dense_row, j);
                }
            }
        }
    }
    
    print_matrix("B_dense (reconstructed, row-major)", h_B_dense, K, N, "row-major");
    
    // CPU reference
    printf("\nComputing CPU reference...\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += h_A[i + k * M] * h_B_dense[k * N + j];
            }
            h_C_cpu[i * N + j] = sum;
        }
    }
    
    print_matrix("C_cpu (row-major)", h_C_cpu, M, N, "row-major");
    
    // cuBLAS test
    printf("\n=== Testing cuBLAS ===\n");
    
    // Convert B to column-major for cuBLAS
    float* h_B_dense_col = (float*)malloc(K * N * sizeof(float));
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B_dense_col[i + j * K] = h_B_dense[i * N + j];
        }
    }
    
    print_matrix("B_dense_col (column-major for cuBLAS)", h_B_dense_col, K, N, "column-major");
    
    // Device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_dense_col, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    
    // Call cuBLAS
    cublas_dense_gemm(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back
    float* h_C_cublas_col = (float*)malloc(M * N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_C_cublas_col, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    print_matrix("C_cublas (cuBLAS output, column-major)", h_C_cublas_col, M, N, "column-major");
    
    // Transpose to row-major
    transpose_matrix(h_C_cublas_col, M, N);
    print_matrix("C_cublas (converted to row-major)", h_C_cublas_col, M, N, "row-major");
    
    // Verify
    printf("\n=== Verification ===\n");
    int errors = 0;
    for (int i = 0; i < M && i < 8; i++) {
        for (int j = 0; j < N && j < 8; j++) {
            float diff = fabs(h_C_cublas_col[i * N + j] - h_C_cpu[i * N + j]);
            if (diff > 1e-3) {
                printf("Mismatch at [%d,%d]: cuBLAS=%.6f, CPU=%.6f, diff=%.6f\n",
                       i, j, h_C_cublas_col[i * N + j], h_C_cpu[i * N + j], diff);
                errors++;
            }
        }
    }
    
    if (errors == 0) {
        printf("✓ All checked elements match!\n");
    } else {
        printf("✗ Found %d mismatches\n", errors);
    }
    
    // ===== Test NM-SpMM Kernel =====
    printf("\n\n=== Testing NM-SpMM Kernel ===\n");
    
    // Need device memory for NM-SpMM test
    float *d_B_sparse, *d_C_nmspmm;
    int *d_D;
    CUDA_CHECK(cudaMalloc(&d_B_sparse, W * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D, W * Q * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_C_nmspmm, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_B_sparse, h_B, W * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D, h_D, W * Q * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C_nmspmm, 0, M * N * sizeof(float)));
    
    printf("Calling nmspmm_kernel_optimized with:\n");
    printf("  M=%d, N=%d, K=%d, W=%d, Q=%d\n", M, N, K, W, Q);
    
    // Call NM-SpMM kernel
    nmspmm_kernel_optimized(d_A, d_B_sparse, d_D, d_C_nmspmm, M, N, K, W);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel executed successfully!\n");
        
        // Copy result back
        float* h_C_nmspmm = (float*)malloc(M * N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_C_nmspmm, d_C_nmspmm, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        
        print_matrix("C_nmspmm (row-major)", h_C_nmspmm, M, N, "row-major");
        
        // Verify
        printf("\n=== Verification (NM-SpMM vs CPU) ===\n");
        int errors = 0;
        for (int i = 0; i < M && i < 8; i++) {
            for (int j = 0; j < N && j < 8; j++) {
                float diff = fabs(h_C_nmspmm[i * N + j] - h_C_cpu[i * N + j]);
                if (diff > 1e-3) {
                    printf("Mismatch at [%d,%d]: NM-SpMM=%.6f, CPU=%.6f, diff=%.6f\n",
                           i, j, h_C_nmspmm[i * N + j], h_C_cpu[i * N + j], diff);
                    errors++;
                }
            }
        }
        
        if (errors == 0) {
            printf("✓ All checked elements match!\n");
        } else {
            printf("✗ Found %d mismatches\n", errors);
        }
        
        free(h_C_nmspmm);
    }
    
    cudaFree(d_B_sparse);
    cudaFree(d_D);
    cudaFree(d_C_nmspmm);
    
    // ===== Test nmSparse Kernels =====
    printf("\n\n=== Testing nmSparse Kernels ===\n");
    
    // Need B_idx on device for nmSparse kernels
    int *d_B_idx;
    float *d_C_col_major;
    CUDA_CHECK(cudaMalloc(&d_B_idx, W * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B_sparse, W * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C_col_major, M * N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_B_idx, h_B_idx, W * N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_sparse, h_B, W * N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Test 1: nmSparse Baseline
    printf("\n--- nmSparse Baseline ---\n");
    CUDA_CHECK(cudaMemset(d_C_col_major, 0, M * N * sizeof(float)));
    
    nmsparse_baseline(d_A, d_B_sparse, d_B_idx, d_C_col_major, M, N, K, W);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Baseline kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Baseline kernel execution error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Baseline kernel executed successfully!\n");
        
        float* h_C_baseline = (float*)malloc(M * N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_C_baseline, d_C_col_major, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Transpose from column-major to row-major
        transpose_matrix(h_C_baseline, M, N);
        print_matrix("C_baseline (row-major)", h_C_baseline, M, N, "row-major");
        
        // Verify
        printf("\n=== Verification (Baseline vs CPU) ===\n");
        errors = 0;
        for (int i = 0; i < M && i < 8; i++) {
            for (int j = 0; j < N && j < 8; j++) {
                float diff = fabs(h_C_baseline[i * N + j] - h_C_cpu[i * N + j]);
                if (diff > 1e-3) {
                    printf("Mismatch at [%d,%d]: Baseline=%.6f, CPU=%.6f, diff=%.6f\n",
                           i, j, h_C_baseline[i * N + j], h_C_cpu[i * N + j], diff);
                    errors++;
                }
            }
        }
        if (errors == 0) printf("✓ Baseline matches CPU!\n");
        else printf("✗ Found %d mismatches\n", errors);
        
        free(h_C_baseline);
    }
    
    // Test 2: nmSparse Double Buffer
    printf("\n--- nmSparse Double Buffer ---\n");
    CUDA_CHECK(cudaMemset(d_C_col_major, 0, M * N * sizeof(float)));
    
    nmsparse_double_buffer(d_A, d_B_sparse, d_B_idx, d_C_col_major, M, N, K, W);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Double buffer kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Double buffer kernel execution error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Double buffer kernel executed successfully!\n");
        
        float* h_C_dblbuf = (float*)malloc(M * N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_C_dblbuf, d_C_col_major, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Transpose from column-major to row-major
        transpose_matrix(h_C_dblbuf, M, N);
        print_matrix("C_double_buffer (row-major)", h_C_dblbuf, M, N, "row-major");
        
        // Verify
        printf("\n=== Verification (Double Buffer vs CPU) ===\n");
        errors = 0;
        for (int i = 0; i < M && i < 8; i++) {
            for (int j = 0; j < N && j < 8; j++) {
                float diff = fabs(h_C_dblbuf[i * N + j] - h_C_cpu[i * N + j]);
                if (diff > 1e-3) {
                    printf("Mismatch at [%d,%d]: DblBuf=%.6f, CPU=%.6f, diff=%.6f\n",
                           i, j, h_C_dblbuf[i * N + j], h_C_cpu[i * N + j], diff);
                    errors++;
                }
            }
        }
        if (errors == 0) printf("✓ Double buffer matches CPU!\n");
        else printf("✗ Found %d mismatches\n", errors);
        
        free(h_C_dblbuf);
    }
    
    // Test 3: nmSparse Prefetch
    printf("\n--- nmSparse Prefetch ---\n");
    CUDA_CHECK(cudaMemset(d_C_col_major, 0, M * N * sizeof(float)));
    
    nmsparse_prefetch(d_A, d_B_sparse, d_B_idx, d_C_col_major, M, N, K, W);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Prefetch kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Prefetch kernel execution error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Prefetch kernel executed successfully!\n");
        
        float* h_C_prefetch = (float*)malloc(M * N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_C_prefetch, d_C_col_major, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Transpose from column-major to row-major
        transpose_matrix(h_C_prefetch, M, N);
        print_matrix("C_prefetch (row-major)", h_C_prefetch, M, N, "row-major");
        
        // Verify
        printf("\n=== Verification (Prefetch vs CPU) ===\n");
        errors = 0;
        for (int i = 0; i < M && i < 8; i++) {
            for (int j = 0; j < N && j < 8; j++) {
                float diff = fabs(h_C_prefetch[i * N + j] - h_C_cpu[i * N + j]);
                if (diff > 1e-3) {
                    printf("Mismatch at [%d,%d]: Prefetch=%.6f, CPU=%.6f, diff=%.6f\n",
                           i, j, h_C_prefetch[i * N + j], h_C_cpu[i * N + j], diff);
                    errors++;
                }
            }
        }
        if (errors == 0) printf("✓ Prefetch matches CPU!\n");
        else printf("✗ Found %d mismatches\n", errors);
        
        free(h_C_prefetch);
    }
    
    printf("\n=== All Tests Complete! ===\n");
    
    cudaFree(d_B_idx);
    cudaFree(d_B_sparse);
    cudaFree(d_C_col_major);
    
    // Cleanup
    free(h_A); free(h_B); free(h_D); free(h_B_idx);
    free(h_B_dense); free(h_B_dense_col);
    free(h_C_cpu); free(h_C_cublas_col);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}
