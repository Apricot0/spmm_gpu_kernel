#include "sparse_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

/**
 * Initialize data for 50% sparsity (2:4 structured sparsity pattern)
 */
void init_data_50_sparsity(float* A, float* B, int* D, float* C, int M, int N, int K) {
    srand(time(NULL));
    
    const int W = K / 2;  // 50% sparsity
    const int Q = (N + VEC_LEN - 1) / VEC_LEN;  // Compressed index dimension
    
    // Initialize output to zero
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }
    
    // Initialize dense matrix A (column-major)
    for (int j = 0; j < K; j++) {
        for (int i = 0; i < M; i++) {
            A[i + j * M] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
    
    // Initialize sparse matrix B values (row-major: W×N)
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
    
    // Create structured sparse index pattern (row-major: W×N initially)
    // For 50% sparsity (2:4 pattern), each group of 4 elements has 2 non-zeros
    int* temp_idx = (int*)malloc(sizeof(int) * W * N);
    
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < N; j++) {
            // Map sparse row i to dense row based on structured pattern
            // For 50% sparsity, map pair-wise: (0,1)->0,2, (2,3)->4,6, etc.
            int block = i / 2;
            int offset = i % 2;
            temp_idx[i * N + j] = block * 4 + offset * 2;
        }
    }
    
    // transform indices to compressed layout (W×Q) expected by kernel
    const int Ns = 32;  // Block size
    const int Qs = Ns / VEC_LEN;  // Q-tiles
    
    // first copy to a temporary buffer in row-major layout
    int* temp_D = (int*)malloc(sizeof(int) * W * Q);
    for (int i = 0; i < W; i++) {
        for (int j = 0; j < Q; j++) {
            int src_col = j * VEC_LEN;
            if (src_col < N) {
                temp_D[i * Q + j] = temp_idx[i * N + src_col];
            } else {
                temp_D[i * Q + j] = 0;
            }
        }
    }
    
    // apply the layout transformation to match PreProcessing_low_sparsity
    // transform from row-major [W][Q] to blocked layout
    for (int j = 0; j < Q; j += Qs) {
        int* p = D + j * W;
        for (int row = 0; row < W; row++) {
            for (int col = 0; col < Qs && (j + col) < Q; col++) {
                *p = temp_D[row * Q + j + col];
                p++;
            }
        }
    }
    
    free(temp_idx);
    free(temp_D);
}

//CPU reference implementation for verification
 
void cpu_reference(float* A, float* B, int* B_idx, float* C, int M, int N, int K, int W) {
    // C = A * B_sparse
    // A is M×K (column-major), B_sparse is K×N (represented as W×N values + indices)
    // C is M×N (row-major)
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < W; k++) {
                int col_idx = B_idx[k * N + j];
                if (col_idx >= 0 && col_idx < K) {
                    sum += A[i + col_idx * M] * B[k * N + j];
                }
            }
            C[i * N + j] = sum;
        }
    }
}

// verify GPU results against CPU reference

bool verify_result(float* C_gpu, float* C_cpu, int M, int N) {
    const float rtol = 1e-3f;
    const float atol = 1e-5f;
    int errors = 0;
    const int max_errors = 10;
    
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(C_gpu[i] - C_cpu[i]);
        float threshold = atol + rtol * fabs(C_cpu[i]);
        
        if (diff > threshold) {
            if (errors < max_errors) {
                int row = i / N;
                int col = i % N;
                printf("Mismatch at [%d,%d]: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                       row, col, C_gpu[i], C_cpu[i], diff);
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("Total mismatches: %d / %d (%.2f%%)\n", 
               errors, M * N, 100.0f * errors / (M * N));
        return false;
    }
    
    printf("✓ Verification passed! All results match within tolerance.\n");
    return true;
}
