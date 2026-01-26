#include "sparse_kernels.h"
#include <cuda_runtime.h>
#include <cstdio>

/**
 * Prefetch next tile while computing current one
 *  uses registers for prefetching
 */
__global__ void nmsparse_prefetch_kernel(
    float* g_vec,        // Dense matrix A (M×K), column-major
    float* g_mat_data,   // Sparse values B (W×N), row-major
    int* g_mat_index,    // Sparse indices (W×N), row-major
    float* g_data,       // Output C (M×N), column-major
    const int M,
    const int N, 
    const int K,
    const int W          // W = K * (1 - sparsity)
)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_N = 32;
    const int BLOCK_SIZE_K_SPARSE = 64;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;
    
    extern __shared__ float shared_mem[];
    
    int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
    int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;
    
    const int A_THREADS_PER_ROW = BLOCK_SIZE_M / 4;
    const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;
    
    const int THREADS_PER_BLOCK = (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N);
    
    const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
    const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;
    
    float* A_shared = shared_mem;
    float* B_shared = A_shared + BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE;
    
    float A_reg[THREAD_SIZE_M];
    float B_reg[THREAD_SIZE_N];
    float C_reg[THREAD_SIZE_N][THREAD_SIZE_M] = {0};
    
    int tid = threadIdx.x;
    
    int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);
    int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);
    
    int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;
    
    int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 4;
    
    // variables to hold prefetched data for next tile
    //  trigger cache loads - "unused" warning is expected

    float4 prefetch_A[16];  // max iterations for A (BLOCK_SIZE_K_SPARSE / A_STRIDES)
    float4 prefetch_B[16];  // max iterations for B
    int prefetch_count_A = 0;
    int prefetch_count_B = 0;
    
    (void)prefetch_A;
    (void)prefetch_B;
    
    // loop over K dimension in tiles
    for (int tile = 0; tile < (W + BLOCK_SIZE_K_SPARSE - 1) / BLOCK_SIZE_K_SPARSE; tile++) {
        int K_SPARSE_BLOCK_START = tile * BLOCK_SIZE_K_SPARSE;
        
        float* A_global_ptr = g_vec + M_BLOCK_START;
        float* B_global_ptr = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
        int* B_index_global_ptr = g_mat_index + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
        
        __syncthreads();
        
        // load current tile into shared memory
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += A_STRIDES) {
            if (K_SPARSE_BLOCK_START + i + A_BLOCK_ROW_START < W) {
                int idx = *(B_index_global_ptr + (i + A_BLOCK_ROW_START) * N);
                FETCH_FLOAT4(A_shared[(i + A_BLOCK_ROW_START) * BLOCK_SIZE_M + A_BLOCK_COL_START]) = 
                    FETCH_FLOAT4(A_global_ptr[idx * M + A_BLOCK_COL_START]);
            }
        }
        
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES) {
            if (K_SPARSE_BLOCK_START + i + B_BLOCK_ROW_START < W) {
                FETCH_FLOAT4(B_shared[(i + B_BLOCK_ROW_START) * BLOCK_SIZE_N + B_BLOCK_COL_START]) = 
                    FETCH_FLOAT4(B_global_ptr[(i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START]);
            }
        }
        
        // start loading next tile while waiting for sync
        // overlaps the memory access with the upcoming computation
        if (tile < (W + BLOCK_SIZE_K_SPARSE - 1) / BLOCK_SIZE_K_SPARSE - 1) {
            int next_K_START = (tile + 1) * BLOCK_SIZE_K_SPARSE;
            float* next_A_ptr = g_vec + M_BLOCK_START;
            float* next_B_ptr = g_mat_data + next_K_START * N + N_BLOCK_START;
            int* next_B_index_ptr = g_mat_index + next_K_START * N + N_BLOCK_START;
            
            prefetch_count_A = 0;
            prefetch_count_B = 0;
            
            // prefetch A data for next tile
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += A_STRIDES) {
                if (next_K_START + i + A_BLOCK_ROW_START < W) {
                    int idx = *(next_B_index_ptr + (i + A_BLOCK_ROW_START) * N);
                    prefetch_A[prefetch_count_A++] = 
                        FETCH_FLOAT4(next_A_ptr[idx * M + A_BLOCK_COL_START]);
                }
            }
            
            // prefetch B data for next tile
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES) {
                if (next_K_START + i + B_BLOCK_ROW_START < W) {
                    prefetch_B[prefetch_count_B++] = 
                        FETCH_FLOAT4(next_B_ptr[(i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START]);
                }
            }
        }
        
        __syncthreads();
        
        // compute tile (while prefetch data sits in registers)
        int tile_size = min(BLOCK_SIZE_K_SPARSE, W - K_SPARSE_BLOCK_START);
#pragma unroll
        for (int i = 0; i < tile_size; i++) {
#pragma unroll
            for (int k = 0; k < THREAD_SIZE_M; k++) {
                A_reg[k] = A_shared[i * BLOCK_SIZE_M + t_M * THREAD_SIZE_M + k];
            }
#pragma unroll
            for (int k = 0; k < THREAD_SIZE_N; k++) {
                B_reg[k] = B_shared[i * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + k];
            }
            
#pragma unroll
            for (int k = 0; k < THREAD_SIZE_N; k++) {
#pragma unroll
                for (int j = 0; j < THREAD_SIZE_M; j++) {
                    C_reg[k][j] += B_reg[k] * A_reg[j];
                }
            }
        }
        

    }
    
    // write results
#pragma unroll
    for (int i = 0; i < THREAD_SIZE_N; i++) {
#pragma unroll
        for (int j = 0; j < THREAD_SIZE_M; j++) {
            g_data[(BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * t_N + i) * M + 
                   (BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * t_M + j)] = C_reg[i][j];
        }
    }
}


void nmsparse_prefetch(
    float* A,
    float* B,
    int* B_idx,
    float* C,
    int M, int N, int K, int W
)
{
    const int BLOCK_SIZE_M = 32;
    const int BLOCK_SIZE_N = 32;
    const int THREAD_SIZE_M = 4;
    const int THREAD_SIZE_N = 4;
    const int BLOCK_SIZE_K_SPARSE = 64;
    
    dim3 dimBlock((BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N));
    dim3 dimGrid(M / BLOCK_SIZE_M, N / BLOCK_SIZE_N);
    
    // same shared memory as baseline
    int shared_mem_size = sizeof(float) * (BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE + 
                                           BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);
    
    nmsparse_prefetch_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(
        A, B, B_idx, C, M, N, K, W
    );
}
