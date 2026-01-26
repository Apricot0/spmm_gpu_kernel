#include "sparse_kernels.h"
#include <cuda_runtime.h>
#include <cstdio>

//  async copy 
#define CP_ASYNC_CG(addr, ptr, bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(addr), "l"(ptr), "n"(bytes))

#define CP_ASYNC_COMMIT_GROUP() \
    asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_ALL() \
    asm volatile("cp.async.wait_all;\n" ::)

/**
 * nmSparse with B-only double buffering + optional SPLIT-K.
 *
 *  - Same tiling as baseline:
 *      block = 32x32 tile of C
 *      each thread computes a 4x4 sub-tile
 *  - W is processed in tiles of BLOCK_SIZE_K_SPARSE along the sparse dimension
 *  - B tiles are double-buffered with cp.async
 *  - A tiles are loaded synchronously via indices
 *  - SPLIT-K: gridDim.z slices the W-tiles among blocks in K dimension
 *      - When SPLIT_K == 1: behavior reduces to the original double-buffer kernel
 *      - When SPLIT_K  > 1: each blockIdx.z handles a disjoint subset of W-tiles
 *                           and partial results are accumulated with atomicAdd
 */
__global__ void nmsparse_double_buffer_kernel(
    float* g_vec,        // Dense matrix A (M×K), column-major
    float* g_mat_data,   // Sparse values B (W×N), row-major
    int*   g_mat_index,  // Sparse indices (W×N), row-major
    float* g_data,       // Output C (M×N), column-major
    const int M,
    const int N,
    const int K,
    const int W          // W = K * (1 - sparsity)
)
{
    const int BLOCK_SIZE_M        = 32;
    const int BLOCK_SIZE_N        = 32;
    const int BLOCK_SIZE_K_SPARSE = 32;
    const int THREAD_SIZE_M       = 4;
    const int THREAD_SIZE_N       = 4;

    extern __shared__ float shared_mem[];

    const int M_BLOCK_START = blockIdx.x * BLOCK_SIZE_M;
    const int N_BLOCK_START = blockIdx.y * BLOCK_SIZE_N;

    const int A_THREADS_PER_ROW = BLOCK_SIZE_M / 4;  // 8
    const int B_THREADS_PER_ROW = BLOCK_SIZE_N / 4;  // 8

    const int THREADS_PER_BLOCK =
        (BLOCK_SIZE_M / THREAD_SIZE_M) * (BLOCK_SIZE_N / THREAD_SIZE_N); // 64

    const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;  // 8
    const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;  // 8


    //   A_shared           : [BLOCK_SIZE_K_SPARSE][BLOCK_SIZE_M]
    //   B_shared[0], [1]   : two buffers [BLOCK_SIZE_K_SPARSE][BLOCK_SIZE_N]
    const int A_TILE_SIZE = BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE;
    const int B_TILE_SIZE = BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE;

    float* A_shared = shared_mem;
    float* B_shared[2];
    B_shared[0] = A_shared + A_TILE_SIZE;
    B_shared[1] = B_shared[0] + B_TILE_SIZE;

    float A_reg[THREAD_SIZE_M];
    float B_reg[THREAD_SIZE_N];
    float C_reg[THREAD_SIZE_N][THREAD_SIZE_M] = {0.0f};

    const int tid = threadIdx.x;

    const int t_N = tid % (BLOCK_SIZE_N / THREAD_SIZE_N);   // 0..7
    const int t_M = tid / (BLOCK_SIZE_N / THREAD_SIZE_N);   // 0..7

    const int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
    const int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

    const int A_BLOCK_COL_START = (tid % A_THREADS_PER_ROW) * 4;
    const int B_BLOCK_COL_START = (tid % B_THREADS_PER_ROW) * 4;

    int read_buf  = 0;
    int write_buf = 1;

    //  SPLIT-K partitioning 

    const int tiles_total = (W + BLOCK_SIZE_K_SPARSE - 1) / BLOCK_SIZE_K_SPARSE;
    const int SPLIT_K    = gridDim.z > 0 ? gridDim.z : 1;
    const int split_id   = blockIdx.z;  // 0 .. SPLIT_K-1

    // distribute tiles 
    const int tiles_per_split = (tiles_total + SPLIT_K - 1) / SPLIT_K; // ceil
    const int tile_start      = split_id * tiles_per_split;
    const int tile_end        = min(tiles_total, tile_start + tiles_per_split);

    if (tile_start >= tile_end) {
        return;
    }

    // preload the first tile 

    {
        const int K_SPARSE_BLOCK_START = tile_start * BLOCK_SIZE_K_SPARSE;

        float* A_global_ptr       = g_vec      + M_BLOCK_START;
        float* B_global_ptr       = g_mat_data + K_SPARSE_BLOCK_START * N + N_BLOCK_START;
        int*   B_index_global_ptr = g_mat_index+ K_SPARSE_BLOCK_START * N + N_BLOCK_START;

        __syncthreads();

        // A tile using indices from B 
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += A_STRIDES) {
            int row = i + A_BLOCK_ROW_START;
            if (K_SPARSE_BLOCK_START + row < W) {
                int idx = *(B_index_global_ptr + row * N);
                FETCH_FLOAT4(A_shared[row * BLOCK_SIZE_M + A_BLOCK_COL_START]) =
                    FETCH_FLOAT4(A_global_ptr[idx * M + A_BLOCK_COL_START]);
            }
        }

        // load B tile into B_shared[read_buf]
#pragma unroll
        for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES) {
            int row = i + B_BLOCK_ROW_START;
            if (K_SPARSE_BLOCK_START + row < W) {
                FETCH_FLOAT4(B_shared[read_buf][row * BLOCK_SIZE_N + B_BLOCK_COL_START]) =
                    FETCH_FLOAT4(B_global_ptr[row * N + B_BLOCK_COL_START]);
            }
        }

        __syncthreads();
    }


    for (int tile = tile_start; tile < tile_end; ++tile) {
        const int K_SPARSE_BLOCK_START = tile * BLOCK_SIZE_K_SPARSE;

        // prefetch next tile’s B into write_buf, if any
        const int next_tile = tile + 1;
        if (next_tile < tile_end) {
            const int NEXT_K_START = next_tile * BLOCK_SIZE_K_SPARSE;
            float* next_B_global_ptr = g_mat_data + NEXT_K_START * N + N_BLOCK_START;

#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += B_STRIDES) {
                int row = i + B_BLOCK_ROW_START;
                if (NEXT_K_START + row < W) {
                    float* dst = &B_shared[write_buf][row * BLOCK_SIZE_N + B_BLOCK_COL_START];
                    float* src = next_B_global_ptr + row * N + B_BLOCK_COL_START;
                    unsigned int addr = __cvta_generic_to_shared(dst);
                    CP_ASYNC_CG(addr, src, 16);   // 16 bytes = float4
                }
            }
            CP_ASYNC_COMMIT_GROUP();
        }

        //  compute with current A_shared and B_shared[read_buf] 

        const int tile_size = min(BLOCK_SIZE_K_SPARSE, W - K_SPARSE_BLOCK_START);

#pragma unroll
        for (int kk = 0; kk < tile_size; ++kk) {
#pragma unroll
            for (int m = 0; m < THREAD_SIZE_M; ++m) {
                A_reg[m] = A_shared[kk * BLOCK_SIZE_M + t_M * THREAD_SIZE_M + m];
            }
#pragma unroll
            for (int n = 0; n < THREAD_SIZE_N; ++n) {
                B_reg[n] = B_shared[read_buf][kk * BLOCK_SIZE_N + t_N * THREAD_SIZE_N + n];
            }

#pragma unroll
            for (int n = 0; n < THREAD_SIZE_N; ++n) {
#pragma unroll
                for (int m = 0; m < THREAD_SIZE_M; ++m) {
                    C_reg[n][m] += B_reg[n] * A_reg[m];
                }
            }
        }

        // A_shared for next tile 

        if (next_tile < tile_end) {
            const int NEXT_K_START = next_tile * BLOCK_SIZE_K_SPARSE;

            CP_ASYNC_WAIT_ALL();
            __syncthreads();

            float* A_global_ptr       = g_vec      + M_BLOCK_START;
            int*   B_index_global_ptr = g_mat_index+ NEXT_K_START * N + N_BLOCK_START;

#pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K_SPARSE; i += A_STRIDES) {
                int row = i + A_BLOCK_ROW_START;
                if (NEXT_K_START + row < W) {
                    int idx = *(B_index_global_ptr + row * N);
                    FETCH_FLOAT4(A_shared[row * BLOCK_SIZE_M + A_BLOCK_COL_START]) =
                        FETCH_FLOAT4(A_global_ptr[idx * M + A_BLOCK_COL_START]);
                }
            }

            __syncthreads();

            int tmp   = read_buf;
            read_buf  = write_buf;
            write_buf = tmp;
        }
    }

    //accumulate

    const int row_base = BLOCK_SIZE_M * blockIdx.x + THREAD_SIZE_M * t_M;
    const int col_base = BLOCK_SIZE_N * blockIdx.y + THREAD_SIZE_N * t_N;

    if (SPLIT_K == 1) {
        // No SPLIT-K
#pragma unroll
        for (int nn = 0; nn < THREAD_SIZE_N; ++nn) {
#pragma unroll
            for (int mm = 0; mm < THREAD_SIZE_M; ++mm) {
                int row = row_base + mm;
                int col = col_base + nn;
                g_data[col * M + row] = C_reg[nn][mm];
            }
        }
    } else {
        // SPLIT-K: partial sums from each z-slice, combine with atomicAdd
#pragma unroll
        for (int nn = 0; nn < THREAD_SIZE_N; ++nn) {
#pragma unroll
            for (int mm = 0; mm < THREAD_SIZE_M; ++mm) {
                int row = row_base + mm;
                int col = col_base + nn;
                atomicAdd(&g_data[col * M + row], C_reg[nn][mm]);
            }
        }
    }
}

/**
 * Wrapper function
 *
 * You can set SPLIT_K = 1 for "normal" double-buffered behavior,
 * or >1 to enable SPLIT-K (remember to zero C before launch when SPLIT_K > 1).
 */
void nmsparse_double_buffer(
    float* A,
    float* B,
    int*   B_idx,
    float* C,
    int M, int N, int K, int W
)
{
    const int BLOCK_SIZE_M        = 32;
    const int BLOCK_SIZE_N        = 32;
    const int THREAD_SIZE_M       = 4;
    const int THREAD_SIZE_N       = 4;
    const int BLOCK_SIZE_K_SPARSE = 32;

    dim3 dimBlock((BLOCK_SIZE_M / THREAD_SIZE_M) *
                  (BLOCK_SIZE_N / THREAD_SIZE_N));

    // Choose SPLIT_K; you can tune this. For your lab, start with 1 (no SPLIT-K).
    const int SPLIT_K = 2;  // or 2/4 if you want to experiment with split-K

    dim3 dimGrid(M / BLOCK_SIZE_M,
                 N / BLOCK_SIZE_N,
                 SPLIT_K);

    // Shared memory: A + 2*B tiles
    int shared_mem_size = sizeof(float) *
        (BLOCK_SIZE_M * BLOCK_SIZE_K_SPARSE +
         2 * BLOCK_SIZE_N * BLOCK_SIZE_K_SPARSE);

    nmsparse_double_buffer_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(
        A, B, B_idx, C, M, N, K, W
    );
}
