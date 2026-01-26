#include "sparse_kernels.h"
#include <cuda_runtime.h>
#include <cstdio>

// Helper macros for vectorized loads
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])

// PTX assembly for async copy
#define CP_ASYNC_CG(addr, ptr, bytes) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" ::"r"(addr), "l"(ptr), "n"(bytes))

#define CP_ASYNC_CA(addr, ptr, bytes) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(addr), "l"(ptr), "n"(bytes))

#define CP_ASYNC_COMMIT_GROUP() \
    asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_ALL() \
    asm volatile("cp.async.wait_all;\n" ::)

/**
 * Single N:M SpMM kernel for 50% sparsity
 * Based on original kernel_32x32_4x4_low_sparsity
 * 
 * Template parameters:
 * - Ms, Ns: Block tile size (32x32)
 * - Ks: K-dimension tile for A (32)
 * - Ws: Sparse K-dimension tile for B (16 for 50% sparsity)
 * - Mt, Nt: Thread tile size (4x4)
 */
template <
    const int Ms,
    const int Ns,
    const int Ks,
    const int Ws,
    const int Mt,
    const int Nt>
__global__ void kernel_nmspmm_low_sparsity(
    float* A,      // Dense matrix A (M×K), column-major
    float* B,      // Sparse values of B (W×N), row-major
    int* D,        // Indices for sparse B (W×Q), row-major, Q = N/VEC_LEN
    float* C,      // Output matrix C (M×N), row-major
    int M, int N, int K, int W
)
{
    const int Qs = (Ns + VEC_LEN - 1) / VEC_LEN;

    extern __shared__ char smem[];
    float At[2][Mt], Bt[2][Nt], Ct[Mt][Nt] = { 0.0f };

    // Double-buffered shared memory
    float* As_write_ptr = (float*)smem; // [Ks][Ms]
    float* As_read_ptr = As_write_ptr + Ks * Ms;

    float* Bs_write_ptr = (float*)(smem + 2 * Ks * Ms * sizeof(float));
    float* Bs_read_ptr = Bs_write_ptr + Ws * Ns; // [Ws][Ns]

    int* Ds_write_ptr = (int*)(smem + 2 * (Ks * Ms + Ws * Ns) * sizeof(float));
    int* Ds_read_ptr = Ds_write_ptr + Ws * Qs; // [Ws][Qs]

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;

    const int bi = blockIdx.y * Ms;
    const int bj = blockIdx.x * Ns;

    const int ti = warp_id * 16 + ((lane_id / 16) * 2 + (lane_id % 2)) * 4;
    const int tj = ((lane_id / 2) % 8) * 4;

    const int THREADS_PER_BLOCK = (Ms / Mt) * (Ns / Nt);

    const int A_THREADS_PER_ROW = Ms / 4;
    const int B_THREADS_PER_ROW = Ns / 2;

    const int A_STRIDES = THREADS_PER_BLOCK / A_THREADS_PER_ROW;
    const int B_STRIDES = THREADS_PER_BLOCK / B_THREADS_PER_ROW;

    int A_BLOCK_ROW_START = tid / A_THREADS_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREADS_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREADS_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREADS_PER_ROW * 2;

    float* A_ptr = A + bi;
    float* B_ptr = B + bj;
    // D is W×Q layout (row-major): For element D[row][col], offset = row * Q + col
    const int Q = (N + VEC_LEN - 1) / VEC_LEN;
    const int q_offset = bj / VEC_LEN;  // Column index we need

    int idx[Ws];

    // Initial load (first tile, rows 0 to Ws-1)
#pragma unroll
    for (int i = 0; i < Ks; i += A_STRIDES) {
        FETCH_FLOAT4(As_write_ptr[(i + A_BLOCK_ROW_START) * Ms + A_BLOCK_COL_START])
            = FETCH_FLOAT4(A_ptr[(i + A_BLOCK_ROW_START) * M + A_BLOCK_COL_START]);
    }
#pragma unroll
    for (int i = 0; i < Ws; i += B_STRIDES) {
        FETCH_FLOAT2(Bs_write_ptr[(i + B_BLOCK_ROW_START) * Ns + B_BLOCK_COL_START])
            = FETCH_FLOAT2(B_ptr[(i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START]);
    }
    // Load D values: need D[0..Ws-1][q_offset..q_offset+Qs-1]
    // Stored in Ds as [Ws][Qs] layout
    if (tid < Ws * Qs) {
        int local_row = tid / Qs;
        int local_col = tid % Qs;
        Ds_write_ptr[tid] = D[local_row * Q + q_offset + local_col];
    }

    __syncthreads();
    
    // Convert global indices to tile-relative indices
    // First tile: v=0, so indices should be modulo Ks
#pragma unroll
    for (int p = 0; p < Ws; p++) {
        int global_idx = Ds_write_ptr[p * Qs + tj / VEC_LEN];
        idx[p] = global_idx % Ks;  // Convert to tile-relative
    }

    FETCH_FLOAT4(Bt[0][0]) = FETCH_FLOAT4(Bs_write_ptr[0 * Ns + tj]);
    FETCH_FLOAT4(At[0][0]) = FETCH_FLOAT4(As_write_ptr[idx[0] * Ms + ti]);

    // Main loop with double buffering
    for (int u = Ws, v = Ks; u < W; u += Ws, v += Ks) {

        A_ptr = A + bi + v * M;
        B_ptr = B + bj + u * N;

        // Swap buffers
        {
            float* t;
            t = As_read_ptr, As_read_ptr = As_write_ptr, As_write_ptr = t;
            t = Bs_read_ptr, Bs_read_ptr = Bs_write_ptr, Bs_write_ptr = t;
        }
        {
            int* t;
            t = Ds_read_ptr, Ds_read_ptr = Ds_write_ptr, Ds_write_ptr = t;
        }

        // Async load next tile of D: need D[u..u+Ws-1][q_offset..q_offset+Qs-1]
        if (tid < Ws * Qs) {
            int local_row = tid / Qs;
            int local_col = tid % Qs;
            uint32_t addr = __cvta_generic_to_shared(&Ds_write_ptr[tid]);
            CP_ASYNC_CA(addr, &D[(u + local_row) * Q + q_offset + local_col], 4);
        }

#pragma unroll
        for (int i = 0; i < Ks; i += A_STRIDES) {
            uint32_t addr = __cvta_generic_to_shared(&As_write_ptr[(i + A_BLOCK_ROW_START) * Ms + A_BLOCK_COL_START]);
            CP_ASYNC_CG(addr, &A_ptr[(i + A_BLOCK_ROW_START) * M + A_BLOCK_COL_START], 16);
        }
#pragma unroll
        for (int i = 0; i < Ws; i += B_STRIDES) {
            uint32_t addr = __cvta_generic_to_shared(&Bs_write_ptr[(i + B_BLOCK_ROW_START) * Ns + B_BLOCK_COL_START]);
            CP_ASYNC_CA(addr, &B_ptr[(i + B_BLOCK_ROW_START) * N + B_BLOCK_COL_START], 8);
        }

        CP_ASYNC_COMMIT_GROUP();

        // Compute with current tile while loading next
#pragma unroll
        for (int p = 0; p < Ws - 1; p += 1) {
            FETCH_FLOAT4(Bt[(p + 1) % 2][0]) = FETCH_FLOAT4(Bs_read_ptr[(p + 1) * Ns + tj]);
            FETCH_FLOAT4(At[(p + 1) % 2][0]) = FETCH_FLOAT4(As_read_ptr[idx[p + 1] * Ms + ti]);

#pragma unroll
            for (int i = 0; i < Mt; i++) {
                if (i % 2) {
#pragma unroll
                    for (int j = Nt - 1; j >= 0; j--) {
                        Ct[i][j] += At[p % 2][i] * Bt[p % 2][j];
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < Nt; j++) {
                        Ct[i][j] += At[p % 2][i] * Bt[p % 2][j];
                    }
                }
            }
        }
        
        CP_ASYNC_WAIT_ALL();
        __syncthreads();

        // Convert global indices to tile-relative indices for current tile
#pragma unroll
        for (int p = 0; p < Ws; p++) {
            int global_idx = Ds_write_ptr[p * Qs + tj / VEC_LEN];
            idx[p] = global_idx % Ks;  // Convert to tile-relative
        }

        FETCH_FLOAT4(Bt[0][0]) = FETCH_FLOAT4(Bs_write_ptr[0 * Ns + tj]);
        FETCH_FLOAT4(At[0][0]) = FETCH_FLOAT4(As_write_ptr[idx[0] * Ms + ti]);

#pragma unroll
        for (int i = 0; i < Mt; i++) {
            if (i % 2) {
#pragma unroll
                for (int j = Nt - 1; j >= 0; j--) {
                    Ct[i][j] += At[1][i] * Bt[1][j];
                }
            } else {
#pragma unroll
                for (int j = 0; j < Nt; j++) {
                    Ct[i][j] += At[1][i] * Bt[1][j];
                }
            }
        }
    }

    // Process last tile
    {
        float* t;
        t = As_read_ptr, As_read_ptr = As_write_ptr, As_write_ptr = t;
        t = Bs_read_ptr, Bs_read_ptr = Bs_write_ptr, Bs_write_ptr = t;
    }
    {
        int* t;
        t = Ds_read_ptr, Ds_read_ptr = Ds_write_ptr, Ds_write_ptr = t;
    }

#pragma unroll
    for (int p = 0; p < Ws - 1; p++) {
        FETCH_FLOAT4(Bt[(p + 1) % 2][0]) = FETCH_FLOAT4(Bs_read_ptr[(p + 1) * Ns + tj]);
        FETCH_FLOAT4(At[(p + 1) % 2][0]) = FETCH_FLOAT4(As_read_ptr[idx[p + 1] * Ms + ti]);

#pragma unroll
        for (int i = 0; i < Mt; i++) {
            if (i % 2) {
#pragma unroll
                for (int j = Nt - 1; j >= 0; j--) {
                    Ct[i][j] += At[p % 2][i] * Bt[p % 2][j];
                }
            } else {
#pragma unroll
                for (int j = 0; j < Nt; j++) {
                    Ct[i][j] += At[p % 2][i] * Bt[p % 2][j];
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < Mt; i++) {
        if (i % 2) {
#pragma unroll
            for (int j = Nt - 1; j >= 0; j--) {
                Ct[i][j] += At[1][i] * Bt[1][j];
            }
        } else {
#pragma unroll
            for (int j = 0; j < Nt; j++) {
                Ct[i][j] += At[1][i] * Bt[1][j];
            }
        }
    }

    // Write results
#pragma unroll
    for (int i = 0; i < 4; i++) {
        FETCH_FLOAT4(C[(bi + ti + i) * N + (bj + tj + 0)]) = FETCH_FLOAT4(Ct[i][0]);
    }
}

/**
 * Wrapper function for N:M SpMM kernel
 * Entry point for 50% sparsity
 */
void nmspmm_kernel_optimized(
    float* A,      // Dense matrix A (M×K), column-major
    float* B,      // Sparse values of B (W×N), row-major  
    int* D,        // Indices for sparse B (W×Q), row-major, Q = N/VEC_LEN
    float* C,      // Output matrix C (M×N), row-major
    int M, int N, int K, int W
)
{
    const int Ms = 32;
    const int Ns = 32;
    const int Mt = 4;
    const int Nt = 4;
    const int Ks = 32;
    const int Ws = 16;  // For 50% sparsity
    const int Q = (N + VEC_LEN - 1) / VEC_LEN;

    // Validation
    printf("[NM-SpMM] Kernel launch parameters:\n");
    printf("  Matrix dimensions: M=%d, N=%d, K=%d, W=%d, Q=%d\n", M, N, K, W, Q);
    printf("  Block size: %dx%d, Grid size: %dx%d\n", Ns/Nt, Ms/Mt, N/Ns, M/Ms);
    
    if (M % Ms != 0 || N % Ns != 0) {
        printf("  WARNING: Matrix dimensions not divisible by block size!\n");
        printf("    M %% %d = %d, N %% %d = %d\n", Ms, M % Ms, Ns, N % Ns);
    }
    
    // Check memory bounds
    size_t A_size = (size_t)M * K;
    size_t B_size = (size_t)W * N;
    size_t D_size = (size_t)W * Q;
    size_t C_size = (size_t)M * N;
    
    printf("  Required memory:\n");
    printf("    A: %zu elements (%.2f MB)\n", A_size, A_size * sizeof(float) / 1024.0 / 1024.0);
    printf("    B: %zu elements (%.2f MB)\n", B_size, B_size * sizeof(float) / 1024.0 / 1024.0);
    printf("    D: %zu elements (%.2f MB)\n", D_size, D_size * sizeof(int) / 1024.0 / 1024.0);
    printf("    C: %zu elements (%.2f MB)\n", C_size, C_size * sizeof(float) / 1024.0 / 1024.0);

    dim3 dimBlock(Ns / Nt, Ms / Mt);
    dim3 dimGrid(N / Ns, M / Ms);

    size_t smem_nbytes = 2 * (Ks * Ms + Ws * Ns) * sizeof(float)
        + 2 * (Ws * Ns / VEC_LEN + Ks) * sizeof(int);
    
    printf("  Shared memory: %zu bytes (%.2f KB)\n", smem_nbytes, smem_nbytes / 1024.0);

    // Check device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (smem_nbytes > prop.sharedMemPerBlock) {
        printf("  ERROR: Shared memory exceeds device limit (%zu > %zu)!\n",
               smem_nbytes, prop.sharedMemPerBlock);
        return;
    }

    printf("  Launching kernel...\n");
    kernel_nmspmm_low_sparsity<Ms, Ns, Ks, Ws, Mt, Nt>
        <<<dimGrid, dimBlock, smem_nbytes>>>(A, B, D, C, M, N, K, W);
    
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("  ERROR: Kernel launch failed: %s\n", cudaGetErrorString(launch_err));
        return;
    }
    
    printf("  Kernel launched successfully, waiting for completion...\n");
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        printf("  ERROR: Kernel execution failed: %s\n", cudaGetErrorString(sync_err));
        printf("  This likely indicates an out-of-bounds memory access.\n");
        
        // Try to get more info
        printf("\n  Diagnostic info:\n");
        printf("    Last block that would execute: grid[%d, %d]\n", N/Ns - 1, M/Ms - 1);
        printf("    Last thread in block: thread[%d, %d]\n", Ns/Nt - 1, Ms/Mt - 1);
        printf("    Max output index: C[%d * %d + %d] = C[%zu]\n", 
               M-1, N, N-1, (size_t)(M-1) * N + (N-1));
        printf("    Max A index: A[%d + %d * %d] = A[%zu]\n",
               M-1, K-1, M, (size_t)(M-1) + (K-1) * M);
        printf("    Max B index: B[%d * %d + %d] = B[%zu]\n",
               W-1, N, N-1, (size_t)(W-1) * N + (N-1));
        printf("    Max D index: D[%d * %d + %d] = D[%zu]\n",
               W-1, Q, Q-1, (size_t)(W-1) * Q + (Q-1));
        
        return;
    }
    
    printf("  Kernel completed successfully!\n\n");
}
