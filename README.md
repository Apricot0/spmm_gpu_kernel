# N:M Sparse Matrix Multiplication

GPU kernels for N:M structured sparse matrix multiplication (50% sparsity, 2:4 pattern) on NVIDIA RTX 4070.
Report is located in root as GPU.pdf

## Directory Structure

```
course_project/
├── include/
│   └── sparse_kernels.h          # Function declarations
├── src/
│   ├── nmsparse_baseline.cu      # Single-buffered baseline
│   ├── nmsparse_prefetch.cu      # Register-based prefetching
│   ├── nmsparse_double_buffer.cu # Async double-buffer + Split-K
│   ├── nmspmm_kernel.cu          # Personal Project, not related
│   ├── cublas_gemm.cu            # cuBLAS dense baseline
│   └── helpers.cu                # Data initialization and verification
├── tests/
│   └── test_kernels.cu           # Benchmark harness
├── Makefile
├── profile_ncu.sh                # Nsight Compute profiling
└── analyze_ncu.py                # Profile analysis

```

## Kernel Implementations

### 1. nmSparse Baseline (`nmsparse_baseline.cu`)
Single-buffered sparse SpMM with synchronous loads.
- 32×32 block tiles, 4×4 thread tiles
- 64 sparse elements per K-tile
- Straightforward index-based gathering

### 2. nmSparse Prefetch (`nmsparse_prefetch.cu`)
Register-based prefetching to overlap memory and computation.
- Loads next tile into registers during computation
- Same tiling as baseline
- Tests latency hiding without hardware async

### 3. nmSparse Double-Buffer (`nmsparse_double_buffer.cu`)
PTX async copies with ping-pong buffers and Split-K.
- `cp.async` instructions for hardware acceleration
- 32 sparse elements per K-tile (smaller to fit double buffers)
- Split-K=2 for additional parallelism
- Requires Ampere+ (sm_80+)

### 4. cuBLAS Dense (`cublas_gemm.cu`)
Optimized dense GEMM for baseline comparison.

## Quick Start

### Build
```bash
make clean all
```

### Run Benchmarks
```bash
# Default (1024×1024×1024)
make run

# All sizes (64 to 8192)
make run-all

# Custom
./build/test_sparse_kernels <M> <N> <K> [warmup] [iterations]
./build/test_sparse_kernels 2048 2048 2048 10 100
```

### Debug
```bash
make debug-test  # 64×64×64 with all variants
```

## Profiling

```bash
# Profile with Nsight Compute
./profile_ncu.sh 1024 1024 1024

# Generates: ncu_profile_1024x1024x1024.csv
#           ncu_profile_1024x1024x1024_summary.csv
```

## Analysis

```bash
# Analyze profiling data
python3 analyze_ncu.py <input.csv> [output.csv]

## Scripts

- `profile_ncu.sh`: Collects 20+ hardware metrics (DRAM throughput, SM utilization, bank conflicts, warp stalls)
- `analyze_ncu.py`: Processes raw NCU CSV into statistical summaries

## Prerequisites

Tested on NYU Courant Institute cuda5 server (RTX 4070, CUDA 12.4).
