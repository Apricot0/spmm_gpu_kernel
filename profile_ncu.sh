#!/bin/bash
# Profile CUDA kernels with NCU using CSV output and generate summary statistics
make clean all
# Create ~/tmp directory if it doesn't exist (needed for NCU temporary files)
if [ ! -d "$HOME/tmp" ]; then
    echo "Creating $HOME/tmp directory for NCU temporary files..."
    mkdir -p "$HOME/tmp"
    if [ $? -eq 0 ]; then
        echo "✓ Directory created: $HOME/tmp"
    else
        echo "✗ Failed to create $HOME/tmp directory"
        exit 1
    fi
fi

# Export TMPDIR for NCU to use
export TMPDIR="$HOME/tmp"
echo "Using temporary directory: $TMPDIR"
echo ""

# Check if arguments are provided
if [ $# -lt 3 ]; then
    echo "Usage: $0 <M> <N> <K> [output_basename]"
    echo "Example: $0 1024 1024 1024"
    echo "Example: $0 1024 1024 1024 my_profile"
    echo ""
    echo "This will create:"
    echo "  - <basename>.csv (raw NCU data)"
    echo "  - <basename>_summary.csv (statistical summary for spreadsheets)"
    exit 1
fi

M=$1
N=$2
K=$3
BASENAME=${4:-"ncu_profile_${M}x${N}x${K}"}
CSV_FILE="${BASENAME}.csv"
SUMMARY_FILE="${BASENAME}_summary.csv"

echo "Profiling with dimensions: M=$M, N=$N, K=$K"
echo "CSV output will be saved to: $CSV_FILE"
echo "Summary will be saved to: $SUMMARY_FILE"
echo ""

# Run NCU profiling with CSV output
echo "Running NCU profiler..."
ncu --csv \
-k regex:"kernel_nmsparse_baseline|nmsparse_double_buffer_kernel|nmsparse_prefetch_kernel" \
--metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_bytes.sum,\
smsp__inst_executed.avg.per_cycle_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
smsp__sass_average_branch_targets_threads_uniform.pct,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
lts__t_sectors_op_read.sum,\
lts__t_requests_op_read.sum,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
smsp__warps_launched.sum,\
sm__warps_active.avg.per_cycle_elapsed,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct,\
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.pct,\
smsp__average_warps_issue_stalled_drain_per_issue_active.pct,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.pct \
./build/test_sparse_kernels $M $N $K > "$CSV_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "Error: NCU profiling failed"
    exit 1
fi

echo ""
echo "Profiling complete. Raw CSV data saved to: $CSV_FILE"
echo ""

# Run analysis script
echo "Analyzing results..."
echo ""
python3 analyze_ncu.py "$CSV_FILE" "$SUMMARY_FILE"

echo ""
echo "==================== Files Created ===================="
echo "Raw NCU data (CSV):           $CSV_FILE"
echo "Statistical summary (CSV):    $SUMMARY_FILE"
echo "======================================================="
echo ""
echo "To view raw CSV:              cat $CSV_FILE"
echo "To re-run analysis:           python3 analyze_ncu.py $CSV_FILE"
echo "Import into spreadsheet:      $SUMMARY_FILE"
