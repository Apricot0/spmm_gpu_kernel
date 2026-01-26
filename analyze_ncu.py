#!/usr/bin/env python3
"""
Parse NCU CSV profiling output and show only raw statistics.
No bottleneck analysis or recommendations - just the numbers.
"""

import csv
import sys
from collections import defaultdict
import statistics

def parse_ncu_csv(filename):
    """Parse NCU CSV output file and extract metrics."""
    kernels = defaultdict(lambda: defaultdict(list))
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        
        for row in reader:
            # NCU CSV format (without headers):
            # [0]ID, [1]Process ID, [2]Process Name, [3]Host Name, [4]Kernel Name,
            # [5]Context, [6]Stream, [7]Block, [8]Grid, [9]Section, [10]CC,
            # [11]Section Name, [12]Metric Name, [13]Metric Unit, [14]Metric Value
            
            if len(row) < 15:
                continue
            
            # Extract kernel name (column 4) - simplify by removing parameters
            kernel_full = row[4]
            # Extract just the function name before '('
            kernel_name = kernel_full.split('(')[0] if '(' in kernel_full else kernel_full
            
            # Extract metric name (column 12) and value (column 14)
            metric_name = row[12]
            metric_value = row[14]
            
            # Parse the value
            try:
                # Remove commas and convert to float
                float_value = float(metric_value.replace(',', ''))
                kernels[kernel_name][metric_name].append(float_value)
            except (ValueError, AttributeError):
                pass
    
    return kernels

def calculate_statistics(values):
    """Calculate mean, min, max, and stddev."""
    if not values:
        return None
    
    return {
        'mean': statistics.mean(values),
        'min': min(values),
        'max': max(values),
        'stddev': statistics.stdev(values) if len(values) > 1 else 0.0,
        'count': len(values)
    }

def print_statistics(kernels):
    """Print raw statistics for all kernels."""
    print("=" * 100)
    print("NCU PROFILING STATISTICS")
    print("=" * 100)
    print()
    
    for kernel_name, metrics in sorted(kernels.items()):
        print(f"Kernel: {kernel_name}")
        print("-" * 100)
        
        stats = {}
        for metric_name, values in sorted(metrics.items()):
            stat = calculate_statistics(values)
            if stat:
                stats[metric_name] = stat
        
        print(f"{'Metric':<60} {'Mean':<12} {'Min':<12} {'Max':<12} {'StdDev':<10} {'Count':<6}")
        print("-" * 100)
        
        for metric_name, stat in sorted(stats.items()):
            print(f"{metric_name:<60} {stat['mean']:<12.4f} {stat['min']:<12.4f} "
                  f"{stat['max']:<12.4f} {stat['stddev']:<10.4f} {stat['count']:<6}")
        
        print()
        print("=" * 100)
        print()

def compare_kernels(kernels):
    """Compare metrics across kernels."""
    print("=" * 100)
    print("KERNEL COMPARISON")
    print("=" * 100)
    print()
    
    all_metrics = set()
    for metrics in kernels.values():
        all_metrics.update(metrics.keys())
    
    for metric in sorted(all_metrics):
        print(f"\n{metric}:")
        print("-" * 100)
        kernel_values = []
        for kernel_name, metrics in kernels.items():
            if metric in metrics:
                mean_val = statistics.mean(metrics[metric])
                kernel_values.append((kernel_name, mean_val))
        
        if kernel_values:
            kernel_values.sort(key=lambda x: x[1], reverse=True)
            for kernel_name, value in kernel_values:
                print(f"  {kernel_name:<70} {value:>15.4f}")
    
    print()
    print("=" * 100)

def export_to_csv(kernels, output_file):
    """Export statistics to CSV format."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Kernel', 'Metric', 'Mean', 'Min', 'Max', 'StdDev', 'Count'])
        
        for kernel_name, metrics in sorted(kernels.items()):
            for metric_name, values in sorted(metrics.items()):
                stat = calculate_statistics(values)
                if stat:
                    writer.writerow([
                        kernel_name,
                        metric_name,
                        f"{stat['mean']:.4f}",
                        f"{stat['min']:.4f}",
                        f"{stat['max']:.4f}",
                        f"{stat['stddev']:.4f}",
                        stat['count']
                    ])

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_ncu_csv_simple.py <ncu_csv_file> [output_summary.csv]")
        print("\nExample:")
        print("  ncu --csv ... > profile_output.csv")
        print("  python analyze_ncu_csv_simple.py profile_output.csv")
        print("\nOptionally export summary to CSV:")
        print("  python analyze_ncu_csv_simple.py profile_output.csv summary.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        kernels = parse_ncu_csv(input_file)
        
        if not kernels:
            print("No kernel data found in the CSV file.")
            sys.exit(1)
        
        print_statistics(kernels)
        
        if len(kernels) > 1:
            compare_kernels(kernels)
        
        if output_file:
            export_to_csv(kernels, output_file)
            print(f"\nStatistics exported to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
