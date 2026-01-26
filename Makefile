# Makefile for Simplified N:M Sparse Matrix Multiplication
# Target: sm_89 (RTX 4090)

# Compiler and flags
NVCC = nvcc
ARCH = sm_89
NVCC_FLAGS = -arch=$(ARCH) -O3 -Xcompiler -Wall
LDFLAGS = -lcublas
INC_DIR = include
SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build

# Source files
SOURCES = $(SRC_DIR)/nmsparse_baseline.cu \
          $(SRC_DIR)/nmsparse_double_buffer.cu \
          $(SRC_DIR)/nmsparse_prefetch.cu \
          $(SRC_DIR)/nmspmm_kernel.cu \
          $(SRC_DIR)/cublas_gemm.cu \
          $(SRC_DIR)/helpers.cu \
          $(TEST_DIR)/test_kernels.cu

# Target executable
TARGET = $(BUILD_DIR)/test_sparse_kernels
DEBUG_TARGET = $(BUILD_DIR)/debug_cublas
DEBUG_TEST = $(BUILD_DIR)/debug_test

# Default target
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build target
$(TARGET): $(SOURCES) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) $(SOURCES) -o $(TARGET) $(LDFLAGS)

# Build debug target
$(DEBUG_TARGET): debug_cublas.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) debug_cublas.cu -o $(DEBUG_TARGET) $(LDFLAGS)

# Build debug test with helpers
$(DEBUG_TEST): debug_test.cu $(SRC_DIR)/helpers.cu $(SRC_DIR)/cublas_gemm.cu $(SRC_DIR)/nmspmm_kernel.cu $(SRC_DIR)/nmsparse_baseline.cu $(SRC_DIR)/nmsparse_double_buffer.cu $(SRC_DIR)/nmsparse_prefetch.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -I$(INC_DIR) debug_test.cu $(SRC_DIR)/helpers.cu $(SRC_DIR)/cublas_gemm.cu $(SRC_DIR)/nmspmm_kernel.cu $(SRC_DIR)/nmsparse_baseline.cu $(SRC_DIR)/nmsparse_double_buffer.cu $(SRC_DIR)/nmsparse_prefetch.cu -o $(DEBUG_TEST) $(LDFLAGS)

# Debug cuBLAS
debug: $(DEBUG_TARGET)
	./$(DEBUG_TARGET)

# Debug test with small matrices
debug-test: $(DEBUG_TEST)
	./$(DEBUG_TEST)

# Run test with default parameters (1024x1024x1024)
run: $(TARGET)
	./$(TARGET)

# Run test with custom dimensions (example: 2048x2048x2048)
run-large: $(TARGET)
	./$(TARGET) 2048 2048 2048

# Run test with small dimensions for quick testing
run-small: $(TARGET)
	./$(TARGET) 512 512 512

# Run test with various sizes
run-all: $(TARGET)
	@echo "=== Testing 64x64x64 ==="
	./$(TARGET) 64 64 64 5 50
	@echo ""
	@echo "=== Testing 128x128x128 ==="
	./$(TARGET) 128 128 128 5 50
	@echo ""
	@echo "=== Testing 256x256x256 ==="
	./$(TARGET) 256 256 256 5 50
	@echo ""
	@echo "=== Testing 512x512x512 ==="
	./$(TARGET) 512 512 512 5 50
	@echo ""
	@echo "=== Testing 1024x1024x1024 ==="
	./$(TARGET) 1024 1024 1024 10 100
	@echo ""
	@echo "=== Testing 2048x2048x2048 ==="
	./$(TARGET) 2048 2048 2048 10 100
	@echo ""
	@echo "=== Testing 4096x4096x4096 ==="
	./$(TARGET) 4096 4096 4096 5 50
	@echo ""
	@echo "=== Testing 8192x8192x8192 ==="
	./$(TARGET) 8192 8192 8192 3 20

# Profile with Nsight Compute
profile: $(TARGET)
	ncu --set full -o $(BUILD_DIR)/profile_report ./$(TARGET) 1024 1024 1024 1 1

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Clean including debug
clean-all: clean
	rm -f debug_cublas

# Help message
help:
	@echo "Simplified N:M Sparse Matrix Multiplication - Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make              - Build the project"
	@echo "  make run          - Run with default size (1024x1024x1024)"
	@echo "  make run-small    - Run with small size (512x512x512)"
	@echo "  make run-large    - Run with large size (2048x2048x2048)"
	@echo "  make run-all      - Run with multiple sizes"
	@echo "  make debug        - Build and run cuBLAS debug test"
	@echo "  make debug-test   - Build and run full debug test (64x64x64 with all 3 nmSparse variants)"
	@echo "  make profile      - Profile with Nsight Compute"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean-all    - Remove all artifacts including debug"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Custom run:"
	@echo "  ./build/test_sparse_kernels <M> <N> <K> [warmup] [iterations]"
	@echo "  Example: ./build/test_sparse_kernels 2048 2048 2048 10 100"

.PHONY: all run run-small run-large run-all debug debug-test profile clean clean-all help
