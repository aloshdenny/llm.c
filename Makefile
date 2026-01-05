# ===============================
# Compiler settings (Modal llm.c)
# ===============================
CC ?= g++
CFLAGS = -O2 -Wall -Wextra -std=c++17

# ===============================
# CUDA settings (exact Modal paths)
# ===============================
NVCC ?= /usr/local/cuda/bin/nvcc
FORCE_NVCC_O ?= 3
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O) -Wno-deprecated-gpu-targets
NVCC_INCLUDES = -I/usr/local/lib/python3.12/site-packages/nvidia/cublas/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/cudart/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/nvtx/include \
                -I/usr/local/cuda/include \
                -I/usr/local/cuda/targets/x86_64-linux/include

# EXACT rpaths for your libs
NVCC_LDFLAGS = -L/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib \
               -L/usr/lib/x86_64-linux-gnu \
               -Xlinker '-rpath=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64'
NVCC_LDLIBS = -lcublas -lcublasLt -lnvml

PRECISION ?= BF16
ifeq ($(PRECISION),FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION),FP16)
  PFLAGS = -DENABLE_FP16
else
  PFLAGS = -DENABLE_BF16
endif

# ===============================
# Targets
# ===============================
TARGETS = train_gpt2cu train_gpt2fp32cu

.PHONY: all clean run test

all: $(TARGETS)

train_gpt2cu: train_gpt2.cu
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

train_gpt2fp32cu: train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

# Test execution
run: train_gpt2cu
	@ldd ./train_gpt2cu | grep cublas || echo "Libs linked!"
	./train_gpt2cu --help

test: train_gpt2cu
	@echo "✅ Build success! Binary size:"
	@ls -lh train_gpt2cu
	@echo "✅ Libs found:"
	@ldd ./train_gpt2cu | grep -E "(cublas|nvml)"
	@echo "✅ Runs:"
	@./train_gpt2cu --help | head -5

clean:
	rm -f *.o *.out