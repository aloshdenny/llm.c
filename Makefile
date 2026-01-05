# ===============================
# Compiler settings
# ===============================
CC ?= g++
CFLAGS = -O2 -Wall -Wextra -std=c++17

# ===============================
# CUDA / NVCC settings (symlink method)
# ===============================
NVCC ?= /usr/local/cuda/bin/nvcc
FORCE_NVCC_O ?= 3
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O) -Wno-deprecated-gpu-targets
NVCC_INCLUDES = -I/usr/local/lib/python3.12/site-packages/nvidia/cublas/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/cudart/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/nvtx/include \
                -I/usr/local/cuda/include \
                -I/usr/local/cuda/targets/x86_64-linux/include

# Local symlinks + rpath trick
NVCC_LDFLAGS = -L. -L/usr/lib/x86_64-linux-gnu \
               -Xlinker -rpath=.
NVCC_LDLIBS = -lcublas -lcublasLt -lnvml

USE_CUDNN ?= 0
BUILD_DIR = build
PRECISION ?= BF16

# ===============================
# Pre-build: symlink libs locally
# ===============================
.PHONY: libsyms
libsyms:
	ln -sf /usr/local/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12 ./libcublas.so
	ln -sf /usr/local/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12 ./libcublasLt.so
	ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 ./libnvml.so

# ===============================
# Targets
# ===============================
TARGETS = train_gpt2cu train_gpt2fp32cu

all: libsyms $(TARGETS)

ifeq ($(PRECISION),FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION),FP16)
  PFLAGS = -DENABLE_FP16
else
  PFLAGS = -DENABLE_BF16
endif

train_gpt2cu: train_gpt2.cu libsyms
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

train_gpt2fp32cu: train_gpt2_fp32.cu libsyms
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

clean:
	rm -f libcublas.so libcublasLt.so libnvml.so *.o train_gpt2* $(BUILD_DIR)/*

.PHONY: run
run: train_gpt2cu
	LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH ./train_gpt2cu --help