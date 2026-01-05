# ========================================
# llm.c Makefile for Modal GPU (tested)
# ========================================
# Requirements: profiler line commented in llmc/cuda_common.h:15
# Usage: export LD_LIBRARY_PATH=... then make && ./train_gpt2cu

CC ?= g++
CFLAGS = -O2 -Wall -Wextra -std=c++17

NVCC ?= /usr/local/cuda/bin/nvcc
FORCE_NVCC_O ?= 3
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O) -Wno-deprecated-gpu-targets
NVCC_LDFLAGS = -L/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib \
               -L/usr/lib/x86_64-linux-gnu \
               -L/usr/local/cuda/lib64
NVCC_LDLIBS = -lcublas -lcublasLt -lnvml -lcudart
NVCC_INCLUDES = -I/usr/local/lib/python3.12/site-packages/nvidia/cublas/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/cudart/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/nvtx/include \
                -I/usr/local/cuda/include \
                -I/usr/local/cuda/targets/x86_64-linux/include

USE_CUDNN ?= 0
BUILD_DIR = build
PRECISION ?= BF16

ifeq ($(USE_CUDNN),1)
  NVCC_INCLUDES += -I/usr/local/lib/python3.12/site-packages/nvidia/cudnn/include
  NVCC_LDFLAGS  += -L/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib
  NVCC_LDLIBS   += -lcudnn
  NVCC_FLAGS    += -DENABLE_CUDNN
  NVCC_CUDNN = $(BUILD_DIR)/cudnn_att.o
  $(info → cuDNN enabled)
endif

# Precision
PFLAGS = -DENABLE_BF16
ifeq ($(PRECISION),FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION),FP16)
  PFLAGS = -DENABLE_FP16
endif

TARGETS = train_gpt2cu train_gpt2fp32cu test_gpt2cu

.PHONY: all clean run

all: $(TARGETS)

$(NVCC_CUDNN): llmc/cudnn_att.cpp
	$(NVCC) -c $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $< -o $@

train_gpt2cu: train_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $^ $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

train_gpt2fp32cu: train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) $^ $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

test_gpt2cu: test_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $^ $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

# ========================================
# Modal Runtime Setup (CRITICAL)
# ========================================
run:
	@export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$$LD_LIBRARY_PATH && \
	 ./train_gpt2cu --help

clean:
	rm -f $(BUILD_DIR)/*.o *.o *.out train_gpt2cu train_gpt2fp32cu test_gpt2cu