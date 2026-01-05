# ===============================
# Compiler settings
# ===============================
CC ?= g++
CFLAGS = -O2 -Wall -Wextra -std=c++17
LDFLAGS =
LDLIBS =
INCLUDES =

# ===============================
# CUDA / NVCC settings (Modal exact libs)
# ===============================
NVCC ?= /usr/local/cuda/bin/nvcc
FORCE_NVCC_O ?= 3
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O) -Wno-deprecated-gpu-targets
NVCC_LDFLAGS = -L/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib \
               -L/usr/lib/x86_64-linux-gnu \
               -L/usr/local/cuda/lib64 \
               -Xlinker -rpath=/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib \
               -Xlinker -rpath=/usr/lib/x86_64-linux-gnu \
               -Xlinker -rpath=/usr/local/cuda/lib64
NVCC_LDLIBS = -lcublas -lcublasLt -lnvidia-ml  # Note: -lnvidia-ml not -lnvml
NVCC_INCLUDES = -I/usr/local/lib/python3.12/site-packages/nvidia/cublas/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/cudart/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/nvtx/include \
                -I/usr/local/cuda/include \
                -I/usr/local/cuda/targets/x86_64-linux/include

USE_CUDNN ?= 0
BUILD_DIR = build

ifeq ($(USE_CUDNN),1)
  NVCC_INCLUDES += -I/usr/local/lib/python3.12/site-packages/nvidia/cudnn/include
  NVCC_LDFLAGS  += -L/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib \
                   -Xlinker -rpath=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib
  NVCC_LDLIBS   += -lcudnn
  NVCC_FLAGS    += -DENABLE_CUDNN
  NVCC_CUDNN = $(BUILD_DIR)/cudnn_att.o
  $(info → cuDNN enabled)
else
  $(info → cuDNN disabled by default)
endif

# ===============================
# Precision settings
# ===============================
PRECISION ?= BF16
VALID_PRECISIONS := FP32 FP16 BF16
ifeq ($(filter $(PRECISION),$(VALID_PRECISIONS)),)
  $(error Invalid precision $(PRECISION), valid precisions are $(VALID_PRECISIONS))
endif

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
TARGETS = train_gpt2cu test_gpt2cu train_gpt2fp32cu

.PHONY: all clean
all: $(TARGETS)

$(NVCC_CUDNN): llmc/cudnn_att.cpp
	$(NVCC) -c $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $< -o $@

train_gpt2cu: train_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $^ $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

train_gpt2fp32cu: train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) $^ $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

test_gpt2cu: test_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $^ $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

clean:
	rm -f $(BUILD_DIR)/*.o *.o *.out train_gpt2cu train_gpt2fp32cu test_gpt2cu