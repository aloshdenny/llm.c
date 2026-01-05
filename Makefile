# ===============================
# Compiler settings
# ===============================
CC ?= g++
CFLAGS = -O2 -Wall -Wextra -std=c++17
LDFLAGS =
LDLIBS =
INCLUDES =

# ===============================
# CUDA / NVCC settings (Modal/Conda NVIDIA)
# ===============================
NVCC ?= /usr/local/cuda/bin/nvcc
FORCE_NVCC_O ?= 3
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O)
NVCC_LDFLAGS = -L/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib -L/usr/local/lib/python3.12/site-packages/nvidia/libcublasLt/lib64
NVCC_LDLIBS = -lcublas -lcublasLt -lnvml
# Modal/Conda critical: explicit include paths FIRST
NVCC_INCLUDES = -I/usr/local/lib/python3.12/site-packages/nvidia/cublas/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/cudart/include \
                -I/usr/local/cuda/include

USE_CUDNN ?= 0
BUILD_DIR = build

ifeq ($(USE_CUDNN),1)
  NVCC_INCLUDES += -I/usr/local/lib/python3.12/site-packages/nvidia/cudnn/include
  NVCC_LDFLAGS  += -L/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib
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

# ===============================
# CUDA targets (INCLUDES before sources!)
# ===============================
$(NVCC_CUDNN): llmc/cudnn_att.cpp
	$(NVCC) -c $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $< -o $@

train_gpt2cu: train_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $^ $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

train_gpt2fp32cu: train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) $^ $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

test_gpt2cu: test_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $^ $(NVCC_LDFLAGS) $(NVCC_LDLIBS) -o $@

# ===============================
# Clean
# ===============================
clean:
	rm -f $(BUILD_DIR)/*.o *.o *.out train_gpt2cu train_gpt2fp32cu test_gpt2cu