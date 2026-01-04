# ===============================
# Compiler settings
# ===============================
CC ?= cl
CFLAGS = /Idev /Zi /nologo /W4 /WX- /diagnostics:column /sdl /O2 /Oi /Ot /GL /D _DEBUG /D _CONSOLE /D _UNICODE /D UNICODE /Gm- /EHsc /MD /GS /Gy /fp:fast /Zc:wchar_t /Zc:forScope /Zc:inline /permissive- \
 /external:W3 /Gd /TP /wd4996 /Fd$@.pdb /FC /openmp:llvm
LDFLAGS =
LDLIBS =
INCLUDES =
CFLAGS_COND =

# ===============================
# CUDA / NVCC settings
# ===============================
FORCE_NVCC_O ?= 3
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O)
NVCC_LDFLAGS =
NVCC_LDLIBS = -lcublas -lcublasLt -lnvml
NVCC_INCLUDES =
NVCC_CUDNN =

USE_CUDNN ?= 0
BUILD_DIR = build

# ===============================
# Windows setup
# ===============================
ifeq ($(OS), Windows_NT)
  $(shell if not exist $(BUILD_DIR) mkdir $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := del $(BUILD_DIR)\*.obj
  REMOVE_FILES = del *.exe *.obj *.lib *.exp *.pdb
  OUTPUT_FILE = /link /OUT:$@
  CUDA_OUTPUT_FILE = -o $@ && copy /Y $@.exe $@
else
  $(shell mkdir -p $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := rm -f $(BUILD_DIR)/*.o
  REMOVE_FILES = rm -f
  OUTPUT_FILE = -o $@
  CUDA_OUTPUT_FILE = -o $@
endif

# ===============================
# FORCE REAL NVCC (fixes bug)
# ===============================
NVCC := "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe"

# ===============================
# cuDNN (Windows)
# ===============================
ifeq ($(USE_CUDNN),1)
ifeq ($(OS),Windows_NT)

  ifeq ($(shell if exist "$(HOMEDRIVE)$(HOMEPATH)\cudnn-frontend\include" (echo exists)),exists)
    CUDNN_FRONTEND_PATH = $(HOMEDRIVE)$(HOMEPATH)\cudnn-frontend\include
  else ifeq ($(shell if exist "cudnn-frontend\include" (echo exists)),exists)
    CUDNN_FRONTEND_PATH = cudnn-frontend/include
  else
    $(error [ERROR] cuDNN frontend not found. See README)
  endif

  CUDNN_INCLUDE_PATH = -I"C:\Program Files\NVIDIA\CUDNN\v9.17\include\13.1"
  CUDNN_LIB_PATH = -L"C:\Program Files\NVIDIA\CUDNN\v9.17\lib\13.1\x64"

  NVCC_INCLUDES += -I$(CUDNN_FRONTEND_PATH) $(CUDNN_INCLUDE_PATH)
  NVCC_LDFLAGS += $(CUDNN_LIB_PATH)
  NVCC_LDLIBS += -lcudnn
  NVCC_FLAGS += -DENABLE_CUDNN

  NVCC_CUDNN = $(BUILD_DIR)\cudnn_att.obj
endif
else
  $(info → cuDNN disabled by default. Run make USE_CUDNN=1 to enable)
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
TARGETS = train_gpt2 test_gpt2 train_gpt2cu train_gpt2rawcu train_gpt3cu test_gpt2cu train_gpt2fp32cu test_gpt2fp32cu $(NVCC_CUDNN)

# Quantized training targets
TARGETS_Q115 = train_gpt2q115cu train_gpt3q115cu
# Q1.15 weight-constrained training (bf16 compute, weights clamped to Q1.15 range)
TARGETS_Q115_CONSTRAINED = train_gpt2q115_constrainedcu train_gpt3q115_constrainedcu

.PHONY: all clean q115 q115_constrained
all: $(TARGETS)

q115: $(TARGETS_Q115)

q115_constrained: $(TARGETS_Q115_CONSTRAINED)

# ===============================
# CPU targets
# ===============================
train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

test_gpt2: test_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

# ===============================
# CUDA targets
# ===============================
$(NVCC_CUDNN): llmc/cudnn_att.cpp
	$(NVCC) -c $(NVCC_FLAGS) $(PFLAGS) $< $(NVCC_INCLUDES) -o $@

train_gpt2cu: train_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt2rawcu: train_gpt2_raw.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt3cu: train_gpt3.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt2fp32cu: train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

test_gpt2cu: test_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

test_gpt2fp32cu: test_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

profile_gpt2cu: profile_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -lineinfo $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# ===============================
# Quantized CUDA targets (Q1.15)
# ===============================
train_gpt2q115cu: train_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -DENABLE_Q115 $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt3q115cu: train_gpt3.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -DENABLE_Q115 $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# ===============================
# Q1.15 Weight-Constrained CUDA targets (bf16 compute, weights in Q1.15 range)
# ===============================
train_gpt2q115_constrainedcu: train_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -DENABLE_Q115_WEIGHT_CONSTRAINT $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

train_gpt3q115_constrainedcu: train_gpt3.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -DENABLE_Q115_WEIGHT_CONSTRAINT $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# ===============================
# Clean
# ===============================
clean:
	$(REMOVE_FILES)
	$(REMOVE_BUILD_OBJECT_FILES)
