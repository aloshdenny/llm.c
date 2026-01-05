NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O3 -Wno-deprecated-gpu-targets
NVCC_INCLUDES = -I/usr/local/lib/python3.12/site-packages/nvidia/cublas/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/cudart/include \
                -I/usr/local/lib/python3.12/site-packages/nvidia/nvtx/include \
                -I/usr/local/cuda/include

NVCC_LDFLAGS = -L/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib \
               -L/usr/lib/x86_64-linux-gnu \
               -Wl,--enable-new-dtags \
               -Wl,-rpath,/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib \
               -Wl,-rpath,/usr/lib/x86_64-linux-gnu

NVCC_LIBS = -lcublas -lcublasLt -lnvml

PRECISION ?= BF16
ifeq ($(PRECISION),BF16)
  PFLAGS = -DENABLE_BF16
endif

train_gpt2cu: train_gpt2.cu
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LIBS) -o $@

train_gpt2fp32cu: train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $(NVCC_INCLUDES) $< $(NVCC_LDFLAGS) $(NVCC_LIBS) -o $@

.PHONY: all clean test run

all: train_gpt2cu

test:
	@echo "Building..."
	make train_gpt2cu
	@echo "Size: $$(ls -lh train_gpt2cu)"
	@echo "Libs: $$(ldd train_gpt2cu | grep -E 'cublas|nvml' || echo 'OK')"
	@echo "Test run:"
	./train_gpt2cu --help | head -10

run: train_gpt2cu
	./train_gpt2cu --help

clean:
	rm -f train_gpt2cu train_gpt2fp32cu *.o