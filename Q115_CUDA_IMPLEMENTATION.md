## 1. Core CUDA Files to Edit

### Primary File:
- **train_gpt2.cu** - Main training loop and model structure
  - Lines 67-300: Add Q1.15 type definitions and GPU config
  - Lines 280-450: Update `GPT2Config`, `ParameterTensors`, `ActivationTensors` structs
  - Lines 500-600: Modify checkpoint loading in `gpt2_build_from_checkpoint`
  - Lines 800-1100: Update `gpt2_forward` function calls
  - Lines 1100-1300: Update `gpt2_backward_and_reduce` function calls

### CUDA Kernel Files (in llmc directory):

1. **encoder.cuh** - Embedding layer
   - Update `encoder_forward_kernel` to use Q1.15
   - Update `encoder_backward_kernel` to use float gradients

2. **layernorm.cuh** - Layer normalization
   - Update forward kernels for Q1.15 input/output
   - Keep mean/rstd in float
   - Update backward kernels for float gradients

3. **matmul.cuh** - Matrix multiplication
   - Update `matmul_forward` to use Q1.15 with float accumulation
   - Update `matmul_backward` for float gradients
   - Modify cuBLAS calls to handle mixed precision

4. **attention.cuh** (or cudnn_att.cpp if using cuDNN)
   - Update QKV projection to Q1.15
   - Keep attention scores (softmax) in float
   - Update backward pass for float gradients

5. **gelu.cuh** (likely in matmul.cuh)
   - Update `gelu_forward` kernel for Q1.15 I/O
   - Update `gelu_backward_inplace` for float gradients

6. **fused_classifier.cuh** - Final layer
   - Update logits computation to Q1.15
   - Keep softmax/cross-entropy in float

7. **adamw.cuh** - Optimizer (already open)
   - Update `adamw_kernel3` template to accept Q1.15 params
   - Add conversion from float gradients → Q1.15 parameter updates
   - Keep m_memory and v_memory in float

## 2. Common Utility Files to Edit

8. **cuda_common.h** - CUDA common definitions
   - Add Q1.15 type definitions (`typedef int16_t q115_t`)
   - Add Q1.15 constants (scale, max, min)
   - Add macro guards for Q1.15 mode

9. **cuda_utils.cuh** - CUDA utilities
   - Add `__device__` Q1.15 conversion functions:
     ```cuda
     __device__ inline q115_t float_to_q115(float x);
     __device__ inline float q115_to_float(q115_t x);
     __device__ inline q115_t q115_mul(q115_t a, q115_t b);
     __device__ inline q115_t q115_add(q115_t a, q115_t b);
     ```
   - Add vectorized Q1.15 ops using `int2`/`int4` for performance

## 3. Supporting Files

10. **cublas_common.h** - cuBLAS configuration
    - May need to handle mixed-precision matmul (if cuBLAS supports int16)
    - Otherwise, custom CUDA kernels for Q1.15 matmul

11. **global_norm.cuh** - Gradient norm
    - Keep gradient norm computation in float (no changes needed)

12. **zero.cuh** - Multi-GPU gradients
    - Update gradient reduction to handle float gradients
    - Update parameter sharding to handle Q1.15 params

## 4. Header/Config Files

13. **Create: `llm.c/llmc/q115_common.cuh`** (new file)
    - Centralize all Q1.15 CUDA device functions
    - Include conversion, arithmetic, and vector operations
    - Import this in other .cuh files

## 5. Build System

14. **Makefile** - Build configuration
    - Lines 250-290: Add Q1.15 compilation flags
    - Example: `-DENABLE_Q115` flag to enable/disable

## Key Implementation Steps

### Step 1: Add Q1.15 CUDA Functions
Create `llm.c/llmc/q115_common.cuh`:

````cuda
#ifndef Q115_COMMON_H
#define Q115_COMMON_H

#include <cuda_runtime.h>
#include <stdint.h>

typedef int16_t q115_t;

#define Q115_SCALE 32768.0f
#define Q115_MAX 32767
#define Q115_MIN -32768
#define Q115_OVERFLOW_THRESHOLD 0.95f

__device__ __forceinline__ q115_t float_to_q115(float x) {
    x = fmaxf(-Q115_OVERFLOW_THRESHOLD, fminf(Q115_OVERFLOW_THRESHOLD, x));
    float scaled = x * Q115_SCALE;
    int32_t rounded = __float2int_rn(scaled);
    return (q115_t)max(Q115_MIN, min(Q115_MAX, rounded));
}

__device__ __forceinline__ float q115_to_float(q115_t x) {
    return __int2float_rn(x) / Q115_SCALE;
}

__device__ __forceinline__ q115_t q115_mul(q115_t a, q115_t b) {
    int32_t result = (__mul24(a, b)) >> 15;
    return (q115_t)max(Q115_MIN, min(Q115_MAX, result));
}

__device__ __forceinline__ q115_t q115_add(q115_t a, q115_t b) {
    int32_t result = (int32_t)a + (int32_t)b;
    return (q115_t)max(Q115_MIN, min(Q115_MAX, result));
}

#endif
````

### Step 2: Update Encoder Kernel Example

````cuda
#include "q115_common.cuh"

__global__ void encoder_forward_kernel(q115_t* out,
                                       const int* inp, const q115_t* wte, const q115_t* wpe,
                                       int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * T * C;
    
    if (idx < N) {
        int bt = idx / C;
        int b = bt / T;
        int t = bt % T;
        int c = idx % C;
        
        int ix = inp[b * T + t];
        
        q115_t wte_val = wte[ix * C + c];
        q115_t wpe_val = wpe[t * C + c];
        
        out[idx] = q115_add(wte_val, wpe_val);
    }
}
````

### Step 3: Update AdamW Kernel

````cuda
#include "q115_common.cuh"

template <typename Tp, typename Tg>
__device__ void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, 
                             float* m_memory, float* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, 
                             float beta1_correction, float beta2_correction, 
                             float eps, float weight_decay, float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }

    float grad = grad_scale * (float)grads_memory[idx];
    float m = m_memory[idx];
    float v = v_memory[idx];
    
    m = lerp(grad, m, beta1);
    m_memory[idx] = m;
    
    v = lerp(grad * grad, v, beta2);
    v_memory[idx] = v;
    
    m /= beta1_correction;
    v /= beta2_correction;
    
    // Get old parameter value (convert from Q1.15 if needed)
    float old_param;
    if constexpr (sizeof(Tp) == 2) { // Q1.15
        old_param = q115_to_float(params_memory[idx]);
    } else {
        old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
    }
    
    float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));
    
    // Store updated parameter (convert to Q1.15 if needed)
    if constexpr (sizeof(Tp) == 2) { // Q1.15
        params_memory[idx] = float_to_q115(param);
    } else {
        stochastic_rounding(param, &params_memory[idx], seed);
    }
    
    if (master_params_memory != NULL) { 
        master_params_memory[idx] = param; 
    }
}
````

### Step 4: Update Main Structures in train_gpt2.cu

````cuda
#include "llmc/q115_common.cuh"

typedef struct {
    q115_t* wte; // (V, C) - changed from floatX*
    q115_t* wpe; // (maxT, C) - changed from floatX*
    // ... all other params as q115_t*
} ParameterTensors;

typedef struct {
    q115_t* encoded; // (B, T, C) - changed from floatX*
    // ... some activations q115_t*, some float* (like att, ln_mean, etc.)
} ActivationTensors;
````

## Testing Strategy

1. **Compile with flag**: `make train_gpt2cu ENABLE_Q115=1`
2. **Start small**: Test encoder → layernorm → matmul individually
3. **Compare outputs**: Run float32 and Q1.15 versions side-by-side
4. **Monitor saturation**: Add counters for overflow/underflow events
5. **Validate training**: Check loss curves match reasonably

## Estimated Effort

- **Q1.15 utilities**: 1-2 hours
- **Kernel updates**: 4-6 hours (7 kernel files)
- **Optimizer update**: 1-2 hours
- **Testing/debugging**: 4-8 hours
- **Total**: ~2-3 days of focused work

The CUDA implementation is more complex than CPU due to kernel optimizations, but the core logic remains similar to the CPU version you already implemented.