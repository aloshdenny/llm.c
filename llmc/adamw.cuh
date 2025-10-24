/*
AdamW kernel
*/

// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#ifdef ENABLE_Q115
#include "q115_common.cuh"
#endif

// ----------------------------------------------------------------------------
// CUDA kernels

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

template <typename Tp, typename Tg>
__device__ void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             float grad_scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }  // guard

    // get the gradient, m, and v for this parameter
    float grad = grad_scale * (float)grads_memory[idx];
    float m = m_memory[idx];
    float v = v_memory[idx];
    // update the first moment (momentum)
    m = lerp(grad, m, beta1);
    m_memory[idx] = m;
    // update the second moment (RMSprop)
    v = lerp(grad * grad, v, beta2);
    v_memory[idx] = v;
    m /= beta1_correction;  // m_hat
    v /= beta2_correction;  // v_hat
    
    // fetch the old value of this parameter as a float
    float old_param;
#ifdef ENABLE_Q115
    // In Q1.15 mode, Tp is int16_t (q115_t)
    if constexpr (sizeof(Tp) == 2) {
        // Convert from Q1.15 to float
        old_param = q115_to_float(params_memory[idx]);
        // Use master weights if available (for higher precision accumulation)
        if (master_params_memory != NULL) {
            old_param = master_params_memory[idx];
        }
    } else {
        old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
    }
#else
    old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
#endif
    
    // update this parameter
    float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));
    
    // Store updated parameter
#ifdef ENABLE_Q115
    if constexpr (sizeof(Tp) == 2) {
        // Convert from float to Q1.15
        params_memory[idx] = float_to_q115(param);
    } else {
        // use stochastic rounding for non-Q1.15 types
        stochastic_rounding(param, &params_memory[idx], seed);
    }
#else
    // use stochastic rounding to go from float to lower precision
    stochastic_rounding(param, &params_memory[idx], seed);
#endif
    
    // write the full, float version of the param into our master copy, if we maintain one
    // this will be used in the next update
    if (master_params_memory != NULL) { 
        master_params_memory[idx] = param; 
    }
}

template <typename Tp, typename Tg>
__global__ void adamw_kernel3(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                              ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              float grad_scale, unsigned int seed) {
    adamw_update(params_memory + blockIdx.y * w_stride,
                 master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL,
                 grads_memory + blockIdx.y * g_stride,
                 m_memory + blockIdx.y * s_stride,
                 v_memory + blockIdx.y * s_stride,
                 num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale,
                 seed
                 );
}

template <typename Tp>
__global__ void init_from_master_kernel(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                                          ptrdiff_t w_stride, ptrdiff_t s_stride, unsigned int seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }
    params_memory += blockIdx.y * w_stride; // adjust for layer offset
    master_params_memory += blockIdx.y * s_stride;
    stochastic_rounding(master_params_memory[idx], &params_memory[idx], seed);
}

template <typename Tp, typename Tg>
void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, float* m_memory, float* v_memory, size_t num_parameters,
                  ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  float grad_scale, unsigned int seed, cudaStream_t stream) {
    // AdamW update
    int block_size = 512;
    int num_blocks = CEIL_DIV(num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    adamw_kernel3<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory, grads_memory,
                                                         m_memory, v_memory, num_parameters, w_stride, g_stride, s_stride,
                                                         learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay,
                                                         grad_scale, seed);
    cudaCheck(cudaGetLastError());
}

template <typename Tp>
void init_from_master(Tp* params_memory, float* master_params_memory, size_t num_parameters,
                        ptrdiff_t w_stride, ptrdiff_t s_stride, int num_slices, unsigned int seed, cudaStream_t stream) {
    int block_size = 512; // must match block size of adamw_update so that RNG also matches
    int num_blocks = CEIL_DIV(num_parameters, block_size);
    init_from_master_kernel<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>
                             (params_memory, master_params_memory, num_parameters, w_stride, s_stride, seed);
    cudaCheck(cudaGetLastError());
}
