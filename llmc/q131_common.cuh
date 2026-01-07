/*
Q1.31 Fixed-Point Arithmetic Utilities for CUDA
Provides device functions for Q1.31 format operations

Q1.31 format:
- 32-bit signed integer representing values in [-1, 1)
- 1 sign bit, 31 fractional bits
- Much higher precision than Q1.15 (65536x more resolution)
- Scale factor: 2^31 = 2147483648

Key differences from Q1.15:
- Uses int32_t storage instead of int16_t
- MAC operations require int64_t accumulation
- Scale constant is 2^31 (stored as 64-bit to avoid overflow)
- ~2x memory usage compared to Q1.15
*/

#ifndef Q131_COMMON_CUH
#define Q131_COMMON_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// ============================================================================
// Compile-time guard: fail if Q1.15 code paths are referenced
// ============================================================================
#ifdef ENABLE_Q115
#error "Q1.15 mode is incompatible with Q1.31. Use ENABLE_Q131 only."
#endif

// ============================================================================
// Q1.31 Type and Constants
// ============================================================================

// Q1.31 type definition: signed 32-bit integer representing range [-1, 1)
typedef int32_t q131_t;

// Q1.31 scale as 64-bit to prevent overflow during conversions
// 2^31 = 2147483648
#define Q131_SCALE_BITS 31
#define Q131_SCALE 2147483648.0  // 2^31 as double for precision
#define Q131_SCALE_INT 2147483648LL  // 2^31 as int64_t

// Q1.31 value limits
#define Q131_MAX 2147483647       // Maximum Q1.31 value (0.9999999995...)
#define Q131_MIN (-2147483647-1)  // Minimum Q1.31 value (-1.0), written to avoid overflow
#define Q131_OVERFLOW_THRESHOLD 0.9999999f  // Clamp values to prevent overflow

// Q1.31 smallest non-zero value is approximately 4.66e-10 (1/2^31)
#define Q131_MIN_NONZERO (1.0 / Q131_SCALE)

// Threshold to detect if weights collapsed to Q1.15 levels
// If max absolute weight is < 2^16, weights are effectively Q1.15 range
#define Q131_Q115_COLLAPSE_THRESHOLD (1 << 16)

// ============================================================================
// Runtime Validation Macros
// ============================================================================

// Assert that weights haven't collapsed to Q1.15 levels
#define Q131_ASSERT_NOT_COLLAPSED(max_abs_weight) do { \
    if ((max_abs_weight) < Q131_Q115_COLLAPSE_THRESHOLD) { \
        printf("ERROR: Q1.31 weights collapsed to Q1.15 range! max_abs=%d\n", (max_abs_weight)); \
    } \
} while(0)

// ============================================================================
// Dynamic Scaling Configuration
// ============================================================================

// Scale factors for different tensor types
// Q1.31 has much more headroom, so we use larger scales
#define Q131_EMBEDDING_SCALE 1.0f
#define Q131_ATTENTION_SCALE 2.0f
#define Q131_FFN_SCALE 4.0f
#define Q131_LOGITS_SCALE 32.0f  // Larger range for logits

// Gradient scales
#define Q131_GRAD_EMBEDDING_SCALE 0.1f
#define Q131_GRAD_ATTENTION_SCALE 0.5f
#define Q131_GRAD_FFN_SCALE 0.5f
#define Q131_GRAD_LOGITS_SCALE 2.0f

// ============================================================================
// LayerNorm and Normalization Constants
// Q1.31 has better resolution, so we can use smaller epsilon
// ============================================================================
#define Q131_LAYERNORM_EPS 1e-5f  // Can use standard epsilon with Q1.31

// Activation clamping range (wider than Q1.15 due to better precision)
#define Q131_ACTIVATION_CLAMP_MIN -8.0f
#define Q131_ACTIVATION_CLAMP_MAX 8.0f

// ============================================================================
// Conversion Functions: Float <-> Q1.31
// ============================================================================

// Convert float to Q1.31 with clamping
// Uses round-to-nearest-even for best accuracy
__device__ __forceinline__ q131_t float_to_q131(float x) {
    // Clamp input to valid range
    x = fmaxf(-1.0f, fminf(Q131_OVERFLOW_THRESHOLD, x));
    // Scale and round to nearest integer using double precision
    // to avoid overflow in intermediate calculations
    double scaled = (double)x * Q131_SCALE;
    int64_t rounded = llrintf(scaled);  // Round to nearest, ties to even
    // Safety clamp to Q1.31 range
    rounded = max((int64_t)Q131_MIN, min((int64_t)Q131_MAX, rounded));
    return (q131_t)rounded;
}

// Convert Q1.31 to float
__device__ __forceinline__ float q131_to_float(q131_t x) {
    return (float)((double)x / Q131_SCALE);
}

// Convert float to Q1.31 with explicit scale factor
__device__ __forceinline__ q131_t float_to_q131_scaled(float x, float scale) {
    float normalized = x / scale;
    return float_to_q131(normalized);
}

// Convert Q1.31 to float with explicit scale factor
__device__ __forceinline__ float q131_to_float_scaled(q131_t x, float scale) {
    return q131_to_float(x) * scale;
}

// ============================================================================
// Host-side Conversion Functions (for CPU initialization)
// ============================================================================

// Host version of float to Q1.31 conversion
inline q131_t host_float_to_q131(float x) {
    // Clamp input to valid range
    if (x < -1.0f) x = -1.0f;
    if (x > 0.9999999f) x = 0.9999999f;
    // Scale using double precision
    double scaled = (double)x * Q131_SCALE;
    int64_t rounded = (int64_t)llrint(scaled);  // Round to nearest even
    // Safety clamp
    if (rounded > Q131_MAX) rounded = Q131_MAX;
    if (rounded < Q131_MIN) rounded = Q131_MIN;
    return (q131_t)rounded;
}

// Host version of Q1.31 to float conversion
inline float host_q131_to_float(q131_t x) {
    return (float)((double)x / Q131_SCALE);
}

// ============================================================================
// Q1.31 Arithmetic Operations with int64 Accumulation
// ============================================================================

// Q1.31 multiplication: (a * b) >> 31
// CRITICAL: Uses 64-bit intermediate to prevent overflow
__device__ __forceinline__ q131_t q131_mul(q131_t a, q131_t b) {
    int64_t result = ((int64_t)a * (int64_t)b) >> Q131_SCALE_BITS;
    return (q131_t)max((int64_t)Q131_MIN, min((int64_t)Q131_MAX, result));
}

// Q1.31 addition with saturation
__device__ __forceinline__ q131_t q131_add(q131_t a, q131_t b) {
    int64_t result = (int64_t)a + (int64_t)b;
    return (q131_t)max((int64_t)Q131_MIN, min((int64_t)Q131_MAX, result));
}

// Q1.31 subtraction with saturation
__device__ __forceinline__ q131_t q131_sub(q131_t a, q131_t b) {
    int64_t result = (int64_t)a - (int64_t)b;
    return (q131_t)max((int64_t)Q131_MIN, min((int64_t)Q131_MAX, result));
}

// Q1.31 negation with saturation
__device__ __forceinline__ q131_t q131_neg(q131_t a) {
    // Special case: -(-1.0) saturates to max value
    if (a == Q131_MIN) return Q131_MAX;
    return -a;
}

// ============================================================================
// Multiply-Accumulate Operations (MAC) with int64 Accumulation
// These are the critical operations for matmul kernels
// ============================================================================

// MAC with int64 accumulator: acc += a * b
// The accumulator stays in int64, only final result is shifted back
__device__ __forceinline__ void q131_mac_int64(int64_t& acc, q131_t a, q131_t b) {
    acc += (int64_t)a * (int64_t)b;
}

// Finalize int64 accumulator to Q1.31
// Shifts right by 31 bits and saturates
__device__ __forceinline__ q131_t q131_finalize_acc(int64_t acc) {
    int64_t result = acc >> Q131_SCALE_BITS;
    return (q131_t)max((int64_t)Q131_MIN, min((int64_t)Q131_MAX, result));
}

// MAC with float accumulator (for mixed precision paths)
__device__ __forceinline__ void q131_mac_float(float& acc, q131_t a, q131_t b) {
    acc += q131_to_float(a) * q131_to_float(b);
}

// Scaled MAC with float accumulator
__device__ __forceinline__ void q131_mac_float_scaled(float& acc, q131_t a, float scale_a, 
                                                       q131_t b, float scale_b) {
    acc += q131_to_float_scaled(a, scale_a) * q131_to_float_scaled(b, scale_b);
}

// ============================================================================
// Vectorized Q1.31 Operations (for performance)
// ============================================================================

// Load 2 Q1.31 values as int64_t (64-bit aligned load)
__device__ __forceinline__ int64_t load_q131x2(const q131_t* ptr) {
    return *reinterpret_cast<const int64_t*>(ptr);
}

// Store 2 Q1.31 values from int64_t
__device__ __forceinline__ void store_q131x2(q131_t* ptr, int64_t val) {
    *reinterpret_cast<int64_t*>(ptr) = val;
}

// Extract individual Q1.31 values from packed int64_t
__device__ __forceinline__ q131_t extract_q131_low(int64_t packed) {
    return (q131_t)(packed & 0xFFFFFFFF);
}

__device__ __forceinline__ q131_t extract_q131_high(int64_t packed) {
    return (q131_t)(packed >> 32);
}

// Pack two Q1.31 values into int64_t
__device__ __forceinline__ int64_t pack_q131x2(q131_t low, q131_t high) {
    return ((int64_t)(uint32_t)low) | (((int64_t)(uint32_t)high) << 32);
}

// ============================================================================
// Helper Functions for Numerical Safety
// ============================================================================

// Check if value will overflow in Q1.31
__device__ __forceinline__ bool q131_will_overflow(float x) {
    return (x >= 1.0f) || (x <= -1.0f);
}

// Check if value will overflow in Q1.31 given a scale
__device__ __forceinline__ bool q131_will_overflow_scaled(float x, float scale) {
    float normalized = x / scale;
    return q131_will_overflow(normalized);
}

// Get absolute value of Q1.31 (for statistics)
__device__ __forceinline__ uint32_t q131_abs(q131_t x) {
    return (x >= 0) ? (uint32_t)x : (uint32_t)(-x);
}

// ============================================================================
// Statistics and Debugging Helpers
// ============================================================================

// Atomic max for tracking max absolute weight (for collapse detection)
__device__ __forceinline__ void q131_atomic_max_abs(uint32_t* max_abs, q131_t x) {
    uint32_t abs_val = q131_abs(x);
    atomicMax(max_abs, abs_val);
}

// Kernel to compute max absolute Q1.31 value in a buffer
__global__ void q131_max_abs_kernel(const q131_t* data, size_t n, uint32_t* result) {
    uint32_t thread_max = 0;
    
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        uint32_t abs_val = q131_abs(data[i]);
        thread_max = max(thread_max, abs_val);
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    
    // First thread in each warp updates global max
    if ((threadIdx.x & 31) == 0) {
        atomicMax(result, thread_max);
    }
}

// Kernel to compute logit standard deviation (for training monitoring)
__global__ void q131_logit_std_kernel(const float* logits, size_t n, float* mean_out, float* std_out) {
    __shared__ float s_sum;
    __shared__ float s_sum_sq;
    
    if (threadIdx.x == 0) {
        s_sum = 0.0f;
        s_sum_sq = 0.0f;
    }
    __syncthreads();
    
    float thread_sum = 0.0f;
    float thread_sum_sq = 0.0f;
    
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float val = logits[i];
        thread_sum += val;
        thread_sum_sq += val * val;
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
    }
    
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&s_sum, thread_sum);
        atomicAdd(&s_sum_sq, thread_sum_sq);
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float mean = s_sum / n;
        float variance = (s_sum_sq / n) - (mean * mean);
        *mean_out = mean;
        *std_out = sqrtf(variance);
    }
}

// ============================================================================
// Dot Product Helpers
// ============================================================================

// Dot product with int64 accumulation (Q1.31 inputs, Q1.31 result)
__device__ __forceinline__ q131_t q131_dot_product(const q131_t* a, const q131_t* b, int n) {
    int64_t acc = 0;
    for (int i = 0; i < n; i++) {
        q131_mac_int64(acc, a[i], b[i]);
    }
    return q131_finalize_acc(acc);
}

// Dot product with float accumulation (Q1.31 inputs, float result)
__device__ __forceinline__ float q131_dot_product_float(const q131_t* a, const q131_t* b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        q131_mac_float(acc, a[i], b[i]);
    }
    return acc;
}

// Scaled dot product
__device__ __forceinline__ float q131_dot_product_scaled(const q131_t* a, float scale_a,
                                                          const q131_t* b, float scale_b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        q131_mac_float_scaled(acc, a[i], scale_a, b[i], scale_b);
    }
    return acc;
}

#endif // Q131_COMMON_CUH
