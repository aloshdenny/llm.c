/*
Q1.15 Fixed-Point Arithmetic Utilities for CUDA
Provides device functions for Q1.15 format operations
*/

#ifndef Q115_COMMON_CUH
#define Q115_COMMON_CUH

#include <cuda_runtime.h>
#include <stdint.h>

// Q1.15 type definition: signed 16-bit integer representing range [-1, 1)
typedef int16_t q115_t;

// Q1.15 constants
#define Q115_SCALE 32768.0f          // 2^15
#define Q115_MAX 32767               // Maximum Q1.15 value (0.999969...)
#define Q115_MIN -32768              // Minimum Q1.15 value (-1.0)
#define Q115_OVERFLOW_THRESHOLD 0.999f  // Clamp values to prevent overflow

// ----------------------------------------------------------------------------
// Q1.15 Conversion Functions

// Convert float to Q1.15 with clamping
__device__ __forceinline__ q115_t float_to_q115(float x) {
    // Clamp input to valid range
    x = fmaxf(-Q115_OVERFLOW_THRESHOLD, fminf(Q115_OVERFLOW_THRESHOLD, x));
    // Scale and round to nearest integer
    float scaled = x * Q115_SCALE;
    int32_t rounded = __float2int_rn(scaled);
    // Additional safety clamp
    return (q115_t)max(Q115_MIN, min(Q115_MAX, rounded));
}

// Convert Q1.15 to float
__device__ __forceinline__ float q115_to_float(q115_t x) {
    return __int2float_rn(x) / Q115_SCALE;
}

// ----------------------------------------------------------------------------
// Q1.15 Arithmetic Operations

// Q1.15 multiplication: (a * b) >> 15
// Uses 32-bit intermediate result to prevent overflow
__device__ __forceinline__ q115_t q115_mul(q115_t a, q115_t b) {
    int32_t result = ((int32_t)a * (int32_t)b) >> 15;
    return (q115_t)max(Q115_MIN, min(Q115_MAX, result));
}

// Q1.15 addition with saturation
__device__ __forceinline__ q115_t q115_add(q115_t a, q115_t b) {
    int32_t result = (int32_t)a + (int32_t)b;
    return (q115_t)max(Q115_MIN, min(Q115_MAX, result));
}

// Q1.15 subtraction with saturation
__device__ __forceinline__ q115_t q115_sub(q115_t a, q115_t b) {
    int32_t result = (int32_t)a - (int32_t)b;
    return (q115_t)max(Q115_MIN, min(Q115_MAX, result));
}

// Q1.15 negation with saturation
__device__ __forceinline__ q115_t q115_neg(q115_t a) {
    // Special case: -(-1.0) saturates to max value
    if (a == Q115_MIN) return Q115_MAX;
    return -a;
}

// ----------------------------------------------------------------------------
// Vectorized Q1.15 Operations (for performance)

// Load 2 Q1.15 values as int32_t
__device__ __forceinline__ int32_t load_q115x2(const q115_t* ptr) {
    return *reinterpret_cast<const int32_t*>(ptr);
}

// Store 2 Q1.15 values from int32_t
__device__ __forceinline__ void store_q115x2(q115_t* ptr, int32_t val) {
    *reinterpret_cast<int32_t*>(ptr) = val;
}

// Load 4 Q1.15 values as int64_t
__device__ __forceinline__ int64_t load_q115x4(const q115_t* ptr) {
    return *reinterpret_cast<const int64_t*>(ptr);
}

// Store 4 Q1.15 values from int64_t
__device__ __forceinline__ void store_q115x4(q115_t* ptr, int64_t val) {
    *reinterpret_cast<int64_t*>(ptr) = val;
}

// ----------------------------------------------------------------------------
// Helper Functions

// Check if value will overflow in Q1.15
__device__ __forceinline__ bool q115_will_overflow(float x) {
    return (x >= 1.0f) || (x <= -1.0f);
}

// Saturating conversion with overflow detection
__device__ __forceinline__ q115_t float_to_q115_saturate(float x, int* overflow_count = nullptr) {
    bool overflow = q115_will_overflow(x);
    if (overflow && overflow_count != nullptr) {
        atomicAdd(overflow_count, 1);
    }
    return float_to_q115(x);
}

// ----------------------------------------------------------------------------
// Mixed Precision Operations (Q1.15 with float accumulation)

// Multiply-accumulate: acc += a * b (Q1.15 inputs, float accumulator)
__device__ __forceinline__ void q115_mac_float(float& acc, q115_t a, q115_t b) {
    acc += q115_to_float(a) * q115_to_float(b);
}

// Dot product helper (Q1.15 vectors, float result)
__device__ __forceinline__ float q115_dot_product(const q115_t* a, const q115_t* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += q115_to_float(a[i]) * q115_to_float(b[i]);
    }
    return sum;
}

#endif // Q115_COMMON_CUH
