# Q1.15 Implementation TODO List

## Critical Remaining Tasks

### 1. Fix Gradient Activation Structure
**Problem**: `grads_acts` needs to be all-float but points to mixed-type `ActivationTensors`

**Solution**: Create a separate `ActivationTensorsGrad` struct (all float):
```c
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    // ... all other activations as float
} ActivationTensorsGrad;
```

Then update `GPT2` struct:
```c
ActivationTensorsGrad grads_acts;
```

### 2. Create Float Activation Allocator
```c
float* malloc_and_point_activation_grads(ActivationTensorsGrad* acts, size_t* act_sizes) {
    // Similar to current allocator but all float
}
```

### 3. Update gpt2_zero_grad()
```c
void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { 
        memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); 
    }
    if(model->grads_acts_memory != NULL) { 
        // Calculate total float activation size
        size_t total_float_acts = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            total_float_acts += model->act_sizes[i];
        }
        memset(model->grads_acts_memory, 0, total_float_acts * sizeof(float)); 
    }
}
```

### 4. Update gpt2_update() Optimizer
**Problem**: Need to update Q1.15 parameters from float gradients

```c
void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // ... Adam computation in float ...
    for (size_t i = 0; i < model->num_parameters; i++) {
        float param = q115_to_float(model->params_memory[i]);
        float grad = model->grads_memory[i];
        
        // Update m and v (float)
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        
        // Bias correction
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));
        
        // Update in float
        float updated_param = param - learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
        
        // Store updated moments
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        
        // Convert back to Q1.15
        model->params_memory[i] = float_to_q115(updated_param);
    }
}
```

### 5. Update gpt2_backward() Function Calls
All the backward function calls in `gpt2_backward()` need to be checked for proper type usage:

- `crossentropy_softmax_backward()`: ✅ Already correct (dlogits: float, logits: Q1.15)
- `matmul_backward()`: ✅ Already updated
- `layernorm_backward()`: ✅ Already updated
- `residual_backward()`: ✅ Already updated
- `attention_backward()`: ✅ Already updated
- `gelu_backward()`: ✅ Already updated
- `encoder_backward()`: ✅ Already correct

### 6. Update gpt2_forward() Function Calls
All forward function calls need to be verified:

```c
// Layer loop - get Q1.15 pointers correctly
float* l_ln1w = params.ln1w + l * C;  // Should be: q115_t* l_ln1w = ...
```

All pointer arithmetic in gpt2_forward() for params should use `q115_t*` instead of `float*`.

### 7. Fix crossentropy_softmax_backward() Signature
Currently expects float logits, but we have Q1.15:

```c
void crossentropy_softmax_backward(float* dlogits,  // output gradients (float)
                                    float* dlosses, 
                                    float* probs,    // input probs (float)
                                    int* targets,
                                    int B, int T, int V, int Vp);
```

This is actually correct - `dlogits` are output gradients (float), not related to forward Q1.15 logits.

### 8. Compile and Test
```bash
cd /Users/aloshdenny/vscode/llm.c
make train_gpt2
```

Expected errors to fix:
- Type mismatches in gpt2_forward() pointer arithmetic
- Missing ActivationTensorsGrad definition
- Memory allocation size calculations

## Quick Reference: Type Summary

### Parameters:
- Storage: `q115_t*` (Q1.15)
- Gradients: `float*`
- Optimizer: `float*` (m_memory, v_memory)

### Activations (forward):
- Mixed: Some `q115_t*`, some `float*` (see ActivationTensors)

### Activation Gradients (backward):
- All: `float*`

### Intermediate Computations:
- Matmul accumulation: `float`
- Attention scores: `float`
- LayerNorm stats: `float`
- Softmax: `float`

## Testing Checklist

- [ ] Code compiles without errors
- [ ] Code compiles without warnings
- [ ] Random initialization runs
- [ ] Checkpoint loading works (converts to Q1.15)
- [ ] Forward pass completes
- [ ] Backward pass completes
- [ ] Optimizer update runs
- [ ] Loss decreases over iterations
- [ ] Generated text is coherent
- [ ] No NaN/Inf values in training
- [ ] Memory usage is ~50% of original

## Notes

1. The conversion factor (0.5x for checkpoints, 0.1 range for random init) may need tuning
2. Monitor training curves carefully for signs of precision loss
3. Consider adding saturation counters for debugging
4. May want to add runtime flag to disable Q1.15 (compile-time)
