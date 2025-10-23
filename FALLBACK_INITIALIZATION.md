# GPT-2 Fallback Initialization

## Overview
The training code now includes comprehensive fallback mechanisms to handle missing checkpoint files. The system will gracefully degrade and initialize the model with random weights if necessary.

## Features Added

### 1. Random Initialization Function (`gpt2_init_random`)
Creates a GPT-2 model from scratch with random weights when no checkpoint is available.

**Parameters:**
- `max_seq_len`: Maximum sequence length (e.g., 1024)
- `vocab_size`: Vocabulary size (e.g., 50257 for GPT-2)
- `num_layers`: Number of transformer layers (e.g., 12 for GPT-2 124M)
- `num_heads`: Number of attention heads (e.g., 12 for GPT-2 124M)
- `channels`: Embedding dimension (e.g., 768 for GPT-2 124M)

**Initialization:**
- Weights are randomly initialized in the range [-0.02, 0.02]
- Vocab size is automatically padded to a multiple of 128 for efficiency
- Uses a seeded RNG for reproducibility

### 2. Enhanced Checkpoint Loading (`gpt2_build_from_checkpoint`)

**Supported Checkpoint Formats:**
- **Version 3**: float32 format (`.bin` files)
- **Version 5**: bfloat16 format (`_bf16.bin` files)

**Fallback Chain:**
The function now handles multiple failure scenarios:

1. **File not found** → Falls back to random initialization
2. **File cannot be opened** → Falls back to random initialization
3. **Invalid magic number** → Falls back to random initialization
4. **Unsupported version** → Falls back to random initialization

### 3. BFloat16 Support

Added helper functions to handle bfloat16 checkpoints:
- `bf16_to_fp32()`: Converts bfloat16 values to float32
- `read_parameters()`: Reads parameters from file, automatically handling both fp32 and bf16 formats

### 4. Smart Checkpoint Discovery (in `main()`)

The main function now tries to load checkpoints in order of preference:

```c
1. gpt2_124M.bin (float32)
2. gpt2_124M_bf16.bin (bfloat16)
3. Random initialization (if neither exists)
```

## Usage Examples

### Example 1: Standard Usage (with checkpoint)
```bash
# If gpt2_124M.bin exists
./train_gpt2
# Output: "Found checkpoint: gpt2_124M.bin"
```

### Example 2: Using BFloat16 Checkpoint
```bash
# If only gpt2_124M_bf16.bin exists
./train_gpt2
# Output: "Found checkpoint: gpt2_124M_bf16.bin"
# Output: "[GPT-2] Loading checkpoint: gpt2_124M_bf16.bin (version 5 - bfloat16)"
```

### Example 3: No Checkpoint Available
```bash
# If no checkpoint files exist
./train_gpt2
# Output: "No checkpoint files found. Using random initialization."
# Output: "Checkpoint file not found: gpt2_124M.bin"
# Output: "Falling back to random initialization with default GPT-2 124M hyperparameters"
```

### Example 4: Programmatic Random Initialization
```c
GPT2 model;
// Initialize GPT-2 Small (custom config)
gpt2_init_random(&model, 1024, 50257, 6, 6, 384);
```

## Default Hyperparameters

When falling back to random initialization, the following GPT-2 124M hyperparameters are used:

| Parameter | Value |
|-----------|-------|
| max_seq_len | 1024 |
| vocab_size | 50257 |
| padded_vocab_size | 50304 (rounded to multiple of 128) |
| num_layers | 12 |
| num_heads | 12 |
| channels | 768 |
| num_parameters | ~124M |

## File Format Compatibility

### Float32 Checkpoints (Version 3)
- File: `gpt2_124M.bin`
- Format: 32-bit floating point
- Size: ~4 bytes per parameter

### BFloat16 Checkpoints (Version 5)
- File: `gpt2_124M_bf16.bin`
- Format: 16-bit brain floating point
- Size: ~2 bytes per parameter
- Converted to fp32 on load for computation

### Debug State Files
Debug state files (`gpt2_124M_debug_state.bin`) are optional and not required for training. They are only used for debugging and validation purposes.

## Benefits

1. **Robustness**: Training can proceed even without pre-trained weights
2. **Flexibility**: Supports multiple checkpoint formats automatically
3. **Development**: Easy to start training from scratch for experimentation
4. **Storage**: Can use smaller bf16 checkpoints to save disk space
5. **Error Recovery**: Graceful degradation instead of crashes

## Notes

- Random initialization will not produce good results immediately (high loss)
- Training from scratch requires more iterations to converge
- Pre-trained checkpoints are recommended for best results
- The fallback is primarily for development and debugging scenarios
