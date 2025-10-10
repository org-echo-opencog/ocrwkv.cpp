# OpenCog ML Inference Engine for RWKV

This document describes the OpenCog ML inference engine implementation that provides a pure inference interface for RWKV models, designed for integration with OpenCog's cognitive architecture.

## Overview

The OpenCog ML inference engine serves as a bridge between RWKV neural language models and OpenCog's symbolic reasoning system. It provides:

- **Pure Inference Interface**: No training capabilities, focused solely on inference
- **Tensor Abstraction**: High-level tensor operations compatible with OpenCog's data structures
- **State Management**: Efficient state caching and management for recurrent inference
- **Multi-threaded Support**: Engine cloning for parallel inference
- **Error Handling**: Comprehensive error reporting and validation
- **Language Bindings**: Both C API and Python wrappers

## Architecture

The OpenCog ML system consists of several layers:

1. **Core C Library** (`opencog_ml.h/cpp`): Low-level inference engine
2. **Python Bindings** (`python/opencog_ml/`): High-level Python interface
3. **Examples** (`examples/`): Demonstration code
4. **Tests** (`tests/test_opencog_ml.c`): Comprehensive test suite

### Key Components

- **Engine Management**: Initialize, clone, and manage inference contexts
- **Tensor Operations**: Create, manipulate, and convert tensors
- **Inference Functions**: Single token, sequence, and batch inference
- **State Management**: Handle recurrent state for RWKV models
- **Utility Functions**: Debug, system info, and helper functions

## Getting Started

### Building

The OpenCog ML library is built alongside the main rwkv.cpp project:

```bash
# Configure build
cmake . -DRWKV_STANDALONE=OFF

# Build everything
cmake --build . --config Release

# This creates:
# - libopencog_ml.so (shared library)
# - opencog_ml_example (C example)
# - test_opencog_ml (test suite)
```

### C API Usage

```c
#include "opencog_ml.h"

// Configure the engine
struct opencog_ml_engine_config config = {
    .model_path = "path/to/model.bin",
    .n_threads = 4,
    .n_gpu_layers = 0,
    .enable_state_caching = true,
    .temperature = 1.0f,
    .top_k = 50,
    .top_p = 0.9f
};

// Initialize engine
struct opencog_ml_engine* engine = opencog_ml_init_engine(&config);

// Create initial state
struct opencog_ml_tensor* state = opencog_ml_create_initial_state(engine);

// Perform inference
uint32_t token = 42;
struct opencog_ml_inference_result* result = opencog_ml_infer_token(
    engine, token, state
);

if (result && result->error == OPENCOG_ML_ERROR_NONE) {
    printf("Predicted token: %u\n", result->predicted_token);
    printf("Confidence: %f\n", result->confidence);
}

// Clean up
opencog_ml_free_inference_result(result);
opencog_ml_free_tensor(state);
opencog_ml_free_engine(engine);
```

### Python API Usage

```python
from opencog_ml import OpenCogMLEngine, EngineConfig

# Configure the engine
config = EngineConfig(
    model_path="path/to/model.bin",
    n_threads=4,
    n_gpu_layers=0,
    enable_state_caching=True,
    temperature=1.0,
    top_k=50,
    top_p=0.9
)

# Initialize engine
engine = OpenCogMLEngine(config)

# Create initial state
state = engine.create_initial_state()

# Perform inference
result = engine.infer_token(42, state)

if result.is_successful():
    print(f"Predicted token: {result.predicted_token}")
    print(f"Confidence: {result.confidence}")

# Generate text
tokens, confidences = engine.generate([1, 2, 3], max_length=10)
print(f"Generated tokens: {tokens}")
```

## API Reference

### Core Types

- **`opencog_ml_engine`**: Main inference engine context
- **`opencog_ml_tensor`**: Multi-dimensional tensor representation  
- **`opencog_ml_inference_result`**: Inference output with logits, state, and metadata
- **`opencog_ml_engine_config`**: Engine configuration parameters

### Engine Functions

| Function | Description |
|----------|-------------|
| `opencog_ml_init_engine()` | Initialize engine from configuration |
| `opencog_ml_clone_engine()` | Clone engine for multi-threading |
| `opencog_ml_free_engine()` | Free engine resources |
| `opencog_ml_get_last_error()` | Get last error code |

### Tensor Functions

| Function | Description |
|----------|-------------|
| `opencog_ml_create_tensor()` | Create new tensor with specified shape |
| `opencog_ml_create_tensor_view()` | Create tensor view of existing data |
| `opencog_ml_copy_tensor()` | Copy data between tensors |
| `opencog_ml_free_tensor()` | Free tensor resources |

### Inference Functions

| Function | Description |
|----------|-------------|
| `opencog_ml_infer_token()` | Single token inference |
| `opencog_ml_infer_sequence()` | Sequence inference |
| `opencog_ml_infer_batch()` | Batch inference |
| `opencog_ml_infer_with_sampling()` | Custom sampling parameters |

### State Management

| Function | Description |
|----------|-------------|
| `opencog_ml_create_initial_state()` | Create initial state tensor |
| `opencog_ml_reset_state()` | Reset state to initial values |

### Utility Functions

| Function | Description |
|----------|-------------|
| `opencog_ml_get_vocab_size()` | Get vocabulary size |
| `opencog_ml_get_embed_size()` | Get embedding dimension |
| `opencog_ml_get_state_size()` | Get state tensor size |
| `opencog_ml_set_debug_mode()` | Enable/disable debug logging |

## Integration with OpenCog

The OpenCog ML inference engine is designed to integrate seamlessly with OpenCog's cognitive architecture:

### Symbolic-Subsymbolic Bridge

1. **Concept Activation**: Use RWKV to generate activation patterns for concepts
2. **Attention Allocation**: Guide attention mechanisms using neural predictions
3. **Pattern Recognition**: Enhance pattern matching with neural embeddings
4. **Natural Language Processing**: Process and generate natural language

### Reasoning Integration

1. **Premise Evaluation**: Use neural inference to evaluate premise likelihood
2. **Conclusion Generation**: Generate candidate conclusions using language model
3. **Confidence Estimation**: Provide confidence scores for reasoning steps
4. **Knowledge Grounding**: Ground symbolic knowledge in neural representations

### Example Integration

```python
# Pseudo-code for OpenCog integration
from opencog_ml import OpenCogMLEngine
from opencog import AtomSpace, ConceptNode

# Initialize both systems
atomspace = AtomSpace()
ml_engine = OpenCogMLEngine(config)

# Create concept representations
concept = ConceptNode("intelligence")
concept_embedding = ml_engine.encode_concept(concept)

# Use neural predictions in reasoning
premise = "Artificial intelligence is advancing"
premise_tokens = tokenize(premise)
likelihood = ml_engine.infer_sequence(premise_tokens).confidence

# Generate natural language explanations
explanation_tokens, _ = ml_engine.generate(
    prompt_tokens=encode("Explain: "),
    max_length=50
)
explanation = decode(explanation_tokens)
```

## Performance Considerations

### Memory Usage

- **State Caching**: Enable for repeated inference on similar contexts
- **Batch Processing**: Use batch inference for multiple sequences
- **Engine Cloning**: Clone engines for parallel processing

### Optimization Tips

1. **Use appropriate model size** for your hardware capabilities
2. **Enable GPU layers** if CUDA/Metal support is available
3. **Tune sampling parameters** (temperature, top_k, top_p) for your use case
4. **Cache states** when processing related sequences
5. **Use sequence inference** instead of multiple token inferences when possible

## Error Handling

The OpenCog ML system provides comprehensive error handling:

### Error Types

- `OPENCOG_ML_ERROR_NONE`: No error
- `OPENCOG_ML_ERROR_INVALID_ENGINE`: Invalid engine context
- `OPENCOG_ML_ERROR_INVALID_INPUT`: Invalid input parameters
- `OPENCOG_ML_ERROR_MEMORY_ALLOCATION`: Memory allocation failure
- `OPENCOG_ML_ERROR_INFERENCE_FAILED`: Inference execution failure
- `OPENCOG_ML_ERROR_TENSOR_SHAPE_MISMATCH`: Tensor shape incompatibility
- `OPENCOG_ML_ERROR_UNSUPPORTED_OPERATION`: Unsupported operation

### Best Practices

1. **Always check return values** for NULL pointers
2. **Verify inference results** using the error field
3. **Handle resource cleanup** properly to avoid memory leaks
4. **Use debug mode** during development for detailed logging

## Examples and Testing

### Running Examples

```bash
# C example (requires model file)
./opencog_ml_example path/to/model.bin

# Python example
python examples/opencog_ml_example.py path/to/model.bin
```

### Running Tests

```bash
# Run all tests (some require model file)
./test_opencog_ml [path/to/model.bin]

# Or using CTest
ctest -R opencog_ml_tests
```

## Limitations and Future Work

### Current Limitations

1. **Vocabulary Management**: Token-to-string conversion not implemented
2. **Advanced Sampling**: Only basic sampling algorithms implemented
3. **Model Format**: Currently supports only GGML format RWKV models
4. **GPU Support**: Limited GPU acceleration options

### Future Enhancements

1. **Tokenizer Integration**: Full tokenizer support for text processing
2. **Advanced Sampling**: Implementation of more sophisticated sampling methods
3. **Model Formats**: Support for additional model formats
4. **Streaming Inference**: Support for streaming/incremental inference
5. **Quantization**: Runtime quantization options
6. **Distributed Inference**: Multi-node inference capabilities

## Contributing

When contributing to the OpenCog ML inference engine:

1. **Follow the coding style** established in the existing codebase
2. **Add comprehensive tests** for new functionality
3. **Update documentation** for API changes
4. **Ensure backward compatibility** when possible
5. **Test with multiple model sizes** and configurations

## License

This OpenCog ML implementation follows the same license as the parent rwkv.cpp project.