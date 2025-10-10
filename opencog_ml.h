#ifndef OPENCOG_ML_H
#define OPENCOG_ML_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "rwkv.h"

#if defined(OPENCOG_ML_SHARED)
#    if defined(_WIN32) && !defined(__MINGW32__)
#        if defined(OPENCOG_ML_BUILD)
#            define OPENCOG_ML_API __declspec(dllexport)
#        else
#            define OPENCOG_ML_API __declspec(dllimport)
#        endif
#    else
#        define OPENCOG_ML_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define OPENCOG_ML_API
#endif

#if defined(__cplusplus)
extern "C" {
#endif

    // Forward declarations
    struct opencog_ml_engine;
    struct opencog_ml_tensor;
    struct opencog_ml_inference_result;

    // Error handling
    enum opencog_ml_error_flags {
        OPENCOG_ML_ERROR_NONE = 0,
        OPENCOG_ML_ERROR_INVALID_ENGINE = 1,
        OPENCOG_ML_ERROR_INVALID_INPUT = 2,
        OPENCOG_ML_ERROR_MEMORY_ALLOCATION = 3,
        OPENCOG_ML_ERROR_INFERENCE_FAILED = 4,
        OPENCOG_ML_ERROR_TENSOR_SHAPE_MISMATCH = 5,
        OPENCOG_ML_ERROR_UNSUPPORTED_OPERATION = 6
    };

    // Tensor data types
    enum opencog_ml_tensor_type {
        OPENCOG_ML_TYPE_F32 = 0,
        OPENCOG_ML_TYPE_F16 = 1,
        OPENCOG_ML_TYPE_I32 = 2,
        OPENCOG_ML_TYPE_U32 = 3
    };

    // Inference execution modes
    enum opencog_ml_inference_mode {
        OPENCOG_ML_MODE_SINGLE_TOKEN = 0,
        OPENCOG_ML_MODE_SEQUENCE = 1,
        OPENCOG_ML_MODE_BATCH = 2
    };

    // Tensor representation for OpenCog ML
    struct opencog_ml_tensor {
        void * data;                           // Pointer to tensor data
        size_t * shape;                        // Tensor dimensions
        size_t ndim;                          // Number of dimensions
        enum opencog_ml_tensor_type type;     // Data type
        size_t size;                          // Total number of elements
        bool owns_data;                       // Whether this tensor owns the data
    };

    // Inference result containing logits and state
    struct opencog_ml_inference_result {
        struct opencog_ml_tensor * logits;    // Output logits tensor
        struct opencog_ml_tensor * state;     // Output state tensor
        float confidence;                     // Confidence score (0.0-1.0)
        uint32_t predicted_token;            // Most likely next token
        enum opencog_ml_error_flags error;   // Error status
    };

    // Configuration for the inference engine
    struct opencog_ml_engine_config {
        const char * model_path;              // Path to RWKV model file
        uint32_t n_threads;                   // Number of threads
        uint32_t n_gpu_layers;               // Number of GPU layers
        bool enable_state_caching;           // Enable state caching for efficiency
        float temperature;                   // Sampling temperature
        uint32_t top_k;                      // Top-K sampling parameter
        float top_p;                         // Top-P sampling parameter
    };

    // === Core Engine Functions ===

    // Initialize OpenCog ML inference engine
    OPENCOG_ML_API struct opencog_ml_engine * opencog_ml_init_engine(
        const struct opencog_ml_engine_config * config
    );

    // Clone an existing engine for multi-threaded inference
    OPENCOG_ML_API struct opencog_ml_engine * opencog_ml_clone_engine(
        struct opencog_ml_engine * engine,
        uint32_t n_threads
    );

    // Free the inference engine and all associated resources
    OPENCOG_ML_API void opencog_ml_free_engine(struct opencog_ml_engine * engine);

    // Get the last error from the engine
    OPENCOG_ML_API enum opencog_ml_error_flags opencog_ml_get_last_error(
        struct opencog_ml_engine * engine
    );

    // === Tensor Operations ===

    // Create a new tensor with specified shape and type
    OPENCOG_ML_API struct opencog_ml_tensor * opencog_ml_create_tensor(
        const size_t * shape,
        size_t ndim,
        enum opencog_ml_tensor_type type
    );

    // Create tensor from existing data (no copy)
    OPENCOG_ML_API struct opencog_ml_tensor * opencog_ml_create_tensor_view(
        void * data,
        const size_t * shape,
        size_t ndim,
        enum opencog_ml_tensor_type type
    );

    // Copy data from one tensor to another
    OPENCOG_ML_API bool opencog_ml_copy_tensor(
        const struct opencog_ml_tensor * src,
        struct opencog_ml_tensor * dst
    );

    // Free tensor and its data
    OPENCOG_ML_API void opencog_ml_free_tensor(struct opencog_ml_tensor * tensor);

    // === Inference Functions ===

    // Pure inference - single token prediction
    OPENCOG_ML_API struct opencog_ml_inference_result * opencog_ml_infer_token(
        struct opencog_ml_engine * engine,
        uint32_t token,
        const struct opencog_ml_tensor * state_in
    );

    // Pure inference - sequence processing
    OPENCOG_ML_API struct opencog_ml_inference_result * opencog_ml_infer_sequence(
        struct opencog_ml_engine * engine,
        const uint32_t * tokens,
        size_t sequence_length,
        const struct opencog_ml_tensor * state_in
    );

    // Pure inference - batch processing (multiple sequences)
    OPENCOG_ML_API struct opencog_ml_inference_result ** opencog_ml_infer_batch(
        struct opencog_ml_engine * engine,
        const uint32_t ** token_sequences,
        const size_t * sequence_lengths,
        size_t batch_size,
        const struct opencog_ml_tensor ** states_in
    );

    // Advanced inference with custom sampling
    OPENCOG_ML_API struct opencog_ml_inference_result * opencog_ml_infer_with_sampling(
        struct opencog_ml_engine * engine,
        uint32_t token,
        const struct opencog_ml_tensor * state_in,
        float temperature,
        uint32_t top_k,
        float top_p
    );

    // Free inference result
    OPENCOG_ML_API void opencog_ml_free_inference_result(
        struct opencog_ml_inference_result * result
    );

    // === State Management ===

    // Initialize a new state tensor
    OPENCOG_ML_API struct opencog_ml_tensor * opencog_ml_create_initial_state(
        struct opencog_ml_engine * engine
    );

    // Reset state to initial values
    OPENCOG_ML_API bool opencog_ml_reset_state(
        struct opencog_ml_engine * engine,
        struct opencog_ml_tensor * state
    );

    // === Utility Functions ===

    // Get model vocabulary size
    OPENCOG_ML_API size_t opencog_ml_get_vocab_size(
        struct opencog_ml_engine * engine
    );

    // Get model embedding dimension
    OPENCOG_ML_API size_t opencog_ml_get_embed_size(
        struct opencog_ml_engine * engine
    );

    // Get model state tensor size
    OPENCOG_ML_API size_t opencog_ml_get_state_size(
        struct opencog_ml_engine * engine
    );

    // Convert token to string (if vocabulary is available)
    OPENCOG_ML_API const char * opencog_ml_token_to_string(
        struct opencog_ml_engine * engine,
        uint32_t token
    );

    // Convert string to token (if vocabulary is available)
    OPENCOG_ML_API uint32_t opencog_ml_string_to_token(
        struct opencog_ml_engine * engine,
        const char * str
    );

    // Get system information
    OPENCOG_ML_API const char * opencog_ml_get_system_info(void);

    // Enable/disable debug logging
    OPENCOG_ML_API void opencog_ml_set_debug_mode(bool enable);

#if defined(__cplusplus)
}
#endif

#endif // OPENCOG_ML_H