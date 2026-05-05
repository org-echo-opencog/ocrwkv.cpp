#include "opencog_ml.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

// Internal tensor structure to track ownership
struct opencog_ml_tensor_internal {
    void * data;
    size_t * shape;
    size_t ndim;
    enum opencog_ml_tensor_type type;
    size_t size;
    bool owns_data; // Whether this tensor owns the data pointer
};

// Internal engine structure
struct opencog_ml_engine {
    struct rwkv_context * rwkv_ctx;
    enum opencog_ml_error_flags last_error;
    bool debug_mode;
    
    // Caching for efficiency
    bool state_caching_enabled;
    struct opencog_ml_tensor * cached_state;
    
    // Sampling parameters
    float temperature;
    uint32_t top_k;
    float top_p;
};

// Global debug flag
static bool global_debug_mode = false;

// Helper function to log debug messages
static void debug_log(const char * format, ...) {
    if (global_debug_mode) {
        va_list args;
        va_start(args, format);
        fprintf(stderr, "[OpenCog ML Debug] ");
        vfprintf(stderr, format, args);
        fprintf(stderr, "\n");
        va_end(args);
    }
}

// Helper function to calculate tensor size
static size_t calculate_tensor_size(const size_t * shape, size_t ndim, enum opencog_ml_tensor_type type) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_elements *= shape[i];
    }
    
    size_t element_size;
    switch (type) {
        case OPENCOG_ML_TYPE_F32:
        case OPENCOG_ML_TYPE_I32:
        case OPENCOG_ML_TYPE_U32:
            element_size = 4;
            break;
        case OPENCOG_ML_TYPE_F16:
            element_size = 2;
            break;
        default:
            return 0;
    }
    
    return total_elements * element_size;
}

// Comparison function for qsort (descending probability order)
typedef struct {
    float prob;
    uint32_t index;
} token_prob_t;

static int compare_token_probs_desc(const void * a, const void * b) {
    const token_prob_t * pa = (const token_prob_t *)a;
    const token_prob_t * pb = (const token_prob_t *)b;
    if (pb->prob > pa->prob) return 1;
    if (pb->prob < pa->prob) return -1;
    return 0;
}

// Helper function for temperature/top-k/top-p sampling
static uint32_t sample_token(const float * logits, size_t vocab_size, float temperature, uint32_t top_k, float top_p) {
    // Greedy sampling when temperature is zero
    if (temperature <= 0.0f) {
        uint32_t best_token = 0;
        float best_logit = logits[0];
        for (size_t i = 1; i < vocab_size; i++) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best_token = (uint32_t)i;
            }
        }
        return best_token;
    }

    // Allocate working arrays
    token_prob_t * probs = (token_prob_t *)malloc(vocab_size * sizeof(token_prob_t));
    if (!probs) {
        // Fall back to greedy on allocation failure
        uint32_t best_token = 0;
        float best_logit = logits[0];
        for (size_t i = 1; i < vocab_size; i++) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best_token = (uint32_t)i;
            }
        }
        return best_token;
    }

    // Find max logit for numerical stability, then compute softmax
    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i].prob = expf(logits[i] - max_logit);
        probs[i].index = (uint32_t)i;
        sum_exp += probs[i].prob;
    }
    for (size_t i = 0; i < vocab_size; i++) {
        probs[i].prob /= sum_exp;
    }

    // Apply top-k: zero out all but the top-k tokens
    size_t k = (top_k > 0 && (size_t)top_k < vocab_size) ? (size_t)top_k : vocab_size;
    if (k < vocab_size) {
        qsort(probs, vocab_size, sizeof(token_prob_t), compare_token_probs_desc);
        for (size_t i = k; i < vocab_size; i++) {
            probs[i].prob = 0.0f;
        }
    }

    // Apply top-p (nucleus sampling): zero out tokens below cumulative threshold
    if (top_p > 0.0f && top_p < 1.0f) {
        if (k == vocab_size) {
            // Need sorted order for top-p if top-k didn't sort already
            qsort(probs, vocab_size, sizeof(token_prob_t), compare_token_probs_desc);
        }
        float cumulative = 0.0f;
        size_t cutoff = vocab_size;
        for (size_t i = 0; i < vocab_size; i++) {
            cumulative += probs[i].prob;
            if (cumulative > top_p) {
                cutoff = i + 1;
                break;
            }
        }
        for (size_t i = cutoff; i < vocab_size; i++) {
            probs[i].prob = 0.0f;
        }
    }

    // Apply temperature scaling and renormalize
    sum_exp = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        if (probs[i].prob > 0.0f) {
            probs[i].prob = powf(probs[i].prob, 1.0f / temperature);
            sum_exp += probs[i].prob;
        }
    }
    if (sum_exp > 0.0f) {
        for (size_t i = 0; i < vocab_size; i++) {
            probs[i].prob /= sum_exp;
        }
    }

    // Multinomial sampling: draw a random value in [0, 1)
    // Seed once on first call using a mix of time and address-space entropy
    static int seeded = 0;
    if (!seeded) {
        // XOR time with a stack-pointer value for sub-second and cross-process variation
        unsigned int seed = (unsigned int)time(NULL);
        seed ^= (unsigned int)(uintptr_t)probs;
        srand(seed);
        seeded = 1;
    }
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    float cumulative = 0.0f;
    uint32_t sampled = probs[0].index;
    for (size_t i = 0; i < vocab_size; i++) {
        cumulative += probs[i].prob;
        if (r < cumulative) {
            sampled = probs[i].index;
            break;
        }
    }

    free(probs);
    return sampled;
}

// === Core Engine Functions ===

struct opencog_ml_engine * opencog_ml_init_engine(const struct opencog_ml_engine_config * config) {
    if (!config || !config->model_path) {
        debug_log("Invalid engine configuration");
        return NULL;
    }
    
    debug_log("Initializing OpenCog ML engine with model: %s", config->model_path);
    
    // Allocate engine structure
    struct opencog_ml_engine * engine = (struct opencog_ml_engine *)calloc(1, sizeof(struct opencog_ml_engine));
    if (!engine) {
        debug_log("Failed to allocate memory for engine");
        return NULL;
    }
    
    // Initialize RWKV context
    engine->rwkv_ctx = rwkv_init_from_file(config->model_path, config->n_threads, config->n_gpu_layers);
    if (!engine->rwkv_ctx) {
        debug_log("Failed to initialize RWKV context");
        free(engine);
        return NULL;
    }
    
    // Set configuration parameters
    engine->state_caching_enabled = config->enable_state_caching;
    engine->temperature = config->temperature;
    engine->top_k = config->top_k;
    engine->top_p = config->top_p;
    engine->debug_mode = global_debug_mode;
    engine->last_error = OPENCOG_ML_ERROR_NONE;
    
    debug_log("OpenCog ML engine initialized successfully");
    return engine;
}

struct opencog_ml_engine * opencog_ml_clone_engine(struct opencog_ml_engine * engine, uint32_t n_threads) {
    if (!engine) {
        debug_log("Cannot clone NULL engine");
        return NULL;
    }
    
    debug_log("Cloning OpenCog ML engine");
    
    // Allocate new engine structure
    struct opencog_ml_engine * cloned_engine = (struct opencog_ml_engine *)calloc(1, sizeof(struct opencog_ml_engine));
    if (!cloned_engine) {
        debug_log("Failed to allocate memory for cloned engine");
        return NULL;
    }
    
    // Clone RWKV context
    cloned_engine->rwkv_ctx = rwkv_clone_context(engine->rwkv_ctx, n_threads);
    if (!cloned_engine->rwkv_ctx) {
        debug_log("Failed to clone RWKV context");
        free(cloned_engine);
        return NULL;
    }
    
    // Copy configuration
    cloned_engine->state_caching_enabled = engine->state_caching_enabled;
    cloned_engine->temperature = engine->temperature;
    cloned_engine->top_k = engine->top_k;
    cloned_engine->top_p = engine->top_p;
    cloned_engine->debug_mode = engine->debug_mode;
    cloned_engine->last_error = OPENCOG_ML_ERROR_NONE;
    
    debug_log("Engine cloned successfully");
    return cloned_engine;
}

void opencog_ml_free_engine(struct opencog_ml_engine * engine) {
    if (!engine) {
        return;
    }
    
    debug_log("Freeing OpenCog ML engine");
    
    if (engine->rwkv_ctx) {
        rwkv_free(engine->rwkv_ctx);
    }
    
    if (engine->cached_state) {
        opencog_ml_free_tensor(engine->cached_state);
    }
    
    free(engine);
}

enum opencog_ml_error_flags opencog_ml_get_last_error(struct opencog_ml_engine * engine) {
    if (!engine) {
        return OPENCOG_ML_ERROR_INVALID_ENGINE;
    }
    
    enum opencog_ml_error_flags error = engine->last_error;
    engine->last_error = OPENCOG_ML_ERROR_NONE; // Reset after reading
    return error;
}

// === Tensor Operations ===

struct opencog_ml_tensor * opencog_ml_create_tensor(const size_t * shape, size_t ndim, enum opencog_ml_tensor_type type) {
    if (!shape || ndim == 0) {
        debug_log("Invalid tensor shape or dimensions");
        return NULL;
    }
    
    struct opencog_ml_tensor * tensor = (struct opencog_ml_tensor *)calloc(1, sizeof(struct opencog_ml_tensor));
    if (!tensor) {
        debug_log("Failed to allocate memory for tensor structure");
        return NULL;
    }
    
    // Copy shape
    tensor->shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (!tensor->shape) {
        debug_log("Failed to allocate memory for tensor shape");
        free(tensor);
        return NULL;
    }
    
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    tensor->ndim = ndim;
    tensor->type = type;
    
    // Calculate size and allocate data
    size_t total_size = calculate_tensor_size(shape, ndim, type);
    if (total_size == 0) {
        debug_log("Invalid tensor size calculation");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    
    tensor->data = malloc(total_size);
    if (!tensor->data) {
        debug_log("Failed to allocate memory for tensor data");
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    
    // Calculate number of elements
    tensor->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }
    
    tensor->owns_data = true; // We allocated the data, so we own it
    
    debug_log("Created tensor with %zu elements", tensor->size);
    return tensor;
}

struct opencog_ml_tensor * opencog_ml_create_tensor_view(void * data, const size_t * shape, size_t ndim, enum opencog_ml_tensor_type type) {
    if (!data || !shape || ndim == 0) {
        debug_log("Invalid parameters for tensor view");
        return NULL;
    }
    
    struct opencog_ml_tensor * tensor = (struct opencog_ml_tensor *)calloc(1, sizeof(struct opencog_ml_tensor));
    if (!tensor) {
        debug_log("Failed to allocate memory for tensor view structure");
        return NULL;
    }
    
    // Copy shape
    tensor->shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (!tensor->shape) {
        debug_log("Failed to allocate memory for tensor view shape");
        free(tensor);
        return NULL;
    }
    
    memcpy(tensor->shape, shape, ndim * sizeof(size_t));
    tensor->ndim = ndim;
    tensor->type = type;
    tensor->data = data; // Note: This is a view, we don't own the data
    
    // Calculate number of elements
    tensor->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }
    
    tensor->owns_data = false; // This is a view, we don't own the data
    
    debug_log("Created tensor view with %zu elements", tensor->size);
    return tensor;
}

bool opencog_ml_copy_tensor(const struct opencog_ml_tensor * src, struct opencog_ml_tensor * dst) {
    if (!src || !dst) {
        debug_log("Invalid tensor pointers for copy operation");
        return false;
    }
    
    if (src->size != dst->size || src->type != dst->type) {
        debug_log("Tensor size or type mismatch for copy operation");
        return false;
    }
    
    size_t data_size = calculate_tensor_size(src->shape, src->ndim, src->type);
    memcpy(dst->data, src->data, data_size);
    
    debug_log("Copied tensor data (%zu bytes)", data_size);
    return true;
}

void opencog_ml_free_tensor(struct opencog_ml_tensor * tensor) {
    if (!tensor) {
        return;
    }
    
    if (tensor->shape) {
        free(tensor->shape);
    }
    
    // Only free data if we own it
    if (tensor->data && tensor->owns_data) {
        free(tensor->data);
    }
    
    free(tensor);
}

// === Inference Functions ===

struct opencog_ml_inference_result * opencog_ml_infer_token(struct opencog_ml_engine * engine, uint32_t token, const struct opencog_ml_tensor * state_in) {
    if (!engine) {
        debug_log("Invalid engine for token inference");
        return NULL;
    }
    
    debug_log("Performing token inference for token: %u", token);
    
    // Allocate result structure
    struct opencog_ml_inference_result * result = (struct opencog_ml_inference_result *)calloc(1, sizeof(struct opencog_ml_inference_result));
    if (!result) {
        debug_log("Failed to allocate memory for inference result");
        engine->last_error = OPENCOG_ML_ERROR_MEMORY_ALLOCATION;
        return NULL;
    }
    
    // Get model dimensions
    size_t state_len = rwkv_get_state_len(engine->rwkv_ctx);
    size_t logits_len = rwkv_get_logits_len(engine->rwkv_ctx);
    
    // Allocate buffers
    float * state_buffer = (float *)malloc(state_len * sizeof(float));
    float * logits_buffer = (float *)malloc(logits_len * sizeof(float));
    
    if (!state_buffer || !logits_buffer) {
        debug_log("Failed to allocate buffers for inference");
        free(state_buffer);
        free(logits_buffer);
        free(result);
        engine->last_error = OPENCOG_ML_ERROR_MEMORY_ALLOCATION;
        return NULL;
    }
    
    // Prepare input state
    const float * input_state = NULL;
    if (state_in && state_in->data && state_in->type == OPENCOG_ML_TYPE_F32) {
        input_state = (const float *)state_in->data;
    }
    
    // Perform RWKV inference
    bool success = rwkv_eval(engine->rwkv_ctx, token, input_state, state_buffer, logits_buffer);
    if (!success) {
        debug_log("RWKV evaluation failed");
        free(state_buffer);
        free(logits_buffer);
        free(result);
        engine->last_error = OPENCOG_ML_ERROR_INFERENCE_FAILED;
        return NULL;
    }
    
    // Create output tensors
    size_t logits_shape[] = { logits_len };
    size_t state_shape[] = { state_len };
    
    result->logits = opencog_ml_create_tensor_view(logits_buffer, logits_shape, 1, OPENCOG_ML_TYPE_F32);
    result->state = opencog_ml_create_tensor_view(state_buffer, state_shape, 1, OPENCOG_ML_TYPE_F32);
    
    if (!result->logits || !result->state) {
        debug_log("Failed to create output tensors");
        // Tensors are views (owns_data=false), so free the raw buffers directly
        opencog_ml_free_tensor(result->logits);
        opencog_ml_free_tensor(result->state);
        free(logits_buffer);
        free(state_buffer);
        free(result);
        engine->last_error = OPENCOG_ML_ERROR_MEMORY_ALLOCATION;
        return NULL;
    }
    
    // Mark these tensors as owning their data since we allocated the buffers
    result->logits->owns_data = true;
    result->state->owns_data = true;
    
    // Sample predicted token
    result->predicted_token = sample_token(logits_buffer, logits_len, engine->temperature, engine->top_k, engine->top_p);
    
    // Calculate confidence (simplified)
    float max_logit = logits_buffer[result->predicted_token];
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits_len; i++) {
        sum_exp += expf(logits_buffer[i] - max_logit);
    }
    result->confidence = 1.0f / sum_exp; // Simplified confidence calculation
    
    result->error = OPENCOG_ML_ERROR_NONE;
    
    debug_log("Token inference completed successfully. Predicted token: %u, confidence: %f", 
              result->predicted_token, result->confidence);
    
    return result;
}

struct opencog_ml_inference_result * opencog_ml_infer_sequence(struct opencog_ml_engine * engine, const uint32_t * tokens, size_t sequence_length, const struct opencog_ml_tensor * state_in) {
    if (!engine || !tokens || sequence_length == 0) {
        debug_log("Invalid parameters for sequence inference");
        if (engine) {
            engine->last_error = OPENCOG_ML_ERROR_INVALID_INPUT;
        }
        return NULL;
    }
    
    debug_log("Performing sequence inference for %zu tokens", sequence_length);
    
    // Allocate result structure
    struct opencog_ml_inference_result * result = (struct opencog_ml_inference_result *)calloc(1, sizeof(struct opencog_ml_inference_result));
    if (!result) {
        debug_log("Failed to allocate memory for sequence inference result");
        engine->last_error = OPENCOG_ML_ERROR_MEMORY_ALLOCATION;
        return NULL;
    }
    
    // Get model dimensions
    size_t state_len = rwkv_get_state_len(engine->rwkv_ctx);
    size_t logits_len = rwkv_get_logits_len(engine->rwkv_ctx);
    
    // Allocate buffers
    float * state_buffer = (float *)malloc(state_len * sizeof(float));
    float * logits_buffer = (float *)malloc(logits_len * sizeof(float));
    
    if (!state_buffer || !logits_buffer) {
        debug_log("Failed to allocate buffers for sequence inference");
        free(state_buffer);
        free(logits_buffer);
        free(result);
        engine->last_error = OPENCOG_ML_ERROR_MEMORY_ALLOCATION;
        return NULL;
    }
    
    // Prepare input state
    const float * input_state = NULL;
    if (state_in && state_in->data && state_in->type == OPENCOG_ML_TYPE_F32) {
        input_state = (const float *)state_in->data;
    }
    
    // Perform RWKV sequence evaluation
    bool success = rwkv_eval_sequence(engine->rwkv_ctx, tokens, sequence_length, input_state, state_buffer, logits_buffer);
    if (!success) {
        debug_log("RWKV sequence evaluation failed");
        free(state_buffer);
        free(logits_buffer);
        free(result);
        engine->last_error = OPENCOG_ML_ERROR_INFERENCE_FAILED;
        return NULL;
    }
    
    // Create output tensors
    size_t logits_shape[] = { logits_len };
    size_t state_shape[] = { state_len };
    
    result->logits = opencog_ml_create_tensor_view(logits_buffer, logits_shape, 1, OPENCOG_ML_TYPE_F32);
    result->state = opencog_ml_create_tensor_view(state_buffer, state_shape, 1, OPENCOG_ML_TYPE_F32);
    
    if (!result->logits || !result->state) {
        debug_log("Failed to create output tensors for sequence");
        // Tensors are views (owns_data=false), so free the raw buffers directly
        opencog_ml_free_tensor(result->logits);
        opencog_ml_free_tensor(result->state);
        free(logits_buffer);
        free(state_buffer);
        free(result);
        engine->last_error = OPENCOG_ML_ERROR_MEMORY_ALLOCATION;
        return NULL;
    }
    
    // Mark these tensors as owning their data since we allocated the buffers
    result->logits->owns_data = true;
    result->state->owns_data = true;
    
    // Sample predicted token from final logits
    result->predicted_token = sample_token(logits_buffer, logits_len, engine->temperature, engine->top_k, engine->top_p);
    
    // Calculate confidence
    float max_logit = logits_buffer[result->predicted_token];
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits_len; i++) {
        sum_exp += expf(logits_buffer[i] - max_logit);
    }
    result->confidence = 1.0f / sum_exp;
    
    result->error = OPENCOG_ML_ERROR_NONE;
    
    debug_log("Sequence inference completed successfully. Predicted token: %u, confidence: %f", 
              result->predicted_token, result->confidence);
    
    return result;
}

struct opencog_ml_inference_result ** opencog_ml_infer_batch(struct opencog_ml_engine * engine, const uint32_t ** token_sequences, const size_t * sequence_lengths, size_t batch_size, const struct opencog_ml_tensor ** states_in) {
    if (!engine || !token_sequences || !sequence_lengths || batch_size == 0) {
        debug_log("Invalid parameters for batch inference");
        if (engine) {
            engine->last_error = OPENCOG_ML_ERROR_INVALID_INPUT;
        }
        return NULL;
    }
    
    debug_log("Performing batch inference for %zu sequences", batch_size);
    
    // Allocate array of result pointers
    struct opencog_ml_inference_result ** results = (struct opencog_ml_inference_result **)calloc(batch_size, sizeof(struct opencog_ml_inference_result *));
    if (!results) {
        debug_log("Failed to allocate memory for batch results");
        engine->last_error = OPENCOG_ML_ERROR_MEMORY_ALLOCATION;
        return NULL;
    }
    
    // Process each sequence in the batch
    for (size_t i = 0; i < batch_size; i++) {
        const struct opencog_ml_tensor * state_in = (states_in && states_in[i]) ? states_in[i] : NULL;
        results[i] = opencog_ml_infer_sequence(engine, token_sequences[i], sequence_lengths[i], state_in);
        
        if (!results[i]) {
            debug_log("Failed to process sequence %zu in batch", i);
            // Clean up allocated results
            for (size_t j = 0; j < i; j++) {
                opencog_ml_free_inference_result(results[j]);
            }
            free(results);
            return NULL;
        }
    }
    
    debug_log("Batch inference completed successfully for %zu sequences", batch_size);
    return results;
}

struct opencog_ml_inference_result * opencog_ml_infer_with_sampling(struct opencog_ml_engine * engine, uint32_t token, const struct opencog_ml_tensor * state_in, float temperature, uint32_t top_k, float top_p) {
    if (!engine) {
        debug_log("Invalid engine for custom sampling inference");
        return NULL;
    }
    
    // Temporarily set custom sampling parameters
    float orig_temp = engine->temperature;
    uint32_t orig_top_k = engine->top_k;
    float orig_top_p = engine->top_p;
    
    engine->temperature = temperature;
    engine->top_k = top_k;
    engine->top_p = top_p;
    
    // Perform inference
    struct opencog_ml_inference_result * result = opencog_ml_infer_token(engine, token, state_in);
    
    // Restore original parameters
    engine->temperature = orig_temp;
    engine->top_k = orig_top_k;
    engine->top_p = orig_top_p;
    
    return result;
}

void opencog_ml_free_inference_result(struct opencog_ml_inference_result * result) {
    if (!result) {
        return;
    }
    
    // Free tensors - they will handle their own data cleanup based on ownership
    opencog_ml_free_tensor(result->logits);
    opencog_ml_free_tensor(result->state);
    free(result);
}

// === State Management ===

struct opencog_ml_tensor * opencog_ml_create_initial_state(struct opencog_ml_engine * engine) {
    if (!engine) {
        debug_log("Invalid engine for initial state creation");
        return NULL;
    }
    
    size_t state_len = rwkv_get_state_len(engine->rwkv_ctx);
    size_t shape[] = { state_len };
    
    struct opencog_ml_tensor * state = opencog_ml_create_tensor(shape, 1, OPENCOG_ML_TYPE_F32);
    if (!state) {
        debug_log("Failed to create initial state tensor");
        return NULL;
    }
    
    // Initialize state using RWKV
    rwkv_init_state(engine->rwkv_ctx, (float *)state->data);
    
    debug_log("Created initial state with %zu elements", state_len);
    return state;
}

bool opencog_ml_reset_state(struct opencog_ml_engine * engine, struct opencog_ml_tensor * state) {
    if (!engine || !state || !state->data || state->type != OPENCOG_ML_TYPE_F32) {
        debug_log("Invalid parameters for state reset");
        return false;
    }
    
    rwkv_init_state(engine->rwkv_ctx, (float *)state->data);
    debug_log("State reset successfully");
    return true;
}

// === Utility Functions ===

size_t opencog_ml_get_vocab_size(struct opencog_ml_engine * engine) {
    if (!engine) {
        return 0;
    }
    return rwkv_get_n_vocab(engine->rwkv_ctx);
}

size_t opencog_ml_get_embed_size(struct opencog_ml_engine * engine) {
    if (!engine) {
        return 0;
    }
    return rwkv_get_n_embed(engine->rwkv_ctx);
}

size_t opencog_ml_get_state_size(struct opencog_ml_engine * engine) {
    if (!engine) {
        return 0;
    }
    return rwkv_get_state_len(engine->rwkv_ctx);
}

const char * opencog_ml_token_to_string(struct opencog_ml_engine * engine, uint32_t token) {
    // TODO: Implement token-to-string conversion if vocabulary is available
    // For now, return NULL to indicate functionality is not available
    (void)engine;
    (void)token;
    return NULL;
}

uint32_t opencog_ml_string_to_token(struct opencog_ml_engine * engine, const char * str) {
    // TODO: Implement string-to-token conversion if vocabulary is available
    // For now, return 0 as a fallback
    (void)engine;
    (void)str;
    return 0;
}

const char * opencog_ml_get_system_info(void) {
    return rwkv_get_system_info_string();
}

void opencog_ml_set_debug_mode(bool enable) {
    global_debug_mode = enable;
    debug_log("Debug mode %s", enable ? "enabled" : "disabled");
}