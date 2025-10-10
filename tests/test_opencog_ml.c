/**
 * Unit tests for OpenCog ML Inference Engine
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "../opencog_ml.h"

// Test utilities
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("FAIL: %s\n", message); \
            return 0; \
        } \
    } while(0)

#define TEST_PASS(message) \
    do { \
        printf("PASS: %s\n", message); \
    } while(0)

// Mock model path for testing (will be set by test runner if available)
static const char* test_model_path = NULL;

int test_tensor_operations() {
    printf("\n=== Testing Tensor Operations ===\n");
    
    // Test tensor creation
    size_t shape[] = {5, 3};
    struct opencog_ml_tensor* tensor = opencog_ml_create_tensor(
        shape, 2, OPENCOG_ML_TYPE_F32
    );
    
    TEST_ASSERT(tensor != NULL, "Tensor creation");
    TEST_ASSERT(tensor->ndim == 2, "Tensor dimensions");
    TEST_ASSERT(tensor->shape[0] == 5 && tensor->shape[1] == 3, "Tensor shape");
    TEST_ASSERT(tensor->size == 15, "Tensor size calculation");
    TEST_ASSERT(tensor->type == OPENCOG_ML_TYPE_F32, "Tensor type");
    
    // Test tensor view creation
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t view_shape[] = {2, 2};
    struct opencog_ml_tensor* view = opencog_ml_create_tensor_view(
        data, view_shape, 2, OPENCOG_ML_TYPE_F32
    );
    
    TEST_ASSERT(view != NULL, "Tensor view creation");
    TEST_ASSERT(view->data == data, "Tensor view data pointer");
    TEST_ASSERT(view->size == 4, "Tensor view size");
    
    // Test tensor copy
    struct opencog_ml_tensor* copy_tensor = opencog_ml_create_tensor(
        view_shape, 2, OPENCOG_ML_TYPE_F32
    );
    bool copy_success = opencog_ml_copy_tensor(view, copy_tensor);
    TEST_ASSERT(copy_success, "Tensor copy operation");
    
    // Verify copy data
    float* original_data = (float*)view->data;
    float* copied_data = (float*)copy_tensor->data;
    bool data_matches = true;
    for (size_t i = 0; i < view->size; i++) {
        if (fabsf(original_data[i] - copied_data[i]) > 1e-6f) {
            data_matches = false;
            break;
        }
    }
    TEST_ASSERT(data_matches, "Tensor copy data integrity");
    
    // Clean up
    opencog_ml_free_tensor(tensor);
    opencog_ml_free_tensor(view);
    opencog_ml_free_tensor(copy_tensor);
    
    TEST_PASS("All tensor operations");
    return 1;
}

int test_engine_management() {
    printf("\n=== Testing Engine Management ===\n");
    
    if (!test_model_path) {
        printf("SKIP: Engine management tests (no model provided)\n");
        return 1;
    }
    
    // Test engine configuration
    struct opencog_ml_engine_config config = {
        .model_path = test_model_path,
        .n_threads = 1,
        .n_gpu_layers = 0,
        .enable_state_caching = true,
        .temperature = 1.0f,
        .top_k = 50,
        .top_p = 0.9f
    };
    
    // Test engine initialization
    struct opencog_ml_engine* engine = opencog_ml_init_engine(&config);
    TEST_ASSERT(engine != NULL, "Engine initialization");
    
    // Test model properties
    size_t vocab_size = opencog_ml_get_vocab_size(engine);
    size_t embed_size = opencog_ml_get_embed_size(engine);
    size_t state_size = opencog_ml_get_state_size(engine);
    
    TEST_ASSERT(vocab_size > 0, "Vocabulary size retrieval");
    TEST_ASSERT(embed_size > 0, "Embedding size retrieval");
    TEST_ASSERT(state_size > 0, "State size retrieval");
    
    printf("Model info: vocab=%zu, embed=%zu, state=%zu\n", 
           vocab_size, embed_size, state_size);
    
    // Test engine cloning
    struct opencog_ml_engine* cloned_engine = opencog_ml_clone_engine(engine, 1);
    TEST_ASSERT(cloned_engine != NULL, "Engine cloning");
    
    // Verify cloned engine properties match
    TEST_ASSERT(opencog_ml_get_vocab_size(cloned_engine) == vocab_size, "Cloned engine vocab size");
    TEST_ASSERT(opencog_ml_get_embed_size(cloned_engine) == embed_size, "Cloned engine embed size");
    TEST_ASSERT(opencog_ml_get_state_size(cloned_engine) == state_size, "Cloned engine state size");
    
    // Test error handling
    enum opencog_ml_error_flags error = opencog_ml_get_last_error(engine);
    TEST_ASSERT(error == OPENCOG_ML_ERROR_NONE, "Initial error state");
    
    // Clean up
    opencog_ml_free_engine(cloned_engine);
    opencog_ml_free_engine(engine);
    
    TEST_PASS("All engine management operations");
    return 1;
}

int test_state_management() {
    printf("\n=== Testing State Management ===\n");
    
    if (!test_model_path) {
        printf("SKIP: State management tests (no model provided)\n");
        return 1;
    }
    
    struct opencog_ml_engine_config config = {
        .model_path = test_model_path,
        .n_threads = 1,
        .n_gpu_layers = 0,
        .enable_state_caching = true,
        .temperature = 1.0f,
        .top_k = 50,
        .top_p = 0.9f
    };
    
    struct opencog_ml_engine* engine = opencog_ml_init_engine(&config);
    TEST_ASSERT(engine != NULL, "Engine initialization for state tests");
    
    // Test initial state creation
    struct opencog_ml_tensor* state = opencog_ml_create_initial_state(engine);
    TEST_ASSERT(state != NULL, "Initial state creation");
    TEST_ASSERT(state->type == OPENCOG_ML_TYPE_F32, "State tensor type");
    
    size_t expected_state_size = opencog_ml_get_state_size(engine);
    TEST_ASSERT(state->size == expected_state_size, "State tensor size");
    
    // Test state reset
    bool reset_success = opencog_ml_reset_state(engine, state);
    TEST_ASSERT(reset_success, "State reset operation");
    
    // Verify state contains valid data (not all zeros or NaN)
    float* state_data = (float*)state->data;
    bool has_valid_data = false;
    bool has_invalid_data = false;
    
    for (size_t i = 0; i < state->size && i < 100; i++) { // Check first 100 elements
        if (!isnan(state_data[i]) && !isinf(state_data[i])) {
            if (fabsf(state_data[i]) > 1e-10f) {
                has_valid_data = true;
            }
        } else {
            has_invalid_data = true;
        }
    }
    
    TEST_ASSERT(!has_invalid_data, "State data validity (no NaN/Inf)");
    // Note: Initial state might be all zeros, so we don't require non-zero values
    
    // Clean up
    opencog_ml_free_tensor(state);
    opencog_ml_free_engine(engine);
    
    TEST_PASS("All state management operations");
    return 1;
}

int test_basic_inference() {
    printf("\n=== Testing Basic Inference ===\n");
    
    if (!test_model_path) {
        printf("SKIP: Basic inference tests (no model provided)\n");
        return 1;
    }
    
    struct opencog_ml_engine_config config = {
        .model_path = test_model_path,
        .n_threads = 1,
        .n_gpu_layers = 0,
        .enable_state_caching = true,
        .temperature = 1.0f,
        .top_k = 50,
        .top_p = 0.9f
    };
    
    struct opencog_ml_engine* engine = opencog_ml_init_engine(&config);
    TEST_ASSERT(engine != NULL, "Engine initialization for inference tests");
    
    struct opencog_ml_tensor* state = opencog_ml_create_initial_state(engine);
    TEST_ASSERT(state != NULL, "Initial state for inference tests");
    
    // Test single token inference
    uint32_t test_token = 1; // Use token 1 which should be valid for most models
    struct opencog_ml_inference_result* result = opencog_ml_infer_token(
        engine, test_token, state
    );
    
    TEST_ASSERT(result != NULL, "Token inference result");
    TEST_ASSERT(result->error == OPENCOG_ML_ERROR_NONE, "Token inference success");
    TEST_ASSERT(result->logits != NULL, "Token inference logits");
    TEST_ASSERT(result->state != NULL, "Token inference output state");
    
    // Verify result properties
    size_t vocab_size = opencog_ml_get_vocab_size(engine);
    TEST_ASSERT(result->logits->size == vocab_size, "Logits size matches vocab");
    TEST_ASSERT(result->predicted_token < vocab_size, "Predicted token in valid range");
    TEST_ASSERT(result->confidence >= 0.0f && result->confidence <= 1.0f, "Confidence in valid range");
    
    // Verify logits are finite
    float* logits_data = (float*)result->logits->data;
    bool logits_valid = true;
    for (size_t i = 0; i < result->logits->size && i < 100; i++) {
        if (isnan(logits_data[i]) || isinf(logits_data[i])) {
            logits_valid = false;
            break;
        }
    }
    TEST_ASSERT(logits_valid, "Logits are finite values");
    
    // Test sequence inference
    uint32_t test_sequence[] = {1, 2, 3};
    size_t sequence_length = sizeof(test_sequence) / sizeof(test_sequence[0]);
    
    struct opencog_ml_inference_result* seq_result = opencog_ml_infer_sequence(
        engine, test_sequence, sequence_length, state
    );
    
    TEST_ASSERT(seq_result != NULL, "Sequence inference result");
    TEST_ASSERT(seq_result->error == OPENCOG_ML_ERROR_NONE, "Sequence inference success");
    TEST_ASSERT(seq_result->logits != NULL, "Sequence inference logits");
    TEST_ASSERT(seq_result->state != NULL, "Sequence inference output state");
    
    // Test custom sampling
    struct opencog_ml_inference_result* custom_result = opencog_ml_infer_with_sampling(
        engine, test_token, state, 0.8f, 20, 0.95f
    );
    
    TEST_ASSERT(custom_result != NULL, "Custom sampling result");
    TEST_ASSERT(custom_result->error == OPENCOG_ML_ERROR_NONE, "Custom sampling success");
    
    // Clean up
    opencog_ml_free_inference_result(result);
    opencog_ml_free_inference_result(seq_result);
    opencog_ml_free_inference_result(custom_result);
    opencog_ml_free_tensor(state);
    opencog_ml_free_engine(engine);
    
    TEST_PASS("All basic inference operations");
    return 1;
}

int test_error_handling() {
    printf("\n=== Testing Error Handling ===\n");
    
    // Test invalid engine configuration
    struct opencog_ml_engine_config invalid_config = {
        .model_path = "/nonexistent/model.bin",
        .n_threads = 1,
        .n_gpu_layers = 0,
        .enable_state_caching = true,
        .temperature = 1.0f,
        .top_k = 50,
        .top_p = 0.9f
    };
    
    struct opencog_ml_engine* invalid_engine = opencog_ml_init_engine(&invalid_config);
    TEST_ASSERT(invalid_engine == NULL, "Invalid model path rejection");
    
    // Test NULL parameter handling
    struct opencog_ml_tensor* null_tensor = opencog_ml_create_tensor(NULL, 0, OPENCOG_ML_TYPE_F32);
    TEST_ASSERT(null_tensor == NULL, "NULL shape parameter handling");
    
    // Test tensor view with NULL data
    size_t shape[] = {2, 2};
    struct opencog_ml_tensor* null_view = opencog_ml_create_tensor_view(
        NULL, shape, 2, OPENCOG_ML_TYPE_F32
    );
    TEST_ASSERT(null_view == NULL, "NULL data parameter handling");
    
    // Test copy with mismatched tensors
    struct opencog_ml_tensor* tensor1 = opencog_ml_create_tensor(shape, 2, OPENCOG_ML_TYPE_F32);
    size_t different_shape[] = {3, 3};
    struct opencog_ml_tensor* tensor2 = opencog_ml_create_tensor(different_shape, 2, OPENCOG_ML_TYPE_F32);
    
    if (tensor1 && tensor2) {
        bool copy_result = opencog_ml_copy_tensor(tensor1, tensor2);
        TEST_ASSERT(!copy_result, "Mismatched tensor copy rejection");
    }
    
    // Clean up
    opencog_ml_free_tensor(tensor1);
    opencog_ml_free_tensor(tensor2);
    
    TEST_PASS("All error handling tests");
    return 1;
}

int test_utility_functions() {
    printf("\n=== Testing Utility Functions ===\n");
    
    // Test debug mode setting (should not crash)
    opencog_ml_set_debug_mode(true);
    opencog_ml_set_debug_mode(false);
    TEST_PASS("Debug mode setting");
    
    // Test system info (should return a non-NULL string)
    const char* system_info = opencog_ml_get_system_info();
    TEST_ASSERT(system_info != NULL, "System info retrieval");
    TEST_ASSERT(strlen(system_info) > 0, "System info non-empty");
    
    printf("System info: %s\n", system_info);
    
    TEST_PASS("All utility functions");
    return 1;
}

int main(int argc, char* argv[]) {
    printf("OpenCog ML Inference Engine Test Suite\n");
    printf("======================================\n");
    
    // Check if model path is provided for inference tests
    if (argc > 1) {
        test_model_path = argv[1];
        printf("Using test model: %s\n", test_model_path);
    } else {
        printf("No model provided - skipping inference tests\n");
        printf("Usage: %s [model_path]\n", argv[0]);
    }
    
    int tests_passed = 0;
    int total_tests = 0;
    
    // Run all tests
    total_tests++; tests_passed += test_tensor_operations();
    total_tests++; tests_passed += test_engine_management();
    total_tests++; tests_passed += test_state_management();
    total_tests++; tests_passed += test_basic_inference();
    total_tests++; tests_passed += test_error_handling();
    total_tests++; tests_passed += test_utility_functions();
    
    // Print results
    printf("\n======================================\n");
    printf("Test Results: %d/%d tests passed\n", tests_passed, total_tests);
    
    if (tests_passed == total_tests) {
        printf("🎉 All tests passed!\n");
        return 0;
    } else {
        printf("❌ Some tests failed.\n");
        return 1;
    }
}