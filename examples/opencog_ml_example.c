/**
 * OpenCog ML Inference Engine C Example
 *
 * This example demonstrates how to use the OpenCog ML C API for RWKV inference.
 * It shows basic inference operations and state management.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../opencog_ml.h"

void print_usage(const char* program_name) {
    printf("Usage: %s <model_path>\n", program_name);
    printf("\nThis example demonstrates the OpenCog ML inference engine C API.\n");
    printf("You need to provide a path to an RWKV model file in ggml format.\n");
    printf("\nTo get a model:\n");
    printf("1. Download an RWKV model from https://huggingface.co/BlinkDL\n");
    printf("2. Convert it using: python python/convert_pytorch_to_ggml.py model.pth model.bin FP16\n");
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    
    printf("OpenCog ML Inference Engine C API Demo\n");
    printf("======================================\n");
    
    // Enable debug mode
    opencog_ml_set_debug_mode(true);
    
    // Print system information
    const char* system_info = opencog_ml_get_system_info();
    printf("System Info: %s\n", system_info);
    
    // Configure the inference engine
    struct opencog_ml_engine_config config = {
        .model_path = model_path,
        .n_threads = 4,
        .n_gpu_layers = 0,
        .enable_state_caching = true,
        .temperature = 1.0f,
        .top_k = 50,
        .top_p = 0.9f
    };
    
    printf("\nInitializing engine with model: %s\n", model_path);
    
    // Initialize the inference engine
    struct opencog_ml_engine* engine = opencog_ml_init_engine(&config);
    if (!engine) {
        printf("Error: Failed to initialize OpenCog ML engine\n");
        return 1;
    }
    
    printf("Engine initialized successfully!\n");
    
    // Get model information
    size_t vocab_size = opencog_ml_get_vocab_size(engine);
    size_t embed_size = opencog_ml_get_embed_size(engine);
    size_t state_size = opencog_ml_get_state_size(engine);
    
    printf("Vocabulary size: %zu\n", vocab_size);
    printf("Embedding size: %zu\n", embed_size);
    printf("State size: %zu\n", state_size);
    
    // Create initial state
    printf("\nCreating initial state...\n");
    struct opencog_ml_tensor* initial_state = opencog_ml_create_initial_state(engine);
    if (!initial_state) {
        printf("Error: Failed to create initial state\n");
        opencog_ml_free_engine(engine);
        return 1;
    }
    
    printf("Initial state created with size: %zu\n", initial_state->size);
    
    // Demonstrate single token inference
    printf("\n--- Single Token Inference ---\n");
    uint32_t test_token = 42;
    
    struct opencog_ml_inference_result* result = opencog_ml_infer_token(
        engine, test_token, initial_state
    );
    
    if (result && result->error == OPENCOG_ML_ERROR_NONE) {
        printf("Input token: %u\n", test_token);
        printf("Predicted next token: %u\n", result->predicted_token);
        printf("Confidence: %.4f\n", result->confidence);
        printf("Logits tensor size: %zu\n", result->logits->size);
        printf("Output state tensor size: %zu\n", result->state->size);
    } else {
        printf("Single token inference failed\n");
        if (result) {
            printf("Error code: %d\n", result->error);
        }
    }
    
    // Demonstrate sequence inference
    printf("\n--- Sequence Inference ---\n");
    uint32_t test_sequence[] = {1, 2, 3, 42, 100};
    size_t sequence_length = sizeof(test_sequence) / sizeof(test_sequence[0]);
    
    struct opencog_ml_inference_result* seq_result = opencog_ml_infer_sequence(
        engine, test_sequence, sequence_length, initial_state
    );
    
    if (seq_result && seq_result->error == OPENCOG_ML_ERROR_NONE) {
        printf("Input sequence: [");
        for (size_t i = 0; i < sequence_length; i++) {
            printf("%u", test_sequence[i]);
            if (i < sequence_length - 1) printf(", ");
        }
        printf("]\n");
        printf("Predicted next token: %u\n", seq_result->predicted_token);
        printf("Confidence: %.4f\n", seq_result->confidence);
    } else {
        printf("Sequence inference failed\n");
        if (seq_result) {
            printf("Error code: %d\n", seq_result->error);
        }
    }
    
    // Demonstrate custom sampling
    printf("\n--- Custom Sampling Inference ---\n");
    struct opencog_ml_inference_result* custom_result = opencog_ml_infer_with_sampling(
        engine, test_token, initial_state, 0.8f, 20, 0.95f
    );
    
    if (custom_result && custom_result->error == OPENCOG_ML_ERROR_NONE) {
        printf("Custom sampling result - predicted token: %u, confidence: %.4f\n",
               custom_result->predicted_token, custom_result->confidence);
    } else {
        printf("Custom sampling inference failed\n");
    }
    
    // Demonstrate engine cloning
    printf("\n--- Engine Cloning ---\n");
    struct opencog_ml_engine* cloned_engine = opencog_ml_clone_engine(engine, 2);
    if (cloned_engine) {
        printf("Engine cloned successfully\n");
        
        // Test cloned engine
        struct opencog_ml_inference_result* clone_result = opencog_ml_infer_token(
            cloned_engine, test_token, NULL
        );
        
        if (clone_result && clone_result->error == OPENCOG_ML_ERROR_NONE) {
            printf("Cloned engine inference successful: predicted token %u\n", 
                   clone_result->predicted_token);
        }
        
        // Clean up clone result
        if (clone_result) {
            opencog_ml_free_inference_result(clone_result);
        }
        
        // Free cloned engine
        opencog_ml_free_engine(cloned_engine);
    } else {
        printf("Failed to clone engine\n");
    }
    
    // Demonstrate state management
    printf("\n--- State Management ---\n");
    struct opencog_ml_tensor* new_state = opencog_ml_create_initial_state(engine);
    if (new_state) {
        printf("Created new state tensor\n");
        
        // Reset state
        bool reset_success = opencog_ml_reset_state(engine, new_state);
        printf("State reset: %s\n", reset_success ? "Success" : "Failed");
        
        // Clean up
        opencog_ml_free_tensor(new_state);
    }
    
    // Test error handling
    printf("\n--- Error Handling ---\n");
    enum opencog_ml_error_flags last_error = opencog_ml_get_last_error(engine);
    printf("Last error code: %d\n", last_error);
    
    // Clean up resources
    printf("\nCleaning up resources...\n");
    
    if (result) {
        opencog_ml_free_inference_result(result);
    }
    
    if (seq_result) {
        opencog_ml_free_inference_result(seq_result);
    }
    
    if (custom_result) {
        opencog_ml_free_inference_result(custom_result);
    }
    
    if (initial_state) {
        opencog_ml_free_tensor(initial_state);
    }
    
    opencog_ml_free_engine(engine);
    
    printf("\n=== Demo completed successfully! ===\n");
    printf("The OpenCog ML inference engine is now ready for integration\n");
    printf("with OpenCog's cognitive architecture.\n");
    
    return 0;
}