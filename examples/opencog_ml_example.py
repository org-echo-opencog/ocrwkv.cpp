#!/usr/bin/env python3
"""
Example demonstrating OpenCog ML Inference Engine for RWKV

This example shows how to use the OpenCog ML interface as a pure inference 
engine for RWKV models. It demonstrates:
- Engine initialization
- Single token inference  
- Sequence inference
- Text generation
- State management
"""

import sys
import os

# Add the parent directory to the path to import opencog_ml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from opencog_ml import OpenCogMLEngine, EngineConfig, Tensor, set_debug_mode, get_system_info
    import numpy as np
except ImportError as e:
    print(f"Error importing OpenCog ML: {e}")
    print("Make sure the library is built and accessible.")
    sys.exit(1)


def demonstrate_basic_inference(model_path: str):
    """Demonstrate basic inference operations."""
    print("=== OpenCog ML Basic Inference Demo ===")
    
    # Enable debug mode for verbose output
    set_debug_mode(True)
    
    # Print system information
    print(f"System Info: {get_system_info()}")
    
    # Configure the inference engine
    config = EngineConfig(
        model_path=model_path,
        n_threads=4,
        n_gpu_layers=0,  # Use CPU for this demo
        enable_state_caching=True,
        temperature=1.0,
        top_k=50,
        top_p=0.9
    )
    
    print(f"Initializing engine with model: {model_path}")
    
    try:
        # Initialize the inference engine
        engine = OpenCogMLEngine(config)
        
        print(f"Engine initialized successfully!")
        print(f"Vocabulary size: {engine.vocab_size}")
        print(f"Embedding size: {engine.embed_size}")
        print(f"State size: {engine.state_size}")
        
        # Create initial state
        print("\nCreating initial state...")
        initial_state = engine.create_initial_state()
        print(f"Initial state shape: {initial_state.shape}")
        
        # Demonstrate single token inference
        print("\n--- Single Token Inference ---")
        test_token = 42  # Example token ID
        result = engine.infer_token(test_token, initial_state)
        
        if result.is_successful():
            print(f"Input token: {test_token}")
            print(f"Predicted next token: {result.predicted_token}")
            print(f"Confidence: {result.confidence:.4f}")
            print(f"Logits shape: {result.logits.shape}")
            print(f"Output state shape: {result.state.shape}")
        else:
            print(f"Single token inference failed: {result.get_error_message()}")
        
        # Demonstrate sequence inference
        print("\n--- Sequence Inference ---")
        test_sequence = [1, 2, 3, 42, 100]  # Example token sequence
        seq_result = engine.infer_sequence(test_sequence, initial_state)
        
        if seq_result.is_successful():
            print(f"Input sequence: {test_sequence}")
            print(f"Predicted next token: {seq_result.predicted_token}")
            print(f"Confidence: {seq_result.confidence:.4f}")
        else:
            print(f"Sequence inference failed: {seq_result.get_error_message()}")
        
        # Demonstrate text generation
        print("\n--- Text Generation ---")
        prompt = [1, 2, 3]  # Simple prompt
        generated_tokens, confidences = engine.generate(
            prompt, 
            max_length=10, 
            temperature=0.8
        )
        
        print(f"Prompt: {prompt}")
        print(f"Generated sequence: {generated_tokens}")
        print(f"Confidence scores: {[f'{c:.3f}' for c in confidences]}")
        
        # Demonstrate batch inference
        print("\n--- Batch Inference ---")
        batch_sequences = [
            [1, 2, 3],
            [4, 5, 6, 7],
            [8, 9]
        ]
        
        batch_results = engine.batch_infer(batch_sequences)
        
        for i, (seq, result) in enumerate(zip(batch_sequences, batch_results)):
            if result.is_successful():
                print(f"Sequence {i+1}: {seq} -> predicted token: {result.predicted_token} (confidence: {result.confidence:.3f})")
            else:
                print(f"Sequence {i+1} failed: {result.get_error_message()}")
        
        # Demonstrate engine cloning for multi-threading
        print("\n--- Engine Cloning ---")
        cloned_engine = engine.clone(n_threads=2)
        print(f"Cloned engine created with vocab size: {cloned_engine.vocab_size}")
        
        # Test cloned engine
        clone_result = cloned_engine.infer_token(test_token)
        if clone_result.is_successful():
            print(f"Cloned engine inference successful: predicted token {clone_result.predicted_token}")
        
        print("\n=== Demo completed successfully! ===")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        return False
    
    return True


def demonstrate_tensor_operations():
    """Demonstrate tensor operations."""
    print("\n=== Tensor Operations Demo ===")
    
    # Create tensors from different data sources
    print("Creating tensors from different data sources...")
    
    # From numpy array
    np_data = np.random.randn(5, 3).astype(np.float32)
    tensor1 = Tensor(np_data)
    print(f"Tensor from numpy: shape={tensor1.shape}, type={tensor1.tensor_type}")
    
    # From Python list
    list_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    tensor2 = Tensor(list_data)
    print(f"Tensor from list: shape={tensor2.shape}, type={tensor2.tensor_type}")
    
    # Convert back to numpy and list
    np_converted = tensor1.to_numpy()
    list_converted = tensor2.to_list()
    
    print(f"Converted back to numpy: shape={np_converted.shape}")
    print(f"Converted back to list: {list_converted}")
    
    print("=== Tensor Operations Demo completed ===")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python opencog_ml_example.py <model_path>")
        print("\nThis example demonstrates the OpenCog ML inference engine.")
        print("You need to provide a path to an RWKV model file in ggml format.")
        print("\nTo get a model:")
        print("1. Download an RWKV model from https://huggingface.co/BlinkDL")
        print("2. Convert it using: python python/convert_pytorch_to_ggml.py model.pth model.bin FP16")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print("OpenCog ML Inference Engine Demo")
    print("=" * 50)
    
    # Demonstrate tensor operations (doesn't require a model)
    demonstrate_tensor_operations()
    
    # Demonstrate inference operations (requires a model)
    print(f"\nLoading model from: {model_path}")
    success = demonstrate_basic_inference(model_path)
    
    if success:
        print("\n🎉 All demonstrations completed successfully!")
        print("\nThe OpenCog ML inference engine is now ready for integration")
        print("with OpenCog's cognitive architecture.")
    else:
        print("\n❌ Some demonstrations failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()