"""
High-level OpenCog ML Inference Engine interface
"""

import ctypes
import numpy as np
from typing import Optional, List, Union, Tuple, Any
from .shared_library import OpenCogMLSharedLibrary, TensorC, InferenceResultC
from .config import EngineConfig, TensorType, ErrorFlags


class Tensor:
    """
    High-level tensor wrapper for OpenCog ML.
    """

    def __init__(self, data: Union[np.ndarray, List[float], List[int]], tensor_type: Optional[int] = None):
        """
        Initialize a tensor from numpy array or list.
        
        Parameters
        ----------
        data : numpy.ndarray or list
            The tensor data
        tensor_type : int, optional
            The tensor type. If None, will be inferred from data type.
        """
        if isinstance(data, list):
            data = np.array(data)
        
        self.data = data
        self.shape = list(data.shape)
        self.ndim = len(self.shape)
        
        if tensor_type is None:
            if data.dtype == np.float32:
                self.tensor_type = TensorType.F32
            elif data.dtype == np.float16:
                self.tensor_type = TensorType.F16
            elif data.dtype == np.int32:
                self.tensor_type = TensorType.I32
            elif data.dtype == np.uint32:
                self.tensor_type = TensorType.U32
            else:
                # Default to F32 and convert
                self.data = data.astype(np.float32)
                self.tensor_type = TensorType.F32
        else:
            self.tensor_type = tensor_type
        
        self._c_tensor = None
        
    @classmethod
    def from_c_tensor(cls, c_tensor_ptr: ctypes.POINTER(TensorC)) -> 'Tensor':
        """Create a Tensor from a C tensor pointer."""
        if not c_tensor_ptr:
            raise ValueError("Invalid C tensor pointer")
            
        c_tensor = c_tensor_ptr.contents
        
        # Get shape
        shape = [c_tensor.shape[i] for i in range(c_tensor.ndim)]
        
        # Get data type
        if c_tensor.type == TensorType.F32:
            dtype = np.float32
        elif c_tensor.type == TensorType.F16:
            dtype = np.float16
        elif c_tensor.type == TensorType.I32:
            dtype = np.int32
        elif c_tensor.type == TensorType.U32:
            dtype = np.uint32
        else:
            raise ValueError(f"Unsupported tensor type: {c_tensor.type}")
        
        # Copy data from C tensor
        data_ptr = ctypes.cast(c_tensor.data, ctypes.POINTER(ctypes.c_float))
        data = np.ctypeslib.as_array(data_ptr, shape=(c_tensor.size,))
        data = data.reshape(shape).astype(dtype).copy()
        
        tensor = cls.__new__(cls)
        tensor.data = data
        tensor.shape = shape
        tensor.ndim = c_tensor.ndim
        tensor.tensor_type = c_tensor.type
        tensor._c_tensor = c_tensor_ptr
        
        return tensor
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.copy()
    
    def to_list(self) -> List:
        """Convert to Python list."""
        return self.data.tolist()
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, type={self.tensor_type}, data={self.data})"


class InferenceResult:
    """
    High-level wrapper for inference results.
    """

    def __init__(self, c_result_ptr: ctypes.POINTER(InferenceResultC)):
        """
        Initialize from a C inference result pointer.
        
        Parameters
        ----------
        c_result_ptr : ctypes.POINTER(InferenceResultC)
            Pointer to the C inference result structure
        """
        if not c_result_ptr:
            raise ValueError("Invalid inference result pointer")
            
        self._c_result = c_result_ptr
        c_result = c_result_ptr.contents
        
        # Convert tensors
        self.logits = Tensor.from_c_tensor(c_result.logits) if c_result.logits else None
        self.state = Tensor.from_c_tensor(c_result.state) if c_result.state else None
        
        self.confidence = float(c_result.confidence)
        self.predicted_token = int(c_result.predicted_token)
        self.error = int(c_result.error)
    
    def is_successful(self) -> bool:
        """Check if the inference was successful."""
        return self.error == ErrorFlags.NONE
    
    def get_error_message(self) -> str:
        """Get a human-readable error message."""
        error_messages = {
            ErrorFlags.NONE: "No error",
            ErrorFlags.INVALID_ENGINE: "Invalid engine",
            ErrorFlags.INVALID_INPUT: "Invalid input",
            ErrorFlags.MEMORY_ALLOCATION: "Memory allocation error",
            ErrorFlags.INFERENCE_FAILED: "Inference failed",
            ErrorFlags.TENSOR_SHAPE_MISMATCH: "Tensor shape mismatch",
            ErrorFlags.UNSUPPORTED_OPERATION: "Unsupported operation"
        }
        return error_messages.get(self.error, f"Unknown error: {self.error}")
    
    def __repr__(self) -> str:
        return f"InferenceResult(predicted_token={self.predicted_token}, confidence={self.confidence:.4f}, error={self.get_error_message()})"


class OpenCogMLEngine:
    """
    High-level OpenCog ML Inference Engine.
    
    This class provides a pure inference interface for RWKV models that can be
    integrated with OpenCog's cognitive architecture.
    """

    def __init__(self, config: EngineConfig, library_path: Optional[str] = None):
        """
        Initialize the OpenCog ML inference engine.
        
        Parameters
        ----------
        config : EngineConfig
            Engine configuration
        library_path : str, optional
            Path to the shared library. If None, will try to find it automatically.
        """
        self._library = OpenCogMLSharedLibrary(library_path)
        self._engine_ptr = self._library.init_engine(config)
        self._config = config
        
        # Cache model information
        self._vocab_size = self._library.get_vocab_size(self._engine_ptr)
        self._embed_size = self._library.get_embed_size(self._engine_ptr)
        self._state_size = self._library.get_state_size(self._engine_ptr)
    
    def clone(self, n_threads: Optional[int] = None) -> 'OpenCogMLEngine':
        """
        Create a clone of this engine for multi-threaded inference.
        
        Parameters
        ----------
        n_threads : int, optional
            Number of threads for the cloned engine. If None, uses the same as original.
        
        Returns
        -------
        OpenCogMLEngine
            A cloned engine instance
        """
        if n_threads is None:
            n_threads = self._config.n_threads
            
        cloned_engine = self.__new__(self.__class__)
        cloned_engine._library = self._library
        cloned_engine._engine_ptr = self._library.clone_engine(self._engine_ptr, n_threads)
        cloned_engine._config = self._config
        cloned_engine._vocab_size = self._vocab_size
        cloned_engine._embed_size = self._embed_size
        cloned_engine._state_size = self._state_size
        
        return cloned_engine
    
    def create_initial_state(self) -> Tensor:
        """
        Create an initial state tensor for inference.
        
        Returns
        -------
        Tensor
            Initial state tensor
        """
        state_ptr = self._library.create_initial_state(self._engine_ptr)
        return Tensor.from_c_tensor(state_ptr)
    
    def infer_token(self, token: int, state: Optional[Tensor] = None) -> InferenceResult:
        """
        Perform inference for a single token.
        
        Parameters
        ----------
        token : int
            Input token ID
        state : Tensor, optional
            Input state. If None, uses initial state.
        
        Returns
        -------
        InferenceResult
            The inference result containing logits, state, and prediction
        """
        state_ptr = state._c_tensor if state else None
        result_ptr = self._library.infer_token(self._engine_ptr, token, state_ptr)
        return InferenceResult(result_ptr)
    
    def infer_sequence(self, tokens: List[int], state: Optional[Tensor] = None) -> InferenceResult:
        """
        Perform inference for a sequence of tokens.
        
        Parameters
        ----------
        tokens : List[int]
            Input token sequence
        state : Tensor, optional
            Input state. If None, uses initial state.
        
        Returns
        -------
        InferenceResult
            The inference result containing logits, state, and prediction
        """
        state_ptr = state._c_tensor if state else None
        result_ptr = self._library.infer_sequence(self._engine_ptr, tokens, state_ptr)
        return InferenceResult(result_ptr)
    
    def generate(self, prompt_tokens: List[int], max_length: int = 100, 
                temperature: Optional[float] = None, top_k: Optional[int] = None, 
                top_p: Optional[float] = None) -> Tuple[List[int], List[float]]:
        """
        Generate a sequence of tokens using the model.
        
        Parameters
        ----------
        prompt_tokens : List[int]
            Initial prompt tokens
        max_length : int, default=100
            Maximum length of generated sequence
        temperature : float, optional
            Sampling temperature. If None, uses engine default.
        top_k : int, optional
            Top-K sampling parameter. If None, uses engine default.
        top_p : float, optional
            Top-P sampling parameter. If None, uses engine default.
        
        Returns
        -------
        Tuple[List[int], List[float]]
            Generated tokens and their confidence scores
        """
        generated_tokens = prompt_tokens.copy()
        confidence_scores = []
        
        # Start with initial state
        state = self.create_initial_state()
        
        # Process prompt
        if len(prompt_tokens) > 1:
            result = self.infer_sequence(prompt_tokens[:-1], state)
            if not result.is_successful():
                raise RuntimeError(f"Prompt processing failed: {result.get_error_message()}")
            state = result.state
        
        # Generate tokens
        current_token = prompt_tokens[-1] if prompt_tokens else 0
        
        for _ in range(max_length):
            result = self.infer_token(current_token, state)
            if not result.is_successful():
                raise RuntimeError(f"Generation failed: {result.get_error_message()}")
            
            current_token = result.predicted_token
            generated_tokens.append(current_token)
            confidence_scores.append(result.confidence)
            state = result.state
            
            # TODO: Add early stopping conditions (e.g., EOS token)
        
        return generated_tokens, confidence_scores
    
    def batch_infer(self, token_sequences: List[List[int]], 
                   states: Optional[List[Tensor]] = None) -> List[InferenceResult]:
        """
        Perform batch inference on multiple sequences.
        
        Parameters
        ----------
        token_sequences : List[List[int]]
            List of token sequences to process
        states : List[Tensor], optional
            List of input states. If None, uses initial states for all sequences.
        
        Returns
        -------
        List[InferenceResult]
            List of inference results
        """
        results = []
        for i, tokens in enumerate(token_sequences):
            state = states[i] if states and i < len(states) else None
            result = self.infer_sequence(tokens, state)
            results.append(result)
        return results
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._vocab_size
    
    @property
    def embed_size(self) -> int:
        """Get the embedding dimension."""
        return self._embed_size
    
    @property
    def state_size(self) -> int:
        """Get the state tensor size."""
        return self._state_size
    
    def get_last_error(self) -> int:
        """Get the last error code."""
        return self._library.get_last_error(self._engine_ptr)
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, '_library') and hasattr(self, '_engine_ptr'):
            self._library.free_engine(self._engine_ptr)
    
    def __repr__(self) -> str:
        return f"OpenCogMLEngine(vocab_size={self.vocab_size}, embed_size={self.embed_size}, state_size={self.state_size})"