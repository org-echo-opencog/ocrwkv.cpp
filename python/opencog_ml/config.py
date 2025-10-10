"""
Configuration classes for OpenCog ML Inference Engine
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class EngineConfig:
    """Configuration for OpenCog ML inference engine."""
    
    model_path: str
    n_threads: int = 4
    n_gpu_layers: int = 0
    enable_state_caching: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.model_path:
            raise ValueError("model_path cannot be empty")
        
        if self.n_threads <= 0:
            raise ValueError("n_threads must be positive")
            
        if self.n_gpu_layers < 0:
            raise ValueError("n_gpu_layers cannot be negative")
            
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
            
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
            
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError("top_p must be between 0 and 1")


@dataclass
class InferenceMode:
    """Enumeration of inference execution modes."""
    
    SINGLE_TOKEN = 0
    SEQUENCE = 1
    BATCH = 2


@dataclass 
class TensorType:
    """Enumeration of supported tensor data types."""
    
    F32 = 0
    F16 = 1
    I32 = 2
    U32 = 3


@dataclass
class ErrorFlags:
    """Error flags for OpenCog ML operations."""
    
    NONE = 0
    INVALID_ENGINE = 1
    INVALID_INPUT = 2
    MEMORY_ALLOCATION = 3
    INFERENCE_FAILED = 4
    TENSOR_SHAPE_MISMATCH = 5
    UNSUPPORTED_OPERATION = 6