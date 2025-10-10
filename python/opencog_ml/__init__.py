"""
OpenCog ML Inference Engine for RWKV

This package provides a pure inference interface for RWKV models that can be integrated
with OpenCog's cognitive architecture. It offers high-level abstractions for neural 
inference while maintaining compatibility with OpenCog's knowledge representation.
"""

from .inference_engine import OpenCogMLEngine, InferenceResult, Tensor
from .config import EngineConfig
from .utils import set_debug_mode, get_system_info

__version__ = "1.0.0"
__author__ = "OpenCog ML Team"

__all__ = [
    "OpenCogMLEngine",
    "InferenceResult", 
    "Tensor",
    "EngineConfig",
    "set_debug_mode",
    "get_system_info"
]