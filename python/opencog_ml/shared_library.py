"""
Low-level interface to the OpenCog ML shared library
"""

import ctypes
import os
import platform
from typing import Optional, List
from .config import EngineConfig, TensorType, ErrorFlags


# Platform-specific library loading
def _get_library_path() -> str:
    """Get the expected path to the shared library."""
    system = platform.system()
    if system == "Windows":
        return "opencog_ml.dll"
    elif system == "Darwin":
        return "libopencog_ml.dylib"
    else:
        return "libopencog_ml.so"


# Pointer types
P_FLOAT = ctypes.POINTER(ctypes.c_float)
P_INT = ctypes.POINTER(ctypes.c_int32)
P_UINT = ctypes.POINTER(ctypes.c_uint32)
P_SIZE_T = ctypes.POINTER(ctypes.c_size_t)


class EngineConfigC(ctypes.Structure):
    """C structure for engine configuration."""
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("n_threads", ctypes.c_uint32),
        ("n_gpu_layers", ctypes.c_uint32),
        ("enable_state_caching", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_uint32),
        ("top_p", ctypes.c_float),
    ]


class TensorC(ctypes.Structure):
    """C structure for tensor representation."""
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", P_SIZE_T),
        ("ndim", ctypes.c_size_t),
        ("type", ctypes.c_int),
        ("size", ctypes.c_size_t),
    ]


class InferenceResultC(ctypes.Structure):
    """C structure for inference results."""
    _fields_ = [
        ("logits", ctypes.POINTER(TensorC)),
        ("state", ctypes.POINTER(TensorC)),
        ("confidence", ctypes.c_float),
        ("predicted_token", ctypes.c_uint32),
        ("error", ctypes.c_int),
    ]


class OpenCogMLSharedLibrary:
    """
    Low-level wrapper around the OpenCog ML shared library.
    """

    def __init__(self, shared_library_path: Optional[str] = None) -> None:
        """
        Load the shared library and set up function signatures.
        
        Parameters
        ----------
        shared_library_path : str, optional
            Path to the shared library. If None, will try to find it automatically.
        """
        if shared_library_path is None:
            shared_library_path = _get_library_path()

        try:
            self.library = ctypes.CDLL(shared_library_path)
        except OSError as e:
            raise RuntimeError(f"Failed to load OpenCog ML library from {shared_library_path}: {e}")

        self._setup_function_signatures()

    def _setup_function_signatures(self) -> None:
        """Set up C function signatures for type safety."""
        
        # Engine management
        self.library.opencog_ml_init_engine.argtypes = [ctypes.POINTER(EngineConfigC)]
        self.library.opencog_ml_init_engine.restype = ctypes.c_void_p

        self.library.opencog_ml_clone_engine.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self.library.opencog_ml_clone_engine.restype = ctypes.c_void_p

        self.library.opencog_ml_free_engine.argtypes = [ctypes.c_void_p]
        self.library.opencog_ml_free_engine.restype = None

        self.library.opencog_ml_get_last_error.argtypes = [ctypes.c_void_p]
        self.library.opencog_ml_get_last_error.restype = ctypes.c_int

        # Tensor operations
        self.library.opencog_ml_create_tensor.argtypes = [P_SIZE_T, ctypes.c_size_t, ctypes.c_int]
        self.library.opencog_ml_create_tensor.restype = ctypes.POINTER(TensorC)

        self.library.opencog_ml_create_tensor_view.argtypes = [ctypes.c_void_p, P_SIZE_T, ctypes.c_size_t, ctypes.c_int]
        self.library.opencog_ml_create_tensor_view.restype = ctypes.POINTER(TensorC)

        self.library.opencog_ml_free_tensor.argtypes = [ctypes.POINTER(TensorC)]
        self.library.opencog_ml_free_tensor.restype = None

        self.library.opencog_ml_copy_tensor.argtypes = [ctypes.POINTER(TensorC), ctypes.POINTER(TensorC)]
        self.library.opencog_ml_copy_tensor.restype = ctypes.c_bool

        # Inference functions
        self.library.opencog_ml_infer_token.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(TensorC)]
        self.library.opencog_ml_infer_token.restype = ctypes.POINTER(InferenceResultC)

        self.library.opencog_ml_infer_sequence.argtypes = [ctypes.c_void_p, P_UINT, ctypes.c_size_t, ctypes.POINTER(TensorC)]
        self.library.opencog_ml_infer_sequence.restype = ctypes.POINTER(InferenceResultC)

        self.library.opencog_ml_infer_with_sampling.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(TensorC), ctypes.c_float, ctypes.c_uint32, ctypes.c_float]
        self.library.opencog_ml_infer_with_sampling.restype = ctypes.POINTER(InferenceResultC)

        self.library.opencog_ml_free_inference_result.argtypes = [ctypes.POINTER(InferenceResultC)]
        self.library.opencog_ml_free_inference_result.restype = None

        # State management
        self.library.opencog_ml_create_initial_state.argtypes = [ctypes.c_void_p]
        self.library.opencog_ml_create_initial_state.restype = ctypes.POINTER(TensorC)

        self.library.opencog_ml_reset_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(TensorC)]
        self.library.opencog_ml_reset_state.restype = ctypes.c_bool

        # Utility functions
        self.library.opencog_ml_get_vocab_size.argtypes = [ctypes.c_void_p]
        self.library.opencog_ml_get_vocab_size.restype = ctypes.c_size_t

        self.library.opencog_ml_get_embed_size.argtypes = [ctypes.c_void_p]
        self.library.opencog_ml_get_embed_size.restype = ctypes.c_size_t

        self.library.opencog_ml_get_state_size.argtypes = [ctypes.c_void_p]
        self.library.opencog_ml_get_state_size.restype = ctypes.c_size_t

        self.library.opencog_ml_get_system_info.argtypes = []
        self.library.opencog_ml_get_system_info.restype = ctypes.c_char_p

        self.library.opencog_ml_set_debug_mode.argtypes = [ctypes.c_bool]
        self.library.opencog_ml_set_debug_mode.restype = None

    def init_engine(self, config: EngineConfig) -> ctypes.c_void_p:
        """Initialize an inference engine."""
        config_c = EngineConfigC(
            model_path=config.model_path.encode('utf-8'),
            n_threads=config.n_threads,
            n_gpu_layers=config.n_gpu_layers,
            enable_state_caching=config.enable_state_caching,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p
        )
        
        ptr = self.library.opencog_ml_init_engine(ctypes.byref(config_c))
        if not ptr:
            raise RuntimeError("Failed to initialize OpenCog ML engine")
        return ptr

    def clone_engine(self, engine_ptr: ctypes.c_void_p, n_threads: int) -> ctypes.c_void_p:
        """Clone an existing engine."""
        ptr = self.library.opencog_ml_clone_engine(engine_ptr, n_threads)
        if not ptr:
            raise RuntimeError("Failed to clone OpenCog ML engine")
        return ptr

    def free_engine(self, engine_ptr: ctypes.c_void_p) -> None:
        """Free an engine."""
        self.library.opencog_ml_free_engine(engine_ptr)

    def get_last_error(self, engine_ptr: ctypes.c_void_p) -> int:
        """Get the last error from the engine."""
        return self.library.opencog_ml_get_last_error(engine_ptr)

    def create_tensor(self, shape: List[int], tensor_type: int) -> ctypes.POINTER(TensorC):
        """Create a new tensor."""
        shape_array = (ctypes.c_size_t * len(shape))(*shape)
        ptr = self.library.opencog_ml_create_tensor(shape_array, len(shape), tensor_type)
        if not ptr:
            raise RuntimeError("Failed to create tensor")
        return ptr

    def free_tensor(self, tensor_ptr: ctypes.POINTER(TensorC)) -> None:
        """Free a tensor."""
        if tensor_ptr:
            self.library.opencog_ml_free_tensor(tensor_ptr)

    def infer_token(self, engine_ptr: ctypes.c_void_p, token: int, state_tensor: Optional[ctypes.POINTER(TensorC)] = None) -> ctypes.POINTER(InferenceResultC):
        """Perform single token inference."""
        result_ptr = self.library.opencog_ml_infer_token(engine_ptr, token, state_tensor)
        if not result_ptr:
            raise RuntimeError("Token inference failed")
        return result_ptr

    def infer_sequence(self, engine_ptr: ctypes.c_void_p, tokens: List[int], state_tensor: Optional[ctypes.POINTER(TensorC)] = None) -> ctypes.POINTER(InferenceResultC):
        """Perform sequence inference."""
        tokens_array = (ctypes.c_uint32 * len(tokens))(*tokens)
        result_ptr = self.library.opencog_ml_infer_sequence(engine_ptr, tokens_array, len(tokens), state_tensor)
        if not result_ptr:
            raise RuntimeError("Sequence inference failed")
        return result_ptr

    def free_inference_result(self, result_ptr: ctypes.POINTER(InferenceResultC)) -> None:
        """Free an inference result."""
        if result_ptr:
            self.library.opencog_ml_free_inference_result(result_ptr)

    def create_initial_state(self, engine_ptr: ctypes.c_void_p) -> ctypes.POINTER(TensorC):
        """Create initial state tensor."""
        state_ptr = self.library.opencog_ml_create_initial_state(engine_ptr)
        if not state_ptr:
            raise RuntimeError("Failed to create initial state")
        return state_ptr

    def get_vocab_size(self, engine_ptr: ctypes.c_void_p) -> int:
        """Get vocabulary size."""
        return self.library.opencog_ml_get_vocab_size(engine_ptr)

    def get_embed_size(self, engine_ptr: ctypes.c_void_p) -> int:
        """Get embedding size."""
        return self.library.opencog_ml_get_embed_size(engine_ptr)

    def get_state_size(self, engine_ptr: ctypes.c_void_p) -> int:
        """Get state size."""
        return self.library.opencog_ml_get_state_size(engine_ptr)

    def get_system_info(self) -> str:
        """Get system information."""
        result = self.library.opencog_ml_get_system_info()
        return result.decode('utf-8') if result else ""

    def set_debug_mode(self, enable: bool) -> None:
        """Enable or disable debug mode."""
        self.library.opencog_ml_set_debug_mode(enable)