"""
Utility functions for OpenCog ML
"""

from .shared_library import OpenCogMLSharedLibrary


def set_debug_mode(enable: bool) -> None:
    """
    Enable or disable debug logging for OpenCog ML.
    
    Parameters
    ----------
    enable : bool
        Whether to enable debug mode
    """
    # Create a temporary library instance to access the function
    try:
        lib = OpenCogMLSharedLibrary()
        lib.set_debug_mode(enable)
    except RuntimeError:
        # Library not available, ignore
        pass


def get_system_info() -> str:
    """
    Get system information string.
    
    Returns
    -------
    str
        System information
    """
    try:
        lib = OpenCogMLSharedLibrary()
        return lib.get_system_info()
    except RuntimeError:
        return "OpenCog ML library not available"