"""Platform detection utilities for VoiceCast.

Provides functions for detecting Apple Silicon, recommending devices,
and gathering platform information.
"""

import platform
from functools import lru_cache


@lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (M1/M2/M3/M4).

    Returns:
        True if on Apple Silicon Mac, False otherwise.
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@lru_cache(maxsize=1)
def is_macos() -> bool:
    """
    Check if running on macOS.

    Returns:
        True if on macOS, False otherwise.
    """
    return platform.system() == "Darwin"


@lru_cache(maxsize=1)
def has_cuda() -> bool:
    """
    Check if CUDA is available.

    Returns:
        True if CUDA is available, False otherwise.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@lru_cache(maxsize=1)
def has_mps() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available.

    Returns:
        True if MPS is available (Apple Silicon), False otherwise.
    """
    if not is_apple_silicon():
        return False
    try:
        import torch

        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def get_recommended_device() -> str:
    """
    Get the recommended compute device for the current platform.

    Priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if has_cuda():
        return "cuda"
    if has_mps():
        return "mps"
    return "cpu"


def get_platform_info() -> dict[str, str | bool]:
    """
    Get comprehensive platform information.

    Returns:
        Dictionary with platform details:
        - system: OS name (Darwin, Linux, Windows)
        - machine: Architecture (arm64, x86_64)
        - python_version: Python version string
        - is_apple_silicon: Boolean
        - recommended_device: Device string
        - has_cuda: Boolean
        - has_mps: Boolean
    """
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "is_apple_silicon": is_apple_silicon(),
        "is_macos": is_macos(),
        "recommended_device": get_recommended_device(),
        "has_cuda": has_cuda(),
        "has_mps": has_mps(),
    }
