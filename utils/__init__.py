"""Utility modules for VoiceCast."""

from utils.platform_utils import (
    get_platform_info,
    get_recommended_device,
    is_apple_silicon,
)

__all__ = [
    "is_apple_silicon",
    "get_recommended_device",
    "get_platform_info",
]
