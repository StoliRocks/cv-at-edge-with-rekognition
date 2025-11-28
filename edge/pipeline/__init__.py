"""
Video pipeline module for Jetson edge deployment.
Supports GStreamer, DeepStream, and OpenCV backends.
"""

from .video_source import VideoSource, VideoSourceType
from .frame_processor import FrameProcessor
from .gstreamer_pipeline import GStreamerPipeline

__all__ = [
    "VideoSource",
    "VideoSourceType",
    "FrameProcessor",
    "GStreamerPipeline",
]
