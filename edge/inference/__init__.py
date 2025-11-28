"""
Inference module for Jetson edge deployment.
Supports TensorRT, ONNX Runtime, and TensorFlow Lite backends.
"""

from .detector import ObjectDetector
from .tensorrt_backend import TensorRTBackend
from .onnx_backend import ONNXBackend
from .tflite_backend import TFLiteBackend

__all__ = [
    "ObjectDetector",
    "TensorRTBackend",
    "ONNXBackend",
    "TFLiteBackend",
]
