"""
Object Detector module for edge inference.
Supports multiple backends: TensorRT, ONNX Runtime, TensorFlow Lite.
Optimized for NVIDIA Jetson devices.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Supported inference backends."""
    TENSORRT = "tensorrt"
    ONNX = "onnx"
    TFLITE = "tflite"
    AUTO = "auto"


@dataclass
class Detection:
    """Represents a single object detection."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized 0-1

    def to_dict(self) -> dict:
        """Convert detection to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": {
                "x1": self.bbox[0],
                "y1": self.bbox[1],
                "x2": self.bbox[2],
                "y2": self.bbox[3],
                "width": self.bbox[2] - self.bbox[0],
                "height": self.bbox[3] - self.bbox[1],
            }
        }

    def get_pixel_bbox(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized bbox to pixel coordinates."""
        return (
            int(self.bbox[0] * img_width),
            int(self.bbox[1] * img_height),
            int(self.bbox[2] * img_width),
            int(self.bbox[3] * img_height),
        )


@dataclass
class InferenceResult:
    """Result of an inference operation."""
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    preprocess_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    frame_id: int = 0
    timestamp: float = 0.0

    @property
    def total_time_ms(self) -> float:
        """Total processing time in milliseconds."""
        return self.preprocess_time_ms + self.inference_time_ms + self.postprocess_time_ms

    def get_detections_by_class(self, class_name: str) -> List[Detection]:
        """Filter detections by class name."""
        return [d for d in self.detections if d.class_name.lower() == class_name.lower()]

    def to_dict(self) -> dict:
        """Convert result to dictionary."""
        return {
            "detections": [d.to_dict() for d in self.detections],
            "timing": {
                "preprocess_ms": self.preprocess_time_ms,
                "inference_ms": self.inference_time_ms,
                "postprocess_ms": self.postprocess_time_ms,
                "total_ms": self.total_time_ms,
            },
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
        }


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @abstractmethod
    def load_model(self, model_path: Path) -> None:
        """Load model from file."""
        pass

    @abstractmethod
    def infer(self, image: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed image."""
        pass

    @abstractmethod
    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Get expected input shape (batch, channels, height, width)."""
        pass

    @abstractmethod
    def warmup(self, iterations: int = 10) -> None:
        """Warm up the model for consistent performance."""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass


class ObjectDetector:
    """
    High-level object detector optimized for Jetson edge devices.
    Supports YOLOv8, YOLOv5, and similar architectures.
    """

    # COCO class names (80 classes)
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]

    def __init__(
        self,
        model_path: str,
        backend: BackendType = BackendType.AUTO,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        target_classes: Optional[List[str]] = None,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "cuda:0",
    ):
        """
        Initialize the object detector.

        Args:
            model_path: Path to the model file (.engine, .onnx, or .tflite)
            backend: Inference backend to use (auto-detect based on file extension)
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            target_classes: List of class names to detect (None = all classes)
            input_size: Model input size (width, height)
            device: Device to run inference on
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target_classes = [c.lower() for c in target_classes] if target_classes else None
        self.input_size = input_size
        self.device = device
        self._frame_count = 0

        # Auto-detect backend if needed
        if backend == BackendType.AUTO:
            backend = self._detect_backend()

        self.backend_type = backend
        self.backend: Optional[InferenceBackend] = None

        logger.info(f"Initializing ObjectDetector with {backend.value} backend")

    def _detect_backend(self) -> BackendType:
        """Auto-detect backend based on model file extension."""
        suffix = self.model_path.suffix.lower()
        if suffix == ".engine" or suffix == ".trt":
            return BackendType.TENSORRT
        elif suffix == ".onnx":
            return BackendType.ONNX
        elif suffix == ".tflite":
            return BackendType.TFLITE
        else:
            logger.warning(f"Unknown model format: {suffix}, defaulting to ONNX")
            return BackendType.ONNX

    def load(self) -> None:
        """Load the model and initialize the backend."""
        if self.backend_type == BackendType.TENSORRT:
            from .tensorrt_backend import TensorRTBackend
            self.backend = TensorRTBackend(device=self.device)
        elif self.backend_type == BackendType.ONNX:
            from .onnx_backend import ONNXBackend
            self.backend = ONNXBackend(device=self.device)
        elif self.backend_type == BackendType.TFLITE:
            from .tflite_backend import TFLiteBackend
            self.backend = TFLiteBackend()
        else:
            raise ValueError(f"Unsupported backend: {self.backend_type}")

        logger.info(f"Loading model from {self.model_path}")
        self.backend.load_model(self.model_path)
        logger.info("Model loaded successfully")

    def warmup(self, iterations: int = 10) -> None:
        """Warm up the model for consistent inference times."""
        if self.backend is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        logger.info(f"Warming up model with {iterations} iterations")
        self.backend.warmup(iterations)
        logger.info("Warmup complete")

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float]]:
        """
        Preprocess image for inference.

        Args:
            image: Input image in BGR format (OpenCV format)

        Returns:
            Tuple of (preprocessed_image, original_size, scale_factors)
        """
        original_height, original_width = image.shape[:2]
        target_width, target_height = self.input_size

        # Calculate scale factors
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        # Resize with letterboxing for aspect ratio preservation
        scale = min(scale_x, scale_y)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Import cv2 here to avoid issues if not installed
        import cv2

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create letterboxed image
        padded = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        pad_x = (target_width - new_width) // 2
        pad_y = (target_height - new_height) // 2
        padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized

        # Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1 and convert to NCHW format
        normalized = rgb.astype(np.float32) / 255.0
        transposed = normalized.transpose(2, 0, 1)  # HWC -> CHW
        batched = np.expand_dims(transposed, axis=0)  # Add batch dimension

        # Store padding info for postprocessing
        self._pad_info = (pad_x, pad_y, scale)

        return batched, (original_width, original_height), (scale_x, scale_y)

    def postprocess(
        self,
        outputs: np.ndarray,
        original_size: Tuple[int, int],
    ) -> List[Detection]:
        """
        Post-process model outputs to get detections.
        Handles YOLOv8 output format.

        Args:
            outputs: Raw model outputs
            original_size: Original image size (width, height)

        Returns:
            List of Detection objects
        """
        # YOLOv8 output shape: (1, 84, 8400) where 84 = 4 (bbox) + 80 (classes)
        # Transpose to (1, 8400, 84)
        if len(outputs.shape) == 3 and outputs.shape[1] == 84:
            outputs = outputs.transpose(0, 2, 1)

        predictions = outputs[0]  # Remove batch dimension

        # Extract boxes and scores
        boxes = predictions[:, :4]  # x_center, y_center, width, height
        scores = predictions[:, 4:]  # Class scores

        # Get best class for each detection
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        mask = confidences >= self.confidence_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert from center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Apply letterbox correction
        pad_x, pad_y, scale = self._pad_info
        target_width, target_height = self.input_size
        original_width, original_height = original_size

        # Remove padding offset and scale back
        x1 = (x1 - pad_x) / scale / original_width
        y1 = (y1 - pad_y) / scale / original_height
        x2 = (x2 - pad_x) / scale / original_width
        y2 = (y2 - pad_y) / scale / original_height

        # Clip to valid range
        x1 = np.clip(x1, 0, 1)
        y1 = np.clip(y1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        y2 = np.clip(y2, 0, 1)

        # Stack boxes for NMS
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS
        indices = self._nms(boxes_xyxy, confidences, self.nms_threshold)

        # Create detections
        detections = []
        for idx in indices:
            class_id = int(class_ids[idx])
            class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else f"class_{class_id}"

            # Filter by target classes if specified
            if self.target_classes and class_name.lower() not in self.target_classes:
                continue

            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidences[idx]),
                bbox=(float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])),
            )
            detections.append(detection)

        return detections

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Apply Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            intersection = w * h
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, image: np.ndarray) -> InferenceResult:
        """
        Run object detection on an image.

        Args:
            image: Input image in BGR format (OpenCV format)

        Returns:
            InferenceResult containing detections and timing info
        """
        if self.backend is None or not self.backend.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        self._frame_count += 1
        timestamp = time.time()

        # Preprocess
        t0 = time.perf_counter()
        preprocessed, original_size, _ = self.preprocess(image)
        preprocess_time = (time.perf_counter() - t0) * 1000

        # Inference
        t0 = time.perf_counter()
        outputs = self.backend.infer(preprocessed)
        inference_time = (time.perf_counter() - t0) * 1000

        # Postprocess
        t0 = time.perf_counter()
        detections = self.postprocess(outputs, original_size)
        postprocess_time = (time.perf_counter() - t0) * 1000

        return InferenceResult(
            detections=detections,
            inference_time_ms=inference_time,
            preprocess_time_ms=preprocess_time,
            postprocess_time_ms=postprocess_time,
            frame_id=self._frame_count,
            timestamp=timestamp,
        )

    def __enter__(self):
        """Context manager entry."""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
