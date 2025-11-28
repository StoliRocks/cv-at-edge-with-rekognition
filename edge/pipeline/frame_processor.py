"""
Frame processor for running inference and computing analytics.
Handles region of interest (ROI) calculations, IoU metrics, and state management.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from ..inference.detector import Detection, InferenceResult, ObjectDetector

logger = logging.getLogger(__name__)


class OccupancyState(Enum):
    """Occupancy state for monitored areas."""
    VACANT = "Vacant"
    OCCUPIED = "Occupied"
    UNKNOWN = "Unknown"


@dataclass
class RegionOfInterest:
    """Defines a region of interest for monitoring."""
    name: str
    x1: float  # Normalized 0-1
    y1: float
    x2: float
    y2: float

    def to_pixel_coords(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coords to pixel coordinates."""
        return (
            int(self.x1 * width),
            int(self.y1 * height),
            int(self.x2 * width),
            int(self.y2 * height),
        )

    @property
    def area(self) -> float:
        """Calculate normalized area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class ProcessingResult:
    """Result of frame processing."""
    frame_id: int
    timestamp: float
    detections: List[Detection]
    target_detections: List[Detection]
    best_detection: Optional[Detection]
    iou: float
    occupancy_state: OccupancyState
    confidence: float
    roi: RegionOfInterest
    inference_time_ms: float
    total_time_ms: float
    state_changed: bool = False
    previous_state: Optional[OccupancyState] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "detections_count": len(self.detections),
            "target_detections_count": len(self.target_detections),
            "best_detection": self.best_detection.to_dict() if self.best_detection else None,
            "iou": self.iou,
            "occupancy_state": self.occupancy_state.value,
            "confidence": self.confidence,
            "roi": {
                "name": self.roi.name,
                "x1": self.roi.x1,
                "y1": self.roi.y1,
                "x2": self.roi.x2,
                "y2": self.roi.y2,
            },
            "timing": {
                "inference_ms": self.inference_time_ms,
                "total_ms": self.total_time_ms,
            },
            "state_changed": self.state_changed,
            "previous_state": self.previous_state.value if self.previous_state else None,
        }


class FrameProcessor:
    """
    Processes video frames for object detection and occupancy monitoring.
    Optimized for garage bay / parking spot detection use cases.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        roi: RegionOfInterest,
        target_classes: List[str] = ["truck"],
        iou_threshold: float = 0.3,
        confidence_threshold: float = 0.5,
        state_persistence_frames: int = 3,
    ):
        """
        Initialize frame processor.

        Args:
            detector: Object detector instance
            roi: Region of interest to monitor
            target_classes: Classes to detect (e.g., ["truck", "car"])
            iou_threshold: Minimum IoU to consider object in ROI
            confidence_threshold: Minimum detection confidence
            state_persistence_frames: Frames before state change is confirmed
        """
        self.detector = detector
        self.roi = roi
        self.target_classes = [c.lower() for c in target_classes]
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.state_persistence_frames = state_persistence_frames

        # State tracking
        self._current_state = OccupancyState.UNKNOWN
        self._state_counter = 0
        self._pending_state: Optional[OccupancyState] = None
        self._state_start_time: Optional[float] = None
        self._frame_count = 0

        # Performance metrics
        self._fps_counter = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0

    def calculate_iou(self, detection: Detection) -> float:
        """
        Calculate Intersection over Union between detection and ROI.

        Args:
            detection: Detection with normalized bounding box

        Returns:
            IoU value between 0 and 1
        """
        # Detection bbox
        d_x1, d_y1, d_x2, d_y2 = detection.bbox

        # ROI bbox
        r_x1, r_y1, r_x2, r_y2 = self.roi.x1, self.roi.y1, self.roi.x2, self.roi.y2

        # Calculate intersection
        inter_x1 = max(d_x1, r_x1)
        inter_y1 = max(d_y1, r_y1)
        inter_x2 = min(d_x2, r_x2)
        inter_y2 = min(d_y2, r_y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Calculate union
        d_area = (d_x2 - d_x1) * (d_y2 - d_y1)
        r_area = (r_x2 - r_x1) * (r_y2 - r_y1)
        union_area = d_area + r_area - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def process_frame(self, frame: np.ndarray) -> ProcessingResult:
        """
        Process a single frame for object detection and occupancy.

        Args:
            frame: Input frame in BGR format

        Returns:
            ProcessingResult with detections and state
        """
        start_time = time.time()
        self._frame_count += 1

        # Run detection
        inference_result = self.detector.detect(frame)

        # Filter for target classes
        target_detections = [
            d for d in inference_result.detections
            if d.class_name.lower() in self.target_classes
            and d.confidence >= self.confidence_threshold
        ]

        # Calculate IoU for each target detection
        best_detection = None
        best_iou = 0.0
        best_confidence = 0.0

        for detection in target_detections:
            iou = self.calculate_iou(detection)
            if iou > best_iou:
                best_iou = iou
                best_detection = detection
                best_confidence = detection.confidence

        # Determine occupancy based on IoU
        detected_state = (
            OccupancyState.OCCUPIED
            if best_iou >= self.iou_threshold
            else OccupancyState.VACANT
        )

        # Apply state persistence (hysteresis)
        state_changed = False
        previous_state = self._current_state

        if detected_state != self._pending_state:
            self._pending_state = detected_state
            self._state_counter = 1
        else:
            self._state_counter += 1

        if self._state_counter >= self.state_persistence_frames:
            if self._pending_state != self._current_state:
                state_changed = True
                previous_state = self._current_state
                self._current_state = self._pending_state
                self._state_start_time = time.time()
                logger.info(
                    f"State changed: {previous_state.value} -> {self._current_state.value} "
                    f"(IoU: {best_iou:.2f}, confidence: {best_confidence:.2f})"
                )

        # Calculate FPS
        self._fps_counter += 1
        elapsed = time.time() - self._fps_start_time
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start_time = time.time()

        total_time = (time.time() - start_time) * 1000

        return ProcessingResult(
            frame_id=self._frame_count,
            timestamp=time.time(),
            detections=inference_result.detections,
            target_detections=target_detections,
            best_detection=best_detection,
            iou=best_iou,
            occupancy_state=self._current_state,
            confidence=best_confidence,
            roi=self.roi,
            inference_time_ms=inference_result.inference_time_ms,
            total_time_ms=total_time,
            state_changed=state_changed,
            previous_state=previous_state if state_changed else None,
        )

    @property
    def current_state(self) -> OccupancyState:
        """Get current occupancy state."""
        return self._current_state

    @property
    def current_fps(self) -> float:
        """Get current processing FPS."""
        return self._current_fps

    @property
    def state_duration(self) -> float:
        """Get duration of current state in seconds."""
        if self._state_start_time is None:
            return 0.0
        return time.time() - self._state_start_time

    def reset_state(self) -> None:
        """Reset state tracking."""
        self._current_state = OccupancyState.UNKNOWN
        self._state_counter = 0
        self._pending_state = None
        self._state_start_time = None


def draw_detections(
    frame: np.ndarray,
    result: ProcessingResult,
    draw_roi: bool = True,
    draw_all_detections: bool = False,
) -> np.ndarray:
    """
    Draw detection results on frame.

    Args:
        frame: Input frame (will be modified in place)
        result: Processing result to visualize
        draw_roi: Draw the region of interest
        draw_all_detections: Draw all detections or just target detections

    Returns:
        Frame with annotations
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for drawing")
        return frame

    height, width = frame.shape[:2]

    # Draw ROI
    if draw_roi:
        roi_coords = result.roi.to_pixel_coords(width, height)
        cv2.rectangle(
            frame,
            (roi_coords[0], roi_coords[1]),
            (roi_coords[2], roi_coords[3]),
            (255, 255, 0),  # Cyan
            2,
        )
        cv2.putText(
            frame,
            f"ROI: {result.roi.name}",
            (roi_coords[0], roi_coords[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    # Draw detections
    detections_to_draw = (
        result.detections if draw_all_detections else result.target_detections
    )

    for detection in detections_to_draw:
        bbox = detection.get_pixel_bbox(width, height)

        # Color based on whether it's the best detection
        if detection == result.best_detection:
            color = (0, 255, 0)  # Green for best match
            thickness = 3
        else:
            color = (0, 0, 255)  # Red for other detections
            thickness = 2

        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

        # Label
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        if detection == result.best_detection:
            label += f" IoU: {result.iou:.2f}"

        cv2.putText(
            frame,
            label,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    # Draw status overlay
    status_color = (0, 255, 0) if result.occupancy_state == OccupancyState.VACANT else (0, 0, 255)
    cv2.putText(
        frame,
        f"Status: {result.occupancy_state.value}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        status_color,
        2,
    )

    cv2.putText(
        frame,
        f"Inference: {result.inference_time_ms:.1f}ms",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return frame
