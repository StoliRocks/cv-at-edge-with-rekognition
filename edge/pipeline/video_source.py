"""
Video source abstraction for multiple input types.
Supports cameras, RTSP streams, video files, and image directories.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from threading import Thread, Event
from typing import Optional, Generator, Tuple, Callable

import numpy as np

logger = logging.getLogger(__name__)


class VideoSourceType(Enum):
    """Supported video source types."""
    FILE = "file"
    CAMERA = "camera"
    RTSP = "rtsp"
    CSI = "csi"  # Jetson CSI camera
    USB = "usb"
    IMAGES = "images"


@dataclass
class FrameInfo:
    """Metadata about a video frame."""
    frame_id: int
    timestamp: float
    width: int
    height: int
    fps: float
    source_type: VideoSourceType


class VideoSource(ABC):
    """Abstract base class for video sources."""

    @abstractmethod
    def open(self) -> bool:
        """Open the video source."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[FrameInfo]]:
        """Read a frame from the source."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the video source."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if source is opened."""
        pass

    @property
    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        """Get frame size (width, height)."""
        pass


class OpenCVVideoSource(VideoSource):
    """
    OpenCV-based video source.
    Simple and portable, works on most platforms.
    """

    def __init__(
        self,
        source: str,
        source_type: VideoSourceType = VideoSourceType.FILE,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        buffer_size: int = 1,
    ):
        """
        Initialize OpenCV video source.

        Args:
            source: Video source (file path, camera index, or RTSP URL)
            source_type: Type of video source
            width: Desired frame width
            height: Desired frame height
            fps: Desired FPS
            buffer_size: OpenCV buffer size (1 = minimal latency)
        """
        self.source = source
        self.source_type = source_type
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size

        self._cap = None
        self._frame_count = 0
        self._actual_fps = fps

    def open(self) -> bool:
        """Open the video source."""
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV not installed. Install with: pip install opencv-python")

        # Determine OpenCV capture source
        if self.source_type == VideoSourceType.FILE:
            cap_source = str(self.source)
        elif self.source_type == VideoSourceType.CAMERA:
            cap_source = int(self.source) if isinstance(self.source, (int, str)) and str(self.source).isdigit() else 0
        elif self.source_type == VideoSourceType.RTSP:
            cap_source = self.source
        elif self.source_type == VideoSourceType.USB:
            cap_source = int(self.source) if str(self.source).isdigit() else f"/dev/video{self.source}"
        else:
            cap_source = str(self.source)

        logger.info(f"Opening video source: {cap_source} (type: {self.source_type.value})")

        # Use GStreamer backend for better performance on Jetson
        if self.source_type in [VideoSourceType.RTSP, VideoSourceType.CSI]:
            self._cap = cv2.VideoCapture(cap_source, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(cap_source)

        if not self._cap.isOpened():
            logger.error(f"Failed to open video source: {cap_source}")
            return False

        # Configure capture properties
        if self.source_type != VideoSourceType.FILE:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        # Get actual properties
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or self.fps

        logger.info(f"Video source opened: {actual_width}x{actual_height} @ {self._actual_fps:.1f} FPS")

        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[FrameInfo]]:
        """Read a frame from the source."""
        if self._cap is None or not self._cap.isOpened():
            return False, None, None

        ret, frame = self._cap.read()
        if not ret:
            return False, None, None

        self._frame_count += 1

        info = FrameInfo(
            frame_id=self._frame_count,
            timestamp=time.time(),
            width=frame.shape[1],
            height=frame.shape[0],
            fps=self._actual_fps,
            source_type=self.source_type,
        )

        return True, frame, info

    def close(self) -> None:
        """Close the video source."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Video source closed")

    def is_opened(self) -> bool:
        """Check if source is opened."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get frame size (width, height)."""
        if self._cap is None:
            return (self.width, self.height)
        import cv2
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class JetsonCSISource(VideoSource):
    """
    Optimized video source for Jetson CSI cameras using GStreamer.
    Supports nvarguscamerasrc for hardware-accelerated capture.
    """

    def __init__(
        self,
        sensor_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        flip_method: int = 0,
    ):
        """
        Initialize Jetson CSI camera source.

        Args:
            sensor_id: CSI camera sensor ID (0 or 1)
            width: Capture width
            height: Capture height
            fps: Frame rate
            flip_method: Flip method (0=none, 1=ccw90, 2=180, 3=cw90, 4=hflip, 5=ur-diag, 6=vflip, 7=ul-diag)
        """
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method

        self._cap = None
        self._frame_count = 0

    def _build_gstreamer_pipeline(self) -> str:
        """Build GStreamer pipeline string for Jetson CSI camera."""
        return (
            f"nvarguscamerasrc sensor-id={self.sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv flip-method={self.flip_method} ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink drop=1"
        )

    def open(self) -> bool:
        """Open the CSI camera."""
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV not installed")

        pipeline = self._build_gstreamer_pipeline()
        logger.info(f"Opening CSI camera with GStreamer pipeline:\n{pipeline}")

        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            logger.error("Failed to open CSI camera")
            return False

        logger.info(f"CSI camera opened: {self.width}x{self.height} @ {self.fps} FPS")
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[FrameInfo]]:
        """Read a frame from the CSI camera."""
        if self._cap is None or not self._cap.isOpened():
            return False, None, None

        ret, frame = self._cap.read()
        if not ret:
            return False, None, None

        self._frame_count += 1

        info = FrameInfo(
            frame_id=self._frame_count,
            timestamp=time.time(),
            width=frame.shape[1],
            height=frame.shape[0],
            fps=self.fps,
            source_type=VideoSourceType.CSI,
        )

        return True, frame, info

    def close(self) -> None:
        """Close the CSI camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("CSI camera closed")

    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get frame size (width, height)."""
        return (self.width, self.height)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BufferedVideoSource:
    """
    Buffered video source with background thread for reduced latency.
    Useful for high-FPS capture when processing can't keep up.
    """

    def __init__(
        self,
        source: VideoSource,
        buffer_size: int = 2,
        drop_frames: bool = True,
    ):
        """
        Initialize buffered video source.

        Args:
            source: Underlying video source
            buffer_size: Maximum frames to buffer
            drop_frames: Drop old frames when buffer is full
        """
        self.source = source
        self.buffer_size = buffer_size
        self.drop_frames = drop_frames

        self._queue: Queue = Queue(maxsize=buffer_size)
        self._thread: Optional[Thread] = None
        self._stop_event = Event()
        self._running = False

    def start(self) -> bool:
        """Start the buffered capture."""
        if not self.source.open():
            return False

        self._stop_event.clear()
        self._running = True
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info("Buffered video capture started")
        return True

    def _capture_loop(self) -> None:
        """Background capture loop."""
        while not self._stop_event.is_set():
            ret, frame, info = self.source.read()
            if not ret:
                if self.source.is_opened():
                    continue
                break

            if self._queue.full() and self.drop_frames:
                try:
                    self._queue.get_nowait()  # Drop oldest frame
                except Empty:
                    pass

            try:
                self._queue.put((frame, info), timeout=0.1)
            except:
                pass

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray], Optional[FrameInfo]]:
        """Read a frame from the buffer."""
        try:
            frame, info = self._queue.get(timeout=timeout)
            return True, frame, info
        except Empty:
            return False, None, None

    def stop(self) -> None:
        """Stop the buffered capture."""
        self._stop_event.set()
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        self.source.close()
        logger.info("Buffered video capture stopped")

    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def create_video_source(
    source: str,
    source_type: Optional[VideoSourceType] = None,
    **kwargs
) -> VideoSource:
    """
    Factory function to create appropriate video source.

    Args:
        source: Video source string (path, URL, device index)
        source_type: Explicit source type (auto-detected if None)
        **kwargs: Additional arguments for the video source

    Returns:
        Configured VideoSource instance
    """
    # Auto-detect source type
    if source_type is None:
        source_lower = str(source).lower()
        if source_lower.startswith("rtsp://"):
            source_type = VideoSourceType.RTSP
        elif source_lower.startswith("/dev/video") or source_lower.isdigit():
            source_type = VideoSourceType.USB
        elif Path(source).is_file():
            source_type = VideoSourceType.FILE
        elif Path(source).is_dir():
            source_type = VideoSourceType.IMAGES
        else:
            source_type = VideoSourceType.CAMERA

    logger.info(f"Creating video source: {source} (type: {source_type.value})")

    # Create appropriate source
    if source_type == VideoSourceType.CSI:
        sensor_id = int(source) if str(source).isdigit() else 0
        return JetsonCSISource(sensor_id=sensor_id, **kwargs)
    else:
        return OpenCVVideoSource(source=source, source_type=source_type, **kwargs)
