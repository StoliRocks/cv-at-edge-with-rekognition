"""
GStreamer pipeline builder for optimized video processing on Jetson.
Provides hardware-accelerated encoding/decoding using NVIDIA NVENC/NVDEC.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CodecType(Enum):
    """Supported video codecs."""
    H264 = "h264"
    H265 = "h265"
    VP8 = "vp8"
    VP9 = "vp9"
    MJPEG = "mjpeg"


class StreamProtocol(Enum):
    """Supported streaming protocols."""
    RTSP = "rtsp"
    RTMP = "rtmp"
    HLS = "hls"
    UDP = "udp"
    FILE = "file"


@dataclass
class PipelineConfig:
    """Configuration for GStreamer pipeline."""
    width: int = 1280
    height: int = 720
    fps: int = 30
    codec: CodecType = CodecType.H264
    bitrate: int = 4000000  # 4 Mbps
    use_hardware_accel: bool = True
    flip_method: int = 0  # 0=none, 2=180, etc.


class GStreamerPipeline:
    """
    GStreamer pipeline builder for NVIDIA Jetson.
    Optimizes for hardware acceleration and low latency.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize GStreamer pipeline builder.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

    def build_csi_capture_pipeline(self, sensor_id: int = 0) -> str:
        """
        Build capture pipeline for Jetson CSI camera.

        Args:
            sensor_id: CSI camera sensor ID

        Returns:
            GStreamer pipeline string
        """
        c = self.config
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), "
            f"width={c.width}, height={c.height}, "
            f"format=NV12, framerate={c.fps}/1 ! "
            f"nvvidconv flip-method={c.flip_method} ! "
            f"video/x-raw, width={c.width}, height={c.height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink drop=1 max-buffers=1"
        )

    def build_usb_capture_pipeline(self, device: str = "/dev/video0") -> str:
        """
        Build capture pipeline for USB camera.

        Args:
            device: Video device path

        Returns:
            GStreamer pipeline string
        """
        c = self.config
        return (
            f"v4l2src device={device} ! "
            f"video/x-raw, width={c.width}, height={c.height}, framerate={c.fps}/1 ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink drop=1 max-buffers=1"
        )

    def build_rtsp_capture_pipeline(self, url: str, latency: int = 100) -> str:
        """
        Build capture pipeline for RTSP stream.

        Args:
            url: RTSP stream URL
            latency: Buffer latency in milliseconds

        Returns:
            GStreamer pipeline string
        """
        c = self.config

        if c.use_hardware_accel:
            # Hardware-accelerated decode on Jetson
            return (
                f"rtspsrc location={url} latency={latency} ! "
                f"rtph264depay ! h264parse ! "
                f"nvv4l2decoder ! "
                f"nvvidconv ! "
                f"video/x-raw, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink drop=1 max-buffers=1"
            )
        else:
            # Software decode
            return (
                f"rtspsrc location={url} latency={latency} ! "
                f"rtph264depay ! h264parse ! "
                f"avdec_h264 ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink drop=1 max-buffers=1"
            )

    def build_file_source_pipeline(self, filepath: str, loop: bool = False) -> str:
        """
        Build pipeline for video file playback.

        Args:
            filepath: Path to video file
            loop: Loop the video

        Returns:
            GStreamer pipeline string
        """
        c = self.config

        if c.use_hardware_accel:
            # Hardware decode
            pipeline = (
                f"filesrc location={filepath} ! "
                f"qtdemux ! h264parse ! "
                f"nvv4l2decoder ! "
                f"nvvidconv ! "
                f"video/x-raw, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink drop=0"
            )
        else:
            # Software decode
            pipeline = (
                f"filesrc location={filepath} ! "
                f"decodebin ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink drop=0"
            )

        return pipeline

    def build_rtsp_output_pipeline(
        self,
        host: str = "0.0.0.0",
        port: int = 8554,
        path: str = "/stream",
    ) -> str:
        """
        Build RTSP output pipeline for streaming.

        Args:
            host: Server host
            port: Server port
            path: Stream path

        Returns:
            GStreamer pipeline string
        """
        c = self.config

        if c.use_hardware_accel:
            encoder = f"nvv4l2h264enc bitrate={c.bitrate} preset-level=1"
        else:
            encoder = f"x264enc tune=zerolatency bitrate={c.bitrate // 1000}"

        return (
            f"appsrc ! "
            f"videoconvert ! "
            f"video/x-raw, format=I420, width={c.width}, height={c.height}, framerate={c.fps}/1 ! "
            f"{encoder} ! "
            f"h264parse ! "
            f"rtspclientsink location=rtsp://{host}:{port}{path}"
        )

    def build_file_output_pipeline(self, filepath: str) -> str:
        """
        Build pipeline for recording to file.

        Args:
            filepath: Output file path

        Returns:
            GStreamer pipeline string
        """
        c = self.config

        if c.use_hardware_accel:
            encoder = f"nvv4l2h264enc bitrate={c.bitrate}"
        else:
            encoder = f"x264enc tune=zerolatency bitrate={c.bitrate // 1000}"

        return (
            f"appsrc ! "
            f"videoconvert ! "
            f"video/x-raw, format=I420, width={c.width}, height={c.height}, framerate={c.fps}/1 ! "
            f"{encoder} ! "
            f"h264parse ! "
            f"mp4mux ! "
            f"filesink location={filepath}"
        )

    def build_udp_output_pipeline(self, host: str, port: int) -> str:
        """
        Build UDP streaming pipeline.

        Args:
            host: Target host
            port: Target port

        Returns:
            GStreamer pipeline string
        """
        c = self.config

        if c.use_hardware_accel:
            encoder = f"nvv4l2h264enc bitrate={c.bitrate}"
        else:
            encoder = f"x264enc tune=zerolatency bitrate={c.bitrate // 1000}"

        return (
            f"appsrc ! "
            f"videoconvert ! "
            f"video/x-raw, format=I420, width={c.width}, height={c.height}, framerate={c.fps}/1 ! "
            f"{encoder} ! "
            f"h264parse ! "
            f"rtph264pay ! "
            f"udpsink host={host} port={port}"
        )

    def build_display_pipeline(self, window_name: str = "Preview") -> str:
        """
        Build pipeline for local display.

        Args:
            window_name: Window title

        Returns:
            GStreamer pipeline string
        """
        c = self.config

        if c.use_hardware_accel:
            # Use nvoverlaysink for Jetson display
            return (
                f"appsrc ! "
                f"nvvidconv ! "
                f"nvoverlaysink"
            )
        else:
            return (
                f"appsrc ! "
                f"videoconvert ! "
                f"autovideosink"
            )


def check_gstreamer_available() -> bool:
    """Check if GStreamer is available."""
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        return True
    except (ImportError, ValueError) as e:
        logger.warning(f"GStreamer not available: {e}")
        return False


def check_nvidia_plugins_available() -> bool:
    """Check if NVIDIA GStreamer plugins are available."""
    try:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)

        # Check for nvvidconv plugin
        factory = Gst.ElementFactory.find("nvvidconv")
        return factory is not None
    except Exception as e:
        logger.warning(f"NVIDIA plugins not available: {e}")
        return False


def get_available_cameras() -> list:
    """Get list of available video devices."""
    cameras = []

    # Check V4L2 devices
    import os
    for i in range(10):
        device = f"/dev/video{i}"
        if os.path.exists(device):
            cameras.append({
                "path": device,
                "type": "v4l2",
                "index": i,
            })

    # Check for CSI cameras (Jetson)
    if check_nvidia_plugins_available():
        # CSI cameras are typically sensor 0 and 1
        cameras.append({"path": "csi://0", "type": "csi", "index": 0})

    return cameras
