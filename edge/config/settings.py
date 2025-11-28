"""
Settings management using Pydantic for validation.
Supports loading from environment variables, YAML, JSON, and TOML files.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import pydantic, fall back to dataclasses if not available
try:
    from pydantic import BaseModel, Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning("Pydantic not available. Using basic dataclass settings.")


if PYDANTIC_AVAILABLE:

    class VideoSettings(BaseModel):
        """Video source configuration."""
        source: str = Field(default="0", description="Video source (file path, camera index, or RTSP URL)")
        source_type: str = Field(default="auto", description="Source type: auto, file, camera, rtsp, csi, usb")
        width: int = Field(default=1280, ge=320, le=4096, description="Frame width")
        height: int = Field(default=720, ge=240, le=2160, description="Frame height")
        fps: int = Field(default=30, ge=1, le=120, description="Target frame rate")
        flip_method: int = Field(default=0, ge=0, le=7, description="CSI camera flip method")
        buffer_size: int = Field(default=2, ge=1, le=10, description="Frame buffer size")

    class InferenceSettings(BaseModel):
        """Inference engine configuration."""
        model_path: str = Field(default="models/yolov8n.onnx", description="Path to model file")
        backend: str = Field(default="auto", description="Backend: auto, tensorrt, onnx, tflite")
        device: str = Field(default="cuda:0", description="Inference device")
        confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection confidence threshold")
        nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="NMS threshold")
        input_size: List[int] = Field(default=[640, 640], description="Model input size [width, height]")
        warmup_iterations: int = Field(default=10, ge=0, description="Warmup iterations")

        @field_validator("input_size")
        @classmethod
        def validate_input_size(cls, v):
            if len(v) != 2:
                raise ValueError("input_size must be [width, height]")
            return v

    class DetectionSettings(BaseModel):
        """Detection and analytics configuration."""
        target_classes: List[str] = Field(default=["truck"], description="Classes to detect")
        iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="IoU threshold for ROI overlap")
        state_persistence_frames: int = Field(default=3, ge=1, description="Frames before state change")
        roi_x1: float = Field(default=0.1, ge=0.0, le=1.0, description="ROI left boundary (normalized)")
        roi_y1: float = Field(default=0.1, ge=0.0, le=1.0, description="ROI top boundary (normalized)")
        roi_x2: float = Field(default=0.85, ge=0.0, le=1.0, description="ROI right boundary (normalized)")
        roi_y2: float = Field(default=0.9, ge=0.0, le=1.0, description="ROI bottom boundary (normalized)")
        roi_name: str = Field(default="garage_bay_01", description="ROI identifier")

    class AWSSettings(BaseModel):
        """AWS service configuration."""
        region: str = Field(default="us-east-1", description="AWS region")
        iot_endpoint: str = Field(default="", description="AWS IoT endpoint")
        iot_thing_name: str = Field(default="JetsonEdgeDevice", description="IoT thing name")
        iot_topic: str = Field(default="garage/001/bay/01", description="MQTT topic for publishing")
        iot_shadow_topic: str = Field(default="", description="IoT shadow topic (auto-generated if empty)")
        s3_bucket: str = Field(default="", description="S3 bucket for image uploads")
        s3_prefix: str = Field(default="frames/", description="S3 key prefix")
        cert_path: str = Field(default="certs/device.cert.pem", description="Path to device certificate")
        key_path: str = Field(default="certs/device.private.key", description="Path to private key")
        root_ca_path: str = Field(default="certs/AmazonRootCA1.pem", description="Path to root CA certificate")
        upload_images: bool = Field(default=False, description="Upload frames to S3")
        upload_interval: int = Field(default=5, ge=1, description="Seconds between S3 uploads")

    class OutputSettings(BaseModel):
        """Output and streaming configuration."""
        enable_display: bool = Field(default=False, description="Show preview window")
        enable_rtsp: bool = Field(default=False, description="Enable RTSP output stream")
        rtsp_port: int = Field(default=8554, ge=1024, le=65535, description="RTSP server port")
        rtsp_path: str = Field(default="/stream", description="RTSP stream path")
        save_video: bool = Field(default=False, description="Save output to file")
        output_path: str = Field(default="output/", description="Output directory")
        draw_detections: bool = Field(default=True, description="Draw detections on frame")
        draw_roi: bool = Field(default=True, description="Draw ROI on frame")

    class LoggingSettings(BaseModel):
        """Logging configuration."""
        level: str = Field(default="INFO", description="Log level")
        format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file: Optional[str] = Field(default=None, description="Log file path")
        enable_metrics: bool = Field(default=True, description="Log performance metrics")

    class Settings(BaseSettings):
        """
        Main application settings.
        Loads from environment variables with EDGE_ prefix.
        """
        model_config = SettingsConfigDict(
            env_prefix="EDGE_",
            env_nested_delimiter="__",
            case_sensitive=False,
            extra="ignore",
        )

        # Application info
        app_name: str = Field(default="jetson-edge-cv", description="Application name")
        app_version: str = Field(default="1.0.0", description="Application version")
        debug: bool = Field(default=False, description="Enable debug mode")

        # Nested settings
        video: VideoSettings = Field(default_factory=VideoSettings)
        inference: InferenceSettings = Field(default_factory=InferenceSettings)
        detection: DetectionSettings = Field(default_factory=DetectionSettings)
        aws: AWSSettings = Field(default_factory=AWSSettings)
        output: OutputSettings = Field(default_factory=OutputSettings)
        logging: LoggingSettings = Field(default_factory=LoggingSettings)

        def get_iot_shadow_topic(self) -> str:
            """Get IoT shadow topic, auto-generating if not set."""
            if self.aws.iot_shadow_topic:
                return self.aws.iot_shadow_topic
            return f"$aws/things/{self.aws.iot_thing_name}/shadow/update"

else:
    # Fallback to dataclasses if pydantic is not available
    from dataclasses import dataclass, field

    @dataclass
    class VideoSettings:
        source: str = "0"
        source_type: str = "auto"
        width: int = 1280
        height: int = 720
        fps: int = 30
        flip_method: int = 0
        buffer_size: int = 2

    @dataclass
    class InferenceSettings:
        model_path: str = "models/yolov8n.onnx"
        backend: str = "auto"
        device: str = "cuda:0"
        confidence_threshold: float = 0.5
        nms_threshold: float = 0.45
        input_size: List[int] = field(default_factory=lambda: [640, 640])
        warmup_iterations: int = 10

    @dataclass
    class DetectionSettings:
        target_classes: List[str] = field(default_factory=lambda: ["truck"])
        iou_threshold: float = 0.3
        state_persistence_frames: int = 3
        roi_x1: float = 0.1
        roi_y1: float = 0.1
        roi_x2: float = 0.85
        roi_y2: float = 0.9
        roi_name: str = "garage_bay_01"

    @dataclass
    class AWSSettings:
        region: str = "us-east-1"
        iot_endpoint: str = ""
        iot_thing_name: str = "JetsonEdgeDevice"
        iot_topic: str = "garage/001/bay/01"
        iot_shadow_topic: str = ""
        s3_bucket: str = ""
        s3_prefix: str = "frames/"
        cert_path: str = "certs/device.cert.pem"
        key_path: str = "certs/device.private.key"
        root_ca_path: str = "certs/AmazonRootCA1.pem"
        upload_images: bool = False
        upload_interval: int = 5

    @dataclass
    class OutputSettings:
        enable_display: bool = False
        enable_rtsp: bool = False
        rtsp_port: int = 8554
        rtsp_path: str = "/stream"
        save_video: bool = False
        output_path: str = "output/"
        draw_detections: bool = True
        draw_roi: bool = True

    @dataclass
    class LoggingSettings:
        level: str = "INFO"
        format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file: Optional[str] = None
        enable_metrics: bool = True

    @dataclass
    class Settings:
        app_name: str = "jetson-edge-cv"
        app_version: str = "1.0.0"
        debug: bool = False
        video: VideoSettings = field(default_factory=VideoSettings)
        inference: InferenceSettings = field(default_factory=InferenceSettings)
        detection: DetectionSettings = field(default_factory=DetectionSettings)
        aws: AWSSettings = field(default_factory=AWSSettings)
        output: OutputSettings = field(default_factory=OutputSettings)
        logging: LoggingSettings = field(default_factory=LoggingSettings)

        def get_iot_shadow_topic(self) -> str:
            if self.aws.iot_shadow_topic:
                return self.aws.iot_shadow_topic
            return f"$aws/things/{self.aws.iot_thing_name}/shadow/update"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def load_settings_from_file(
    filepath: Union[str, Path],
    override_env: bool = True,
) -> Settings:
    """
    Load settings from a configuration file.

    Args:
        filepath: Path to config file (YAML, JSON, or TOML)
        override_env: Allow environment variables to override file settings

    Returns:
        Settings instance
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.warning(f"Config file not found: {filepath}, using defaults")
        return Settings()

    suffix = filepath.suffix.lower()

    try:
        if suffix in [".yaml", ".yml"]:
            import yaml
            with open(filepath) as f:
                data = yaml.safe_load(f)
        elif suffix == ".json":
            import json
            with open(filepath) as f:
                data = json.load(f)
        elif suffix == ".toml":
            import tomllib
            with open(filepath, "rb") as f:
                data = tomllib.load(f)
        else:
            logger.error(f"Unsupported config format: {suffix}")
            return Settings()

        logger.info(f"Loaded configuration from {filepath}")
        return Settings(**data)

    except Exception as e:
        logger.error(f"Failed to load config from {filepath}: {e}")
        return Settings()


def create_default_config(filepath: Union[str, Path]) -> None:
    """
    Create a default configuration file.

    Args:
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    settings = Settings()

    suffix = filepath.suffix.lower()

    if PYDANTIC_AVAILABLE:
        data = settings.model_dump()
    else:
        import dataclasses
        data = dataclasses.asdict(settings)

    try:
        if suffix in [".yaml", ".yml"]:
            import yaml
            with open(filepath, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif suffix == ".json":
            import json
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        else:
            # Default to YAML
            import yaml
            filepath = filepath.with_suffix(".yaml")
            with open(filepath, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Created default config at {filepath}")

    except Exception as e:
        logger.error(f"Failed to create config file: {e}")
