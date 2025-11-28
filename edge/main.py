#!/usr/bin/env python3
"""
Jetson Edge CV Application - Main Entry Point

This application provides real-time object detection on NVIDIA Jetson devices
with AWS IoT integration for cloud connectivity.

Usage:
    python -m edge.main [OPTIONS]
    python -m edge.main --config config.yaml
    python -m edge.main --video /dev/video0 --model models/yolov8n.onnx
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .config import Settings, load_settings_from_file, get_settings
from .inference import ObjectDetector, BackendType
from .pipeline import (
    VideoSource,
    VideoSourceType,
    FrameProcessor,
    RegionOfInterest,
    ProcessingResult,
    draw_detections,
    create_video_source,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class EdgeApplication:
    """
    Main edge application orchestrating video capture, inference, and IoT publishing.
    """

    def __init__(self, settings: Settings):
        """
        Initialize edge application.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self._running = False
        self._detector: Optional[ObjectDetector] = None
        self._video_source: Optional[VideoSource] = None
        self._frame_processor: Optional[FrameProcessor] = None
        self._iot_client = None
        self._s3_uploader = None
        self._last_upload_time = 0.0

        # Statistics
        self._frame_count = 0
        self._start_time = 0.0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    def _setup_logging(self) -> None:
        """Configure logging based on settings."""
        log_level = getattr(logging, self.settings.logging.level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)

        if self.settings.logging.file:
            file_handler = logging.FileHandler(self.settings.logging.file)
            file_handler.setFormatter(
                logging.Formatter(self.settings.logging.format)
            )
            logging.getLogger().addHandler(file_handler)

    def _initialize_detector(self) -> None:
        """Initialize object detector."""
        inf_settings = self.settings.inference

        # Determine backend
        backend_map = {
            "auto": BackendType.AUTO,
            "tensorrt": BackendType.TENSORRT,
            "onnx": BackendType.ONNX,
            "tflite": BackendType.TFLITE,
        }
        backend = backend_map.get(inf_settings.backend.lower(), BackendType.AUTO)

        self._detector = ObjectDetector(
            model_path=inf_settings.model_path,
            backend=backend,
            confidence_threshold=inf_settings.confidence_threshold,
            nms_threshold=inf_settings.nms_threshold,
            target_classes=self.settings.detection.target_classes,
            input_size=tuple(inf_settings.input_size),
            device=inf_settings.device,
        )

        logger.info(f"Loading model: {inf_settings.model_path}")
        self._detector.load()

        if inf_settings.warmup_iterations > 0:
            self._detector.warmup(inf_settings.warmup_iterations)

    def _initialize_video(self) -> None:
        """Initialize video source."""
        vid_settings = self.settings.video

        # Determine source type
        source_type_map = {
            "auto": None,  # Auto-detect
            "file": VideoSourceType.FILE,
            "camera": VideoSourceType.CAMERA,
            "rtsp": VideoSourceType.RTSP,
            "csi": VideoSourceType.CSI,
            "usb": VideoSourceType.USB,
        }
        source_type = source_type_map.get(vid_settings.source_type.lower())

        self._video_source = create_video_source(
            source=vid_settings.source,
            source_type=source_type,
            width=vid_settings.width,
            height=vid_settings.height,
            fps=vid_settings.fps,
            flip_method=vid_settings.flip_method,
        )

        if not self._video_source.open():
            raise RuntimeError(f"Failed to open video source: {vid_settings.source}")

        logger.info(f"Video source opened: {vid_settings.source}")

    def _initialize_frame_processor(self) -> None:
        """Initialize frame processor."""
        det_settings = self.settings.detection

        roi = RegionOfInterest(
            name=det_settings.roi_name,
            x1=det_settings.roi_x1,
            y1=det_settings.roi_y1,
            x2=det_settings.roi_x2,
            y2=det_settings.roi_y2,
        )

        self._frame_processor = FrameProcessor(
            detector=self._detector,
            roi=roi,
            target_classes=det_settings.target_classes,
            iou_threshold=det_settings.iou_threshold,
            confidence_threshold=self.settings.inference.confidence_threshold,
            state_persistence_frames=det_settings.state_persistence_frames,
        )

        logger.info(f"Frame processor initialized with ROI: {det_settings.roi_name}")

    def _initialize_iot(self) -> None:
        """Initialize IoT client if configured."""
        aws_settings = self.settings.aws

        if not aws_settings.iot_endpoint:
            logger.warning("IoT endpoint not configured, skipping IoT initialization")
            return

        try:
            from .iot import IoTClient, MQTTConfig

            config = MQTTConfig(
                endpoint=aws_settings.iot_endpoint,
                cert_path=aws_settings.cert_path,
                key_path=aws_settings.key_path,
                root_ca_path=aws_settings.root_ca_path,
                client_id=aws_settings.iot_thing_name,
                region=aws_settings.region,
            )

            self._iot_client = IoTClient(config)

            if self._iot_client.connect():
                logger.info("Connected to AWS IoT Core")
            else:
                logger.warning("Failed to connect to AWS IoT Core")
                self._iot_client = None

        except Exception as e:
            logger.warning(f"IoT initialization failed: {e}")
            self._iot_client = None

    def _initialize_s3(self) -> None:
        """Initialize S3 uploader if configured."""
        aws_settings = self.settings.aws

        if not aws_settings.s3_bucket or not aws_settings.upload_images:
            return

        try:
            from .iot import S3Uploader

            self._s3_uploader = S3Uploader(
                bucket=aws_settings.s3_bucket,
                region=aws_settings.region,
                key_prefix=aws_settings.s3_prefix,
            )
            self._s3_uploader.start()
            logger.info(f"S3 uploader started (bucket: {aws_settings.s3_bucket})")

        except Exception as e:
            logger.warning(f"S3 uploader initialization failed: {e}")
            self._s3_uploader = None

    def _publish_result(self, result: ProcessingResult, frame: np.ndarray) -> None:
        """Publish detection result to IoT."""
        if self._iot_client is None or not self._iot_client.is_connected:
            return

        payload = result.to_dict()
        payload["device_id"] = self.settings.aws.iot_thing_name

        # Publish to main topic
        topic = self.settings.aws.iot_topic
        self._iot_client.publish(topic, payload)

        # Publish state change to logs topic if state changed
        if result.state_changed:
            logs_topic = f"{topic}/logs"
            log_payload = {
                "event": "state_change",
                "timestamp": result.timestamp,
                "previous_state": result.previous_state.value if result.previous_state else None,
                "new_state": result.occupancy_state.value,
                "confidence": result.confidence,
                "iou": result.iou,
            }
            self._iot_client.publish(logs_topic, log_payload)

    def _upload_frame(self, frame: np.ndarray, result: ProcessingResult) -> None:
        """Upload frame to S3 if configured."""
        if self._s3_uploader is None:
            return

        # Check upload interval
        current_time = time.time()
        if current_time - self._last_upload_time < self.settings.aws.upload_interval:
            return

        metadata = {
            "occupancy_state": result.occupancy_state.value,
            "confidence": str(round(result.confidence, 4)),
            "iou": str(round(result.iou, 4)),
        }

        self._s3_uploader.queue_upload(
            frame=frame,
            frame_id=result.frame_id,
            timestamp=result.timestamp,
            metadata=metadata,
        )

        self._last_upload_time = current_time

    def initialize(self) -> None:
        """Initialize all components."""
        logger.info(f"Initializing {self.settings.app_name} v{self.settings.app_version}")

        self._setup_logging()
        self._initialize_detector()
        self._initialize_video()
        self._initialize_frame_processor()
        self._initialize_iot()
        self._initialize_s3()

        logger.info("Initialization complete")

    def run(self) -> None:
        """Run the main processing loop."""
        self._running = True
        self._start_time = time.time()
        self._frame_count = 0

        # Optional display window
        display_enabled = self.settings.output.enable_display
        if display_enabled:
            try:
                import cv2
                cv2.namedWindow("Edge CV", cv2.WINDOW_NORMAL)
            except ImportError:
                logger.warning("OpenCV not available for display")
                display_enabled = False

        logger.info("Starting main processing loop")

        try:
            while self._running:
                # Read frame
                ret, frame, frame_info = self._video_source.read()
                if not ret:
                    if self.settings.video.source_type == "file":
                        logger.info("End of video file")
                        break
                    continue

                self._frame_count += 1

                # Process frame
                result = self._frame_processor.process_frame(frame)

                # Draw detections if enabled
                if self.settings.output.draw_detections:
                    frame = draw_detections(
                        frame,
                        result,
                        draw_roi=self.settings.output.draw_roi,
                    )

                # Publish to IoT
                self._publish_result(result, frame)

                # Upload frame to S3
                self._upload_frame(frame, result)

                # Display if enabled
                if display_enabled:
                    import cv2
                    cv2.imshow("Edge CV", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Log metrics periodically
                if self.settings.logging.enable_metrics and self._frame_count % 100 == 0:
                    elapsed = time.time() - self._start_time
                    fps = self._frame_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Processed {self._frame_count} frames, "
                        f"FPS: {fps:.1f}, "
                        f"Inference: {result.inference_time_ms:.1f}ms, "
                        f"State: {result.occupancy_state.value}"
                    )

        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Shutdown all components."""
        logger.info("Shutting down...")

        self._running = False

        if self._video_source is not None:
            self._video_source.close()

        if self._iot_client is not None:
            self._iot_client.disconnect()

        if self._s3_uploader is not None:
            self._s3_uploader.stop()

        if self.settings.output.enable_display:
            try:
                import cv2
                cv2.destroyAllWindows()
            except ImportError:
                pass

        # Log final statistics
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        fps = self._frame_count / elapsed if elapsed > 0 else 0
        logger.info(
            f"Shutdown complete. Processed {self._frame_count} frames "
            f"in {elapsed:.1f}s ({fps:.1f} FPS)"
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Jetson Edge CV Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML, JSON, or TOML)",
    )

    parser.add_argument(
        "--video", "-v",
        type=str,
        help="Video source (file path, camera index, or RTSP URL)",
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Path to model file (.onnx, .engine, or .tflite)",
    )

    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["auto", "tensorrt", "onnx", "tflite"],
        help="Inference backend",
    )

    parser.add_argument(
        "--confidence", "-t",
        type=float,
        help="Detection confidence threshold (0-1)",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Enable preview window",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--create-config",
        type=str,
        metavar="PATH",
        help="Create default configuration file and exit",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create default config if requested
    if args.create_config:
        from .config import create_default_config
        create_default_config(args.create_config)
        print(f"Default configuration created at: {args.create_config}")
        sys.exit(0)

    # Load settings
    if args.config:
        settings = load_settings_from_file(args.config)
    else:
        settings = get_settings()

    # Override settings from command line
    if args.video:
        settings.video.source = args.video
    if args.model:
        settings.inference.model_path = args.model
    if args.backend:
        settings.inference.backend = args.backend
    if args.confidence:
        settings.inference.confidence_threshold = args.confidence
    if args.display:
        settings.output.enable_display = True
    if args.debug:
        settings.debug = True
        settings.logging.level = "DEBUG"

    # Run application
    try:
        app = EdgeApplication(settings)
        app.initialize()
        app.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
