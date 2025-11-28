"""
S3 uploader for edge frame uploads.
Supports asynchronous uploads with retry logic.
"""

import io
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Optional, Tuple
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# Try to import boto3
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, EndpointConnectionError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available. Install with: pip install boto3")


@dataclass
class UploadTask:
    """Task for async upload."""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    metadata: dict
    key_prefix: str = ""


class S3Uploader:
    """
    Asynchronous S3 uploader for video frames.
    Uses background thread for non-blocking uploads.
    """

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        key_prefix: str = "frames/",
        max_queue_size: int = 10,
        upload_format: str = "jpg",
        jpeg_quality: int = 85,
        max_retries: int = 3,
    ):
        """
        Initialize S3 uploader.

        Args:
            bucket: S3 bucket name
            region: AWS region
            key_prefix: Prefix for S3 keys
            max_queue_size: Maximum queued uploads
            upload_format: Image format (jpg, png)
            jpeg_quality: JPEG quality (1-100)
            max_retries: Maximum retry attempts
        """
        if not BOTO3_AVAILABLE:
            raise RuntimeError("boto3 not available. Install with: pip install boto3")

        self.bucket = bucket
        self.region = region
        self.key_prefix = key_prefix.rstrip("/") + "/"
        self.upload_format = upload_format.lower()
        self.jpeg_quality = jpeg_quality
        self.max_retries = max_retries

        # Initialize S3 client with retry config
        config = Config(
            retries={"max_attempts": max_retries, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=30,
        )
        self._s3_client = boto3.client("s3", region_name=region, config=config)

        # Upload queue and worker thread
        self._queue: Queue = Queue(maxsize=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # Statistics
        self._uploads_success = 0
        self._uploads_failed = 0
        self._bytes_uploaded = 0

    def start(self) -> None:
        """Start the background upload worker."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True
        self._worker_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self._worker_thread.start()
        logger.info(f"S3 uploader started (bucket: {self.bucket})")

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the background upload worker."""
        self._stop_event.set()
        self._running = False

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)
            self._worker_thread = None

        logger.info(
            f"S3 uploader stopped. "
            f"Success: {self._uploads_success}, Failed: {self._uploads_failed}, "
            f"Bytes: {self._bytes_uploaded:,}"
        )

    def _upload_worker(self) -> None:
        """Background worker for processing upload queue."""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=1.0)
                self._process_upload(task)
                self._queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Upload worker error: {e}")

    def _process_upload(self, task: UploadTask) -> bool:
        """Process a single upload task."""
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV not available for encoding")
            return False

        # Generate S3 key
        timestamp_str = datetime.fromtimestamp(task.timestamp).strftime("%Y/%m/%d/%H%M%S")
        key = f"{self.key_prefix}{timestamp_str}_{task.frame_id}.{self.upload_format}"

        if task.key_prefix:
            key = f"{task.key_prefix}/{timestamp_str}_{task.frame_id}.{self.upload_format}"

        try:
            # Encode frame
            if self.upload_format == "jpg":
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                _, buffer = cv2.imencode(".jpg", task.frame, encode_params)
                content_type = "image/jpeg"
            else:
                _, buffer = cv2.imencode(".png", task.frame)
                content_type = "image/png"

            data = buffer.tobytes()

            # Add metadata
            metadata = {
                "frame_id": str(task.frame_id),
                "timestamp": str(task.timestamp),
                **{k: str(v) for k, v in task.metadata.items()}
            }

            # Upload to S3
            self._s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
                Metadata=metadata,
            )

            self._uploads_success += 1
            self._bytes_uploaded += len(data)
            logger.debug(f"Uploaded frame to s3://{self.bucket}/{key}")
            return True

        except (ClientError, EndpointConnectionError) as e:
            self._uploads_failed += 1
            logger.error(f"S3 upload failed: {e}")
            return False

    def queue_upload(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: Optional[float] = None,
        metadata: Optional[dict] = None,
        key_prefix: str = "",
    ) -> bool:
        """
        Queue a frame for upload.

        Args:
            frame: Frame to upload
            frame_id: Frame identifier
            timestamp: Frame timestamp (default: current time)
            metadata: Additional metadata
            key_prefix: Optional key prefix override

        Returns:
            True if queued successfully
        """
        if not self._running:
            logger.warning("Uploader not running")
            return False

        task = UploadTask(
            frame=frame.copy(),  # Copy to avoid mutation
            frame_id=frame_id,
            timestamp=timestamp or time.time(),
            metadata=metadata or {},
            key_prefix=key_prefix,
        )

        try:
            self._queue.put_nowait(task)
            return True
        except Full:
            logger.warning("Upload queue full, dropping frame")
            return False

    def upload_sync(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> Tuple[bool, str]:
        """
        Upload frame synchronously.

        Args:
            frame: Frame to upload
            frame_id: Frame identifier
            timestamp: Frame timestamp
            metadata: Additional metadata

        Returns:
            Tuple of (success, s3_key)
        """
        task = UploadTask(
            frame=frame,
            frame_id=frame_id,
            timestamp=timestamp or time.time(),
            metadata=metadata or {},
        )

        success = self._process_upload(task)
        timestamp_str = datetime.fromtimestamp(task.timestamp).strftime("%Y/%m/%d/%H%M%S")
        key = f"{self.key_prefix}{timestamp_str}_{frame_id}.{self.upload_format}"

        return success, f"s3://{self.bucket}/{key}" if success else ""

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for an S3 object.

        Args:
            key: S3 object key
            expires_in: URL expiration time in seconds

        Returns:
            Presigned URL or None on error
        """
        try:
            url = self._s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in,
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None

    @property
    def queue_size(self) -> int:
        """Get current upload queue size."""
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        """Get upload statistics."""
        return {
            "uploads_success": self._uploads_success,
            "uploads_failed": self._uploads_failed,
            "bytes_uploaded": self._bytes_uploaded,
            "queue_size": self._queue.qsize(),
            "running": self._running,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
