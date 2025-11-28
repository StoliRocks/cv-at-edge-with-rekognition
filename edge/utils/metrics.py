"""
Metrics collection and reporting for edge application.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    frame_id: int
    timestamp: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float
    detections_count: int
    occupancy_state: str
    confidence: float
    iou: float


@dataclass
class AggregatedMetrics:
    """Aggregated performance metrics."""
    total_frames: int
    elapsed_time_s: float
    avg_fps: float
    avg_inference_ms: float
    avg_total_ms: float
    min_inference_ms: float
    max_inference_ms: float
    p95_inference_ms: float
    p99_inference_ms: float
    state_distribution: Dict[str, int]


class MetricsCollector:
    """
    Collects and aggregates performance metrics.
    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        window_size: int = 1000,
        report_interval: float = 60.0,
    ):
        """
        Initialize metrics collector.

        Args:
            window_size: Number of recent frames to keep for statistics
            report_interval: Seconds between periodic reports
        """
        self.window_size = window_size
        self.report_interval = report_interval

        self._metrics: Deque[FrameMetrics] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._total_frames = 0
        self._state_counts: Dict[str, int] = {}
        self._last_report_time = time.time()

    def record(
        self,
        frame_id: int,
        preprocess_ms: float,
        inference_ms: float,
        postprocess_ms: float,
        detections_count: int,
        occupancy_state: str,
        confidence: float,
        iou: float,
    ) -> None:
        """
        Record metrics for a processed frame.

        Args:
            frame_id: Frame identifier
            preprocess_ms: Preprocessing time in milliseconds
            inference_ms: Inference time in milliseconds
            postprocess_ms: Postprocessing time in milliseconds
            detections_count: Number of detections
            occupancy_state: Current occupancy state
            confidence: Detection confidence
            iou: Intersection over Union
        """
        total_ms = preprocess_ms + inference_ms + postprocess_ms

        metrics = FrameMetrics(
            frame_id=frame_id,
            timestamp=time.time(),
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
            detections_count=detections_count,
            occupancy_state=occupancy_state,
            confidence=confidence,
            iou=iou,
        )

        with self._lock:
            self._metrics.append(metrics)
            self._total_frames += 1
            self._state_counts[occupancy_state] = (
                self._state_counts.get(occupancy_state, 0) + 1
            )

        # Periodic report
        if time.time() - self._last_report_time >= self.report_interval:
            self._periodic_report()

    def _periodic_report(self) -> None:
        """Log periodic metrics report."""
        aggregated = self.get_aggregated_metrics()
        if aggregated is None:
            return

        logger.info(
            f"Metrics Report: "
            f"frames={aggregated.total_frames}, "
            f"fps={aggregated.avg_fps:.1f}, "
            f"inference_ms={aggregated.avg_inference_ms:.1f} "
            f"(p95={aggregated.p95_inference_ms:.1f}, p99={aggregated.p99_inference_ms:.1f})"
        )

        self._last_report_time = time.time()

    def get_aggregated_metrics(self) -> Optional[AggregatedMetrics]:
        """
        Get aggregated metrics from recent frames.

        Returns:
            Aggregated metrics or None if no data
        """
        with self._lock:
            if not self._metrics:
                return None

            metrics_list = list(self._metrics)

        elapsed_time = time.time() - self._start_time
        total_frames = self._total_frames

        # Calculate statistics
        inference_times = [m.inference_ms for m in metrics_list]
        total_times = [m.total_ms for m in metrics_list]

        inference_times_sorted = sorted(inference_times)
        n = len(inference_times_sorted)

        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        return AggregatedMetrics(
            total_frames=total_frames,
            elapsed_time_s=elapsed_time,
            avg_fps=total_frames / elapsed_time if elapsed_time > 0 else 0,
            avg_inference_ms=sum(inference_times) / n if n > 0 else 0,
            avg_total_ms=sum(total_times) / n if n > 0 else 0,
            min_inference_ms=min(inference_times) if inference_times else 0,
            max_inference_ms=max(inference_times) if inference_times else 0,
            p95_inference_ms=inference_times_sorted[p95_idx] if p95_idx < n else 0,
            p99_inference_ms=inference_times_sorted[p99_idx] if p99_idx < n else 0,
            state_distribution=dict(self._state_counts),
        )

    def get_recent_fps(self, window_seconds: float = 5.0) -> float:
        """
        Calculate FPS over recent time window.

        Args:
            window_seconds: Time window in seconds

        Returns:
            Frames per second
        """
        with self._lock:
            if not self._metrics:
                return 0.0

            current_time = time.time()
            cutoff_time = current_time - window_seconds

            recent_count = sum(
                1 for m in self._metrics if m.timestamp >= cutoff_time
            )

        return recent_count / window_seconds if window_seconds > 0 else 0.0

    def get_latency_histogram(self, bins: int = 10) -> Dict[str, List]:
        """
        Get histogram of inference latencies.

        Args:
            bins: Number of histogram bins

        Returns:
            Dictionary with bin edges and counts
        """
        with self._lock:
            if not self._metrics:
                return {"edges": [], "counts": []}

            inference_times = [m.inference_ms for m in self._metrics]

        if not inference_times:
            return {"edges": [], "counts": []}

        min_val = min(inference_times)
        max_val = max(inference_times)
        bin_width = (max_val - min_val) / bins if max_val > min_val else 1

        edges = [min_val + i * bin_width for i in range(bins + 1)]
        counts = [0] * bins

        for t in inference_times:
            bin_idx = min(int((t - min_val) / bin_width), bins - 1)
            counts[bin_idx] += 1

        return {"edges": edges, "counts": counts}

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics.clear()
            self._total_frames = 0
            self._state_counts.clear()
            self._start_time = time.time()
            self._last_report_time = time.time()

    def to_dict(self) -> Dict:
        """Export metrics as dictionary."""
        aggregated = self.get_aggregated_metrics()
        if aggregated is None:
            return {}

        return {
            "total_frames": aggregated.total_frames,
            "elapsed_time_s": aggregated.elapsed_time_s,
            "avg_fps": aggregated.avg_fps,
            "latency": {
                "avg_inference_ms": aggregated.avg_inference_ms,
                "avg_total_ms": aggregated.avg_total_ms,
                "min_inference_ms": aggregated.min_inference_ms,
                "max_inference_ms": aggregated.max_inference_ms,
                "p95_inference_ms": aggregated.p95_inference_ms,
                "p99_inference_ms": aggregated.p99_inference_ms,
            },
            "state_distribution": aggregated.state_distribution,
            "recent_fps": self.get_recent_fps(),
        }
