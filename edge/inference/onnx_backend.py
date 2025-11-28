"""
ONNX Runtime inference backend.
Provides cross-platform inference with GPU acceleration support.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

# ONNX Runtime imports with graceful fallback
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available. Install with: pip install onnxruntime-gpu")


class ONNXBackend:
    """
    ONNX Runtime inference backend.

    Supports:
    - CPU and GPU (CUDA, TensorRT) execution providers
    - FP32 and FP16 precision
    - Cross-platform compatibility
    """

    def __init__(self, device: str = "cuda:0"):
        """
        Initialize ONNX Runtime backend.

        Args:
            device: Device to use ("cuda:0", "cpu", etc.)
        """
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "ONNX Runtime not available. Install with:\n"
                "pip install onnxruntime-gpu  # For GPU support\n"
                "pip install onnxruntime      # For CPU only"
            )

        self.device = device
        self.session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._output_names: Optional[List[str]] = None

        # Configure execution providers
        self.providers = self._get_providers()
        logger.info(f"ONNX Runtime providers: {self.providers}")

    def _get_providers(self) -> List[str]:
        """Get appropriate execution providers based on device."""
        available = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available}")

        if "cuda" in self.device.lower():
            providers = []
            # Prefer TensorRT EP for Jetson
            if "TensorrtExecutionProvider" in available:
                providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            return providers
        else:
            return ["CPUExecutionProvider"]

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.session is not None

    def load_model(self, model_path: Path) -> None:
        """
        Load ONNX model from file.

        Args:
            model_path: Path to .onnx file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading ONNX model from {model_path}")

        # Configure session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Enable memory optimization
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True

        # Create inference session
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=self.providers,
        )

        # Get input/output info
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        self._input_name = inputs[0].name
        self._input_shape = tuple(inputs[0].shape)
        self._output_names = [o.name for o in outputs]

        # Log actual provider being used
        actual_providers = self.session.get_providers()
        logger.info(f"ONNX model loaded, using providers: {actual_providers}")
        logger.info(f"Input: {self._input_name}, shape: {self._input_shape}")
        logger.info(f"Outputs: {self._output_names}")

    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Get expected input shape (batch, channels, height, width)."""
        if self._input_shape is None:
            raise RuntimeError("Model not loaded")

        # Handle dynamic dimensions (often marked as -1 or string)
        shape = []
        for dim in self._input_shape:
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                # Default to 1 for batch, 640 for spatial dimensions
                shape.append(1 if len(shape) == 0 else 640)

        return tuple(shape)

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed image.

        Args:
            image: Preprocessed image array (NCHW format, float32)

        Returns:
            Raw model outputs
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Run inference
        outputs = self.session.run(
            self._output_names,
            {self._input_name: image}
        )

        # Return first output (typically the detection output)
        return outputs[0]

    def warmup(self, iterations: int = 10) -> None:
        """Warm up the model for consistent performance."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        logger.info(f"Warming up ONNX model with {iterations} iterations")

        # Get actual input shape
        shape = list(self.get_input_shape())
        # Ensure batch size is 1
        shape[0] = 1

        # Create dummy input
        dummy_input = np.random.randn(*shape).astype(np.float32)

        for i in range(iterations):
            _ = self.infer(dummy_input)

        logger.info("ONNX warmup complete")

    def get_metadata(self) -> dict:
        """Get model metadata."""
        if not self.is_loaded:
            return {}

        metadata = {}
        try:
            meta = self.session.get_modelmeta()
            metadata = {
                "producer_name": meta.producer_name,
                "graph_name": meta.graph_name,
                "domain": meta.domain,
                "description": meta.description,
                "version": meta.version,
            }
        except Exception as e:
            logger.warning(f"Could not get model metadata: {e}")

        return metadata
