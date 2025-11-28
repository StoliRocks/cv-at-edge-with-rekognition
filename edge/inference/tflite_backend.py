"""
TensorFlow Lite inference backend.
Provides lightweight inference for resource-constrained edge devices.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

# TensorFlow Lite imports with graceful fallback
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
    TFLITE_SOURCE = "tflite_runtime"
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
        TFLITE_SOURCE = "tensorflow"
    except ImportError:
        TFLITE_AVAILABLE = False
        TFLITE_SOURCE = None
        logger.warning(
            "TensorFlow Lite not available. Install with:\n"
            "pip install tflite-runtime  # Lightweight option\n"
            "pip install tensorflow       # Full TensorFlow"
        )


class TFLiteBackend:
    """
    TensorFlow Lite inference backend.

    Supports:
    - CPU inference with optimized kernels
    - Edge TPU acceleration (if available)
    - INT8 quantized models
    """

    def __init__(self, num_threads: int = 4, use_edge_tpu: bool = False):
        """
        Initialize TFLite backend.

        Args:
            num_threads: Number of CPU threads for inference
            use_edge_tpu: Enable Edge TPU acceleration (Coral)
        """
        if not TFLITE_AVAILABLE:
            raise RuntimeError(
                "TensorFlow Lite not available. Install with:\n"
                "pip install tflite-runtime  # Lightweight\n"
                "pip install tensorflow       # Full TensorFlow"
            )

        self.num_threads = num_threads
        self.use_edge_tpu = use_edge_tpu
        self.interpreter: Optional[tflite.Interpreter] = None
        self._input_details: Optional[List[dict]] = None
        self._output_details: Optional[List[dict]] = None

        logger.info(f"TFLite backend using: {TFLITE_SOURCE}")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.interpreter is not None

    def load_model(self, model_path: Path) -> None:
        """
        Load TFLite model from file.

        Args:
            model_path: Path to .tflite file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading TFLite model from {model_path}")

        # Create interpreter
        if self.use_edge_tpu:
            try:
                from pycoral.utils.edgetpu import make_interpreter
                self.interpreter = make_interpreter(str(model_path))
                logger.info("Using Edge TPU accelerator")
            except ImportError:
                logger.warning("Edge TPU library not available, using CPU")
                self._create_cpu_interpreter(model_path)
        else:
            self._create_cpu_interpreter(model_path)

        # Allocate tensors
        self.interpreter.allocate_tensors()

        # Get input/output details
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()

        logger.info(f"TFLite model loaded")
        logger.info(f"Input shape: {self._input_details[0]['shape']}")
        logger.info(f"Input dtype: {self._input_details[0]['dtype']}")

    def _create_cpu_interpreter(self, model_path: Path) -> None:
        """Create CPU interpreter with thread optimization."""
        self.interpreter = tflite.Interpreter(
            model_path=str(model_path),
            num_threads=self.num_threads,
        )

    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Get expected input shape (batch, height, width, channels) for TFLite."""
        if self._input_details is None:
            raise RuntimeError("Model not loaded")
        return tuple(self._input_details[0]["shape"])

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference on preprocessed image.

        Note: TFLite typically expects NHWC format (not NCHW).
        This method handles the conversion if needed.

        Args:
            image: Preprocessed image array

        Returns:
            Raw model outputs
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Get input details
        input_detail = self._input_details[0]
        input_shape = input_detail["shape"]
        input_dtype = input_detail["dtype"]

        # Check if we need to transpose from NCHW to NHWC
        if len(image.shape) == 4 and image.shape[1] == 3:
            # NCHW -> NHWC
            image = image.transpose(0, 2, 3, 1)

        # Handle quantized models
        if input_dtype == np.uint8:
            scale = input_detail.get("quantization_parameters", {}).get("scales", [1.0])[0]
            zero_point = input_detail.get("quantization_parameters", {}).get("zero_points", [0])[0]
            image = ((image / scale) + zero_point).astype(np.uint8)
        elif input_dtype == np.int8:
            scale = input_detail.get("quantization_parameters", {}).get("scales", [1.0])[0]
            zero_point = input_detail.get("quantization_parameters", {}).get("zero_points", [0])[0]
            image = ((image / scale) + zero_point).astype(np.int8)
        else:
            image = image.astype(input_dtype)

        # Set input tensor
        self.interpreter.set_tensor(input_detail["index"], image)

        # Run inference
        self.interpreter.invoke()

        # Get output
        output_detail = self._output_details[0]
        output = self.interpreter.get_tensor(output_detail["index"])

        # Dequantize output if needed
        if output_detail["dtype"] in [np.uint8, np.int8]:
            scale = output_detail.get("quantization_parameters", {}).get("scales", [1.0])[0]
            zero_point = output_detail.get("quantization_parameters", {}).get("zero_points", [0])[0]
            output = (output.astype(np.float32) - zero_point) * scale

        return output

    def warmup(self, iterations: int = 10) -> None:
        """Warm up the model for consistent performance."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        logger.info(f"Warming up TFLite model with {iterations} iterations")

        # Create dummy input matching model's expected format
        input_shape = list(self._input_details[0]["shape"])
        input_dtype = self._input_details[0]["dtype"]

        if input_dtype in [np.uint8, np.int8]:
            dummy_input = np.random.randint(0, 255, input_shape, dtype=input_dtype)
        else:
            dummy_input = np.random.randn(*input_shape).astype(np.float32)

        for i in range(iterations):
            self.interpreter.set_tensor(self._input_details[0]["index"], dummy_input)
            self.interpreter.invoke()

        logger.info("TFLite warmup complete")

    def get_tensor_details(self) -> dict:
        """Get detailed tensor information."""
        if not self.is_loaded:
            return {}

        return {
            "inputs": self._input_details,
            "outputs": self._output_details,
        }
