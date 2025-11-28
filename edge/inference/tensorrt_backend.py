"""
TensorRT inference backend optimized for NVIDIA Jetson devices.
Provides the fastest inference using TensorRT engine files.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# TensorRT imports with graceful fallback
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 - Required for CUDA context
    TRT_AVAILABLE = True
except ImportError as e:
    TRT_AVAILABLE = False
    logger.warning(f"TensorRT not available: {e}. Install with: pip install tensorrt pycuda")


class TensorRTBackend:
    """
    TensorRT inference backend for maximum performance on Jetson.

    Supports:
    - FP32, FP16, and INT8 precision
    - Dynamic batch sizes
    - CUDA stream optimization
    - Memory-efficient inference
    """

    def __init__(self, device: str = "cuda:0"):
        """
        Initialize TensorRT backend.

        Args:
            device: CUDA device to use (e.g., "cuda:0")
        """
        if not TRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT is not available. Please install with:\n"
                "pip install tensorrt pycuda\n"
                "Or use the Jetson SDK which includes TensorRT."
            )

        self.device = device
        self.engine: Optional[trt.ICudaEngine] = None
        self.context: Optional[trt.IExecutionContext] = None
        self.stream: Optional[cuda.Stream] = None
        self.bindings: list = []
        self.inputs: list = []
        self.outputs: list = []
        self._input_shape: Optional[Tuple[int, ...]] = None

        # Initialize TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.engine is not None and self.context is not None

    def load_model(self, model_path: Path) -> None:
        """
        Load TensorRT engine from file.

        Args:
            model_path: Path to .engine or .trt file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading TensorRT engine from {model_path}")

        # Deserialize engine
        with open(model_path, "rb") as f:
            runtime = trt.Runtime(self.trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {model_path}")

        # Create execution context
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        # Create CUDA stream for async operations
        self.stream = cuda.Stream()

        # Allocate buffers
        self._allocate_buffers()

        logger.info(
            f"TensorRT engine loaded: {self.engine.num_bindings} bindings, "
            f"input shape: {self._input_shape}"
        )

    def _allocate_buffers(self) -> None:
        """Allocate GPU memory buffers for inference."""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)

            # Handle dynamic shapes
            if -1 in shape:
                # Use optimization profile shape
                shape = self.context.get_binding_shape(i)

            size = trt.volume(shape)

            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            binding_info = {
                "name": binding_name,
                "dtype": dtype,
                "shape": shape,
                "host": host_mem,
                "device": device_mem,
            }

            if self.engine.binding_is_input(i):
                self.inputs.append(binding_info)
                self._input_shape = shape
            else:
                self.outputs.append(binding_info)

    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Get expected input shape (batch, channels, height, width)."""
        if self._input_shape is None:
            raise RuntimeError("Model not loaded")
        return tuple(self._input_shape)

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

        # Ensure contiguous array
        image = np.ascontiguousarray(image)

        # Copy input to host buffer
        np.copyto(self.inputs[0]["host"], image.ravel())

        # Transfer input to GPU
        cuda.memcpy_htod_async(
            self.inputs[0]["device"],
            self.inputs[0]["host"],
            self.stream
        )

        # Execute inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Transfer outputs back to host
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output["host"], output["device"], self.stream)

        # Synchronize stream
        self.stream.synchronize()

        # Return first output (reshape to expected dimensions)
        output = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        return output

    def warmup(self, iterations: int = 10) -> None:
        """Warm up the model for consistent performance."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        logger.info(f"Warming up TensorRT engine with {iterations} iterations")

        # Create dummy input
        dummy_input = np.random.randn(*self._input_shape).astype(np.float32)

        for i in range(iterations):
            _ = self.infer(dummy_input)

        logger.info("TensorRT warmup complete")

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.synchronize()


def build_engine_from_onnx(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 30,  # 1GB
) -> None:
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        fp16: Enable FP16 precision (recommended for Jetson)
        int8: Enable INT8 precision (requires calibration)
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace memory in bytes
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT not available")

    logger.info(f"Building TensorRT engine from {onnx_path}")

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    # Parse ONNX
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"ONNX Parser Error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    if fp16 and builder.platform_has_fast_fp16:
        logger.info("Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)

    if int8 and builder.platform_has_fast_int8:
        logger.info("Enabling INT8 precision")
        config.set_flag(trt.BuilderFlag.INT8)

    # Build engine
    logger.info("Building TensorRT engine (this may take several minutes)...")
    engine = builder.build_engine(network, config)

    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # Serialize and save
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    logger.info(f"TensorRT engine saved to {engine_path}")
