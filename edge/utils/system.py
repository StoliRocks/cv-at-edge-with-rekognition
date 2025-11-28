"""
System utilities for Jetson device monitoring.
"""

import logging
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """System information."""
    hostname: str
    platform: str
    architecture: str
    python_version: str
    cpu_count: int
    memory_total_mb: int
    memory_available_mb: int
    disk_total_gb: float
    disk_free_gb: float


@dataclass
class JetsonInfo:
    """NVIDIA Jetson device information."""
    model: str
    jetpack_version: str
    l4t_version: str
    cuda_version: str
    tensorrt_version: str
    serial_number: str
    module: str


@dataclass
class GPUMemory:
    """GPU memory information."""
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_percent: float


def get_system_info() -> SystemInfo:
    """Get general system information."""
    import psutil

    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return SystemInfo(
        hostname=platform.node(),
        platform=platform.system(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        memory_total_mb=int(memory.total / (1024 * 1024)),
        memory_available_mb=int(memory.available / (1024 * 1024)),
        disk_total_gb=disk.total / (1024 ** 3),
        disk_free_gb=disk.free / (1024 ** 3),
    )


def get_jetson_info() -> Optional[JetsonInfo]:
    """
    Get NVIDIA Jetson device information.
    Returns None if not running on a Jetson device.
    """
    # Check if we're on a Jetson
    if not Path("/etc/nv_tegra_release").exists():
        return None

    try:
        # Read L4T version
        l4t_version = ""
        with open("/etc/nv_tegra_release") as f:
            content = f.read()
            # Parse: # R35 (release), REVISION: 4.1, ...
            if "R" in content:
                parts = content.split(",")
                if parts:
                    l4t_version = parts[0].replace("#", "").strip()

        # Get Jetson model
        model = "Unknown"
        module = "Unknown"
        try:
            result = subprocess.run(
                ["cat", "/sys/firmware/devicetree/base/model"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                model = result.stdout.strip().replace("\x00", "")
        except Exception:
            pass

        # Get JetPack version from dpkg
        jetpack_version = ""
        try:
            result = subprocess.run(
                ["dpkg-query", "--showformat=${Version}", "--show", "nvidia-jetpack"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                jetpack_version = result.stdout.strip()
        except Exception:
            pass

        # Get CUDA version
        cuda_version = ""
        cuda_path = Path("/usr/local/cuda/version.txt")
        if cuda_path.exists():
            with open(cuda_path) as f:
                cuda_version = f.read().strip().replace("CUDA Version ", "")
        else:
            # Try nvcc
            try:
                result = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "release" in line.lower():
                            cuda_version = line.split("release")[-1].split(",")[0].strip()
                            break
            except Exception:
                pass

        # Get TensorRT version
        tensorrt_version = ""
        try:
            result = subprocess.run(
                ["dpkg-query", "--showformat=${Version}", "--show", "tensorrt"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                tensorrt_version = result.stdout.strip()
        except Exception:
            pass

        # Get serial number
        serial_number = ""
        try:
            result = subprocess.run(
                ["cat", "/sys/firmware/devicetree/base/serial-number"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                serial_number = result.stdout.strip().replace("\x00", "")
        except Exception:
            pass

        # Detect module type from model
        model_lower = model.lower()
        if "orin" in model_lower:
            if "nano" in model_lower:
                module = "Orin Nano"
            elif "nx" in model_lower:
                module = "Orin NX"
            else:
                module = "Orin AGX"
        elif "xavier" in model_lower:
            if "nx" in model_lower:
                module = "Xavier NX"
            else:
                module = "Xavier AGX"
        elif "nano" in model_lower:
            module = "Nano"
        elif "tx2" in model_lower:
            module = "TX2"

        return JetsonInfo(
            model=model,
            jetpack_version=jetpack_version,
            l4t_version=l4t_version,
            cuda_version=cuda_version,
            tensorrt_version=tensorrt_version,
            serial_number=serial_number,
            module=module,
        )

    except Exception as e:
        logger.warning(f"Failed to get Jetson info: {e}")
        return None


def get_gpu_memory() -> Optional[GPUMemory]:
    """
    Get GPU memory usage.
    Works with both NVIDIA desktop GPUs and Jetson.
    """
    # Try tegrastats for Jetson
    tegra_stats_path = Path("/sys/kernel/debug/nvmap/iovmm/maps")
    if tegra_stats_path.exists():
        try:
            # Use tegrastats output parsing
            # This is a simplified version
            result = subprocess.run(
                ["tegrastats", "--interval", "1", "--once"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse tegrastats output
                # Example: RAM 3456/7850MB (lfb 16x4MB) ...
                output = result.stdout
                if "RAM" in output:
                    ram_part = output.split("RAM")[1].split()[0]
                    used, total = ram_part.replace("MB", "").split("/")
                    return GPUMemory(
                        total_mb=int(total),
                        used_mb=int(used),
                        free_mb=int(total) - int(used),
                        utilization_percent=(int(used) / int(total)) * 100,
                    )
        except Exception:
            pass

    # Try nvidia-smi for desktop GPUs
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 4:
                return GPUMemory(
                    total_mb=int(float(parts[0].strip())),
                    used_mb=int(float(parts[1].strip())),
                    free_mb=int(float(parts[2].strip())),
                    utilization_percent=float(parts[3].strip()),
                )
    except Exception:
        pass

    return None


def get_cpu_temperature() -> Optional[float]:
    """Get CPU temperature in Celsius."""
    # Try thermal zone (common on Jetson)
    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_path.exists():
        try:
            with open(thermal_path) as f:
                temp_millicelsius = int(f.read().strip())
                return temp_millicelsius / 1000.0
        except Exception:
            pass

    # Try hwmon
    hwmon_base = Path("/sys/class/hwmon")
    if hwmon_base.exists():
        for hwmon in hwmon_base.iterdir():
            temp_input = hwmon / "temp1_input"
            if temp_input.exists():
                try:
                    with open(temp_input) as f:
                        temp_millicelsius = int(f.read().strip())
                        return temp_millicelsius / 1000.0
                except Exception:
                    continue

    return None


def get_power_consumption() -> Optional[float]:
    """Get current power consumption in Watts (Jetson only)."""
    # Jetson power rails
    power_paths = [
        "/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221/*/hwmon/*/power1_input",
    ]

    import glob

    for pattern in power_paths:
        matches = glob.glob(pattern)
        if matches:
            try:
                with open(matches[0]) as f:
                    power_mw = int(f.read().strip())
                    return power_mw / 1000.0
            except Exception:
                continue

    return None


def set_jetson_power_mode(mode: str = "MAXN") -> bool:
    """
    Set Jetson power mode.

    Args:
        mode: Power mode (MAXN, 15W, 10W, etc.)

    Returns:
        True if successful
    """
    try:
        result = subprocess.run(
            ["nvpmodel", "-m", mode],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to set power mode: {e}")
        return False


def set_jetson_clocks(enable: bool = True) -> bool:
    """
    Enable or disable jetson_clocks for maximum performance.

    Args:
        enable: True to enable max clocks, False to restore defaults

    Returns:
        True if successful
    """
    try:
        cmd = ["jetson_clocks"] if enable else ["jetson_clocks", "--restore"]
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to set jetson clocks: {e}")
        return False
