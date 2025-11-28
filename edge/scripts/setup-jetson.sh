#!/bin/bash
# =============================================================================
# Jetson Edge CV Application - Setup Script
#
# This script sets up the development environment on NVIDIA Jetson devices.
# Supports JetPack 5.x and 6.x
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running on Jetson
check_jetson() {
    if [ -f /etc/nv_tegra_release ]; then
        log_info "Detected NVIDIA Jetson device"
        cat /etc/nv_tegra_release
        return 0
    else
        log_warning "Not running on Jetson device - some features may not work"
        return 1
    fi
}

# Check JetPack version
check_jetpack() {
    if dpkg -l | grep -q nvidia-jetpack; then
        JETPACK_VERSION=$(dpkg-query --showformat='${Version}' --show nvidia-jetpack 2>/dev/null || echo "unknown")
        log_info "JetPack version: $JETPACK_VERSION"
    else
        log_warning "JetPack not detected"
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."

    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-opencv \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        v4l-utils \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-tools \
        gstreamer1.0-libav

    log_success "System dependencies installed"
}

# Create Python virtual environment
create_venv() {
    VENV_PATH="${1:-$HOME/edge-cv-venv}"

    log_info "Creating Python virtual environment at $VENV_PATH..."

    # Create venv with system packages for OpenCV and TensorRT
    python3 -m venv --system-site-packages "$VENV_PATH"

    # Activate and upgrade pip
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip setuptools wheel

    log_success "Virtual environment created at $VENV_PATH"
    log_info "Activate with: source $VENV_PATH/bin/activate"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."

    # Install from requirements
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REQUIREMENTS_FILE="$SCRIPT_DIR/../requirements.txt"

    if [ -f "$REQUIREMENTS_FILE" ]; then
        pip install -r "$REQUIREMENTS_FILE"
    else
        log_warning "requirements.txt not found, installing core packages..."
        pip install numpy pydantic pydantic-settings PyYAML Pillow boto3 awsiotsdk psutil paho-mqtt
    fi

    # Install ONNX Runtime GPU for Jetson (if available)
    # On JetPack 5.x/6.x, this should be pre-installed or available via pip
    if check_jetson; then
        # Check if onnxruntime is available
        if ! python3 -c "import onnxruntime" 2>/dev/null; then
            log_info "Installing ONNX Runtime for Jetson..."
            # For JetPack 5.x: use the wheel from NVIDIA
            # pip install https://nvidia.box.com/shared/static/...
            pip install onnxruntime-gpu 2>/dev/null || log_warning "ONNX Runtime installation failed - manual installation may be required"
        fi
    fi

    log_success "Python dependencies installed"
}

# Download YOLOv8 model
download_model() {
    MODEL_DIR="${1:-./models}"
    MODEL_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"

    log_info "Downloading YOLOv8n model..."

    mkdir -p "$MODEL_DIR"

    if [ ! -f "$MODEL_DIR/yolov8n.onnx" ]; then
        wget -O "$MODEL_DIR/yolov8n.onnx" "$MODEL_URL"
        log_success "Model downloaded to $MODEL_DIR/yolov8n.onnx"
    else
        log_info "Model already exists at $MODEL_DIR/yolov8n.onnx"
    fi
}

# Convert ONNX to TensorRT engine
convert_to_tensorrt() {
    MODEL_PATH="${1:-./models/yolov8n.onnx}"
    ENGINE_PATH="${2:-./models/yolov8n.engine}"

    if ! check_jetson; then
        log_warning "TensorRT conversion requires Jetson device"
        return 1
    fi

    if [ ! -f "$MODEL_PATH" ]; then
        log_error "Model not found: $MODEL_PATH"
        return 1
    fi

    log_info "Converting ONNX to TensorRT engine..."
    log_info "This may take several minutes..."

    # Use trtexec for conversion
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$MODEL_PATH" \
        --saveEngine="$ENGINE_PATH" \
        --fp16 \
        --workspace=4096

    if [ -f "$ENGINE_PATH" ]; then
        log_success "TensorRT engine saved to $ENGINE_PATH"
    else
        log_error "TensorRT conversion failed"
        return 1
    fi
}

# Setup AWS IoT certificates directory
setup_certs_dir() {
    CERTS_DIR="${1:-./certs}"

    log_info "Setting up certificates directory..."

    mkdir -p "$CERTS_DIR"
    chmod 700 "$CERTS_DIR"

    # Download Amazon Root CA
    if [ ! -f "$CERTS_DIR/AmazonRootCA1.pem" ]; then
        wget -O "$CERTS_DIR/AmazonRootCA1.pem" \
            https://www.amazontrust.com/repository/AmazonRootCA1.pem
        log_success "Downloaded Amazon Root CA"
    fi

    log_info "Place your device certificate and private key in $CERTS_DIR:"
    log_info "  - device.cert.pem"
    log_info "  - device.private.key"
}

# Create default configuration
create_config() {
    CONFIG_DIR="${1:-./config}"

    log_info "Creating default configuration..."

    mkdir -p "$CONFIG_DIR"

    cat > "$CONFIG_DIR/config.yaml" << 'EOF'
# Jetson Edge CV Application Configuration
app_name: jetson-edge-cv
app_version: "1.0.0"
debug: false

video:
  source: "0"  # Camera device index or file path
  source_type: auto  # auto, file, camera, rtsp, csi, usb
  width: 1280
  height: 720
  fps: 30
  flip_method: 0  # CSI camera flip (0-7)
  buffer_size: 2

inference:
  model_path: models/yolov8n.onnx  # or .engine for TensorRT
  backend: auto  # auto, tensorrt, onnx, tflite
  device: "cuda:0"
  confidence_threshold: 0.5
  nms_threshold: 0.45
  input_size: [640, 640]
  warmup_iterations: 10

detection:
  target_classes:
    - truck
    - car
  iou_threshold: 0.3
  state_persistence_frames: 3
  roi_x1: 0.10
  roi_y1: 0.10
  roi_x2: 0.85
  roi_y2: 0.90
  roi_name: garage_bay_01

aws:
  region: us-east-1
  iot_endpoint: ""  # Your AWS IoT endpoint
  iot_thing_name: JetsonEdgeDevice
  iot_topic: garage/001/bay/01
  s3_bucket: ""  # Optional S3 bucket for frame uploads
  s3_prefix: frames/
  cert_path: certs/device.cert.pem
  key_path: certs/device.private.key
  root_ca_path: certs/AmazonRootCA1.pem
  upload_images: false
  upload_interval: 5

output:
  enable_display: false
  enable_rtsp: false
  rtsp_port: 8554
  rtsp_path: /stream
  save_video: false
  output_path: output/
  draw_detections: true
  draw_roi: true

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: null
  enable_metrics: true
EOF

    log_success "Configuration created at $CONFIG_DIR/config.yaml"
}

# Set Jetson to max performance mode
set_max_performance() {
    if ! check_jetson; then
        return 1
    fi

    log_info "Setting Jetson to maximum performance mode..."

    # Set power mode to MAXN
    sudo nvpmodel -m 0 2>/dev/null || true

    # Enable max clocks
    sudo jetson_clocks 2>/dev/null || true

    log_success "Performance mode set to maximum"
}

# Main setup function
main() {
    echo "=========================================="
    echo "  Jetson Edge CV Application Setup"
    echo "=========================================="
    echo

    # Parse arguments
    VENV_PATH="$HOME/edge-cv-venv"
    SKIP_SYSTEM_DEPS=false
    SKIP_VENV=false
    DOWNLOAD_MODEL=true
    CONVERT_TENSORRT=false
    SET_PERFORMANCE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --venv)
                VENV_PATH="$2"
                shift 2
                ;;
            --skip-system)
                SKIP_SYSTEM_DEPS=true
                shift
                ;;
            --skip-venv)
                SKIP_VENV=true
                shift
                ;;
            --no-model)
                DOWNLOAD_MODEL=false
                shift
                ;;
            --tensorrt)
                CONVERT_TENSORRT=true
                shift
                ;;
            --max-perf)
                SET_PERFORMANCE=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --venv PATH      Path for virtual environment (default: ~/edge-cv-venv)"
                echo "  --skip-system    Skip system dependency installation"
                echo "  --skip-venv      Skip virtual environment creation"
                echo "  --no-model       Skip model download"
                echo "  --tensorrt       Convert model to TensorRT engine"
                echo "  --max-perf       Set Jetson to max performance mode"
                echo "  -h, --help       Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Check platform
    check_jetson
    check_jetpack

    # Install system dependencies
    if [ "$SKIP_SYSTEM_DEPS" = false ]; then
        install_system_deps
    fi

    # Create virtual environment
    if [ "$SKIP_VENV" = false ]; then
        create_venv "$VENV_PATH"
        source "$VENV_PATH/bin/activate"
    fi

    # Install Python dependencies
    install_python_deps

    # Download model
    if [ "$DOWNLOAD_MODEL" = true ]; then
        download_model "./models"
    fi

    # Convert to TensorRT
    if [ "$CONVERT_TENSORRT" = true ]; then
        convert_to_tensorrt "./models/yolov8n.onnx" "./models/yolov8n.engine"
    fi

    # Setup certificates directory
    setup_certs_dir "./certs"

    # Create default configuration
    create_config "./config"

    # Set performance mode
    if [ "$SET_PERFORMANCE" = true ]; then
        set_max_performance
    fi

    echo
    echo "=========================================="
    log_success "Setup complete!"
    echo "=========================================="
    echo
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source $VENV_PATH/bin/activate"
    echo
    echo "2. Configure AWS IoT credentials in ./config/config.yaml"
    echo
    echo "3. Place your AWS IoT certificates in ./certs/"
    echo
    echo "4. Run the application:"
    echo "   python -m edge.main --config config/config.yaml"
    echo
    echo "5. For TensorRT acceleration (recommended), convert the model:"
    echo "   ./scripts/setup-jetson.sh --tensorrt"
    echo
}

# Run main function
main "$@"
