# Jetson Edge CV Application

A production-ready computer vision application optimized for NVIDIA Jetson devices with AWS IoT integration.

## Features

- **Local Inference**: Run object detection directly on Jetson with TensorRT, ONNX Runtime, or TensorFlow Lite
- **Multiple Video Sources**: Support for CSI cameras, USB cameras, RTSP streams, and video files
- **AWS IoT Integration**: Real-time publishing to AWS IoT Core with device shadow support
- **Configurable ROI**: Define regions of interest for occupancy detection
- **State Management**: Track occupancy state with hysteresis to prevent flickering
- **S3 Upload**: Optional frame upload to S3 for archival or cloud processing
- **Docker Support**: Containerized deployment with NVIDIA runtime

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    JETSON EDGE DEVICE                           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ Video Source│ -> │  Inference  │ -> │  Frame Processor    │ │
│  │ (CSI/USB/   │    │ (TensorRT/  │    │  - Detection Filter │ │
│  │  RTSP/File) │    │  ONNX/TFLite│    │  - IoU Calculation  │ │
│  └─────────────┘    └─────────────┘    │  - State Tracking   │ │
│                                         └──────────┬──────────┘ │
│                                                    │            │
│  ┌────────────────────────────────────────────────┴──────────┐ │
│  │                    IoT Client                              │ │
│  │  - MQTT Publishing to AWS IoT Core                        │ │
│  │  - Device Shadow Updates                                  │ │
│  │  - S3 Frame Uploads (optional)                            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ MQTT over TLS
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       AWS CLOUD                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ IoT Core     │    │ IoT Shadow   │    │ CloudWatch       │  │
│  │ (MQTT Broker)│    │ (State Sync) │    │ (Monitoring)     │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ S3           │    │ Lambda       │    │ Step Functions   │  │
│  │ (Frame Store)│    │ (Processing) │    │ (Orchestration)  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Supported Devices

- NVIDIA Jetson Orin Nano / NX / AGX
- NVIDIA Jetson Xavier NX / AGX Xavier
- NVIDIA Jetson Nano
- JetPack 5.x and 6.x compatible

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd cv-at-edge-with-rekognition/edge

# Run the setup script
./scripts/setup-jetson.sh
```

### 2. Configure

Edit `config/config.yaml` or set environment variables:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your AWS credentials and settings
nano .env
```

### 3. Run

```bash
# Activate virtual environment
source ~/edge-cv-venv/bin/activate

# Run with configuration file
python -m edge.main --config config/config.yaml

# Or with command line arguments
python -m edge.main --video /dev/video0 --model models/yolov8n.onnx --display
```

## Installation

### Prerequisites

- NVIDIA Jetson device with JetPack 5.x or 6.x
- Python 3.10+
- CUDA and TensorRT (included in JetPack)

### Automated Setup

```bash
# Full setup with system dependencies
./scripts/setup-jetson.sh

# With TensorRT model conversion (recommended for best performance)
./scripts/setup-jetson.sh --tensorrt

# Set maximum performance mode
./scripts/setup-jetson.sh --max-perf
```

### Manual Setup

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-opencv \
    libgstreamer1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good

# Create virtual environment
python3 -m venv --system-site-packages ~/edge-cv-venv
source ~/edge-cv-venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Configuration

### Configuration File (YAML)

```yaml
app_name: jetson-edge-cv
debug: false

video:
  source: "0"               # Camera index, file path, or RTSP URL
  source_type: auto         # auto, file, camera, rtsp, csi, usb
  width: 1280
  height: 720
  fps: 30

inference:
  model_path: models/yolov8n.engine  # .onnx or .engine
  backend: auto             # auto, tensorrt, onnx, tflite
  device: "cuda:0"
  confidence_threshold: 0.5

detection:
  target_classes:
    - truck
    - car
  iou_threshold: 0.3
  roi_x1: 0.10              # Region of interest (normalized)
  roi_y1: 0.10
  roi_x2: 0.85
  roi_y2: 0.90

aws:
  region: us-east-1
  iot_endpoint: "xxxxx-ats.iot.us-east-1.amazonaws.com"
  iot_thing_name: JetsonEdgeDevice
  iot_topic: garage/001/bay/01
```

### Environment Variables

All settings can be overridden with environment variables using the `EDGE_` prefix:

```bash
export EDGE_VIDEO__SOURCE=/dev/video0
export EDGE_INFERENCE__MODEL_PATH=models/yolov8n.engine
export EDGE_AWS__IOT_ENDPOINT=xxxxx-ats.iot.us-east-1.amazonaws.com
```

## Model Optimization

### Convert ONNX to TensorRT

For best performance on Jetson, convert your ONNX model to TensorRT:

```bash
# Using trtexec
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/yolov8n.onnx \
    --saveEngine=models/yolov8n.engine \
    --fp16

# Or using the setup script
./scripts/setup-jetson.sh --tensorrt
```

### Download Pre-trained Models

```bash
# YOLOv8 Nano (recommended for Jetson Nano)
wget -O models/yolov8n.onnx \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx

# YOLOv8 Small (for more powerful Jetson devices)
wget -O models/yolov8s.onnx \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.onnx
```

## Docker Deployment

### Build Image

```bash
docker build -t jetson-edge-cv:latest .
```

### Run Container

```bash
docker run --runtime nvidia \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/certs:/app/certs \
    -v $(pwd)/config:/app/config \
    --device /dev/video0:/dev/video0 \
    --network host \
    jetson-edge-cv:latest
```

### Docker Compose

```bash
docker-compose up -d
```

## AWS IoT Setup

### 1. Create IoT Thing

```bash
aws iot create-thing --thing-name JetsonEdgeDevice
```

### 2. Create and Download Certificates

```bash
# Create certificate
aws iot create-keys-and-certificate \
    --set-as-active \
    --certificate-pem-outfile certs/device.cert.pem \
    --private-key-outfile certs/device.private.key

# Download Root CA
wget -O certs/AmazonRootCA1.pem \
    https://www.amazontrust.com/repository/AmazonRootCA1.pem
```

### 3. Attach Policy to Certificate

```bash
# Create policy (see docs/iot-policy.json for example)
aws iot create-policy \
    --policy-name JetsonEdgePolicy \
    --policy-document file://docs/iot-policy.json

# Attach policy
aws iot attach-policy \
    --policy-name JetsonEdgePolicy \
    --target <certificate-arn>

# Attach thing
aws iot attach-thing-principal \
    --thing-name JetsonEdgeDevice \
    --principal <certificate-arn>
```

### 4. Get IoT Endpoint

```bash
aws iot describe-endpoint --endpoint-type iot:Data-ATS
```

## Performance Tuning

### Jetson Power Modes

```bash
# Set to maximum performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Check current mode
nvpmodel -q
```

### Inference Optimization

| Device | Recommended Model | Expected FPS |
|--------|------------------|--------------|
| Jetson Nano | YOLOv8n (TensorRT) | 15-20 FPS |
| Jetson Xavier NX | YOLOv8s (TensorRT) | 30-40 FPS |
| Jetson Orin Nano | YOLOv8s (TensorRT) | 40-60 FPS |
| Jetson Orin NX | YOLOv8m (TensorRT) | 60-80 FPS |

### Memory Optimization

```yaml
# For memory-constrained devices
video:
  width: 640
  height: 480

inference:
  input_size: [320, 320]  # Smaller input size
```

## Monitoring

### View Logs

```bash
# Application logs
tail -f logs/edge-cv.log

# Docker logs
docker-compose logs -f edge-cv
```

### Metrics

The application publishes metrics to AWS CloudWatch (when configured) and logs:
- Frames per second (FPS)
- Inference latency (ms)
- Detection count
- Occupancy state duration

## Troubleshooting

### Camera Not Found

```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink
```

### TensorRT Errors

```bash
# Rebuild engine for current device
/usr/src/tensorrt/bin/trtexec \
    --onnx=models/yolov8n.onnx \
    --saveEngine=models/yolov8n.engine \
    --fp16
```

### IoT Connection Issues

```bash
# Test MQTT connection
mosquitto_pub -h <endpoint> -p 8883 \
    --cafile certs/AmazonRootCA1.pem \
    --cert certs/device.cert.pem \
    --key certs/device.private.key \
    -t test/topic -m "test"
```

## Project Structure

```
edge/
├── inference/          # Inference backends (TensorRT, ONNX, TFLite)
├── pipeline/           # Video capture and processing
├── iot/               # AWS IoT client and S3 uploader
├── config/            # Configuration management
├── utils/             # System utilities and metrics
├── scripts/           # Setup and deployment scripts
├── main.py            # Application entry point
├── Dockerfile         # Container definition
├── docker-compose.yml # Container orchestration
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## License

MIT License - see LICENSE file for details.
