# Jetson Edge CV Application
# Multi-stage build for optimized image size
# Supports NVIDIA Jetson devices with JetPack 5.x / 6.x

# =============================================================================
# Stage 1: Base image with NVIDIA runtime
# =============================================================================
ARG L4T_VERSION=r35.4.1
ARG CUDA_VERSION=11.4
ARG PYTHON_VERSION=3.10

# Use NVIDIA L4T base image for Jetson
FROM nvcr.io/nvidia/l4t-ml:${L4T_VERSION}-py3 AS base

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    libopencv-dev \
    python3-opencv \
    python3-pip \
    python3-dev \
    v4l-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Python dependencies
# =============================================================================
FROM base AS python-deps

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# =============================================================================
# Stage 3: Final application image
# =============================================================================
FROM python-deps AS final

# Create application directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r edge && useradd -r -g edge edge

# Copy application code
COPY --chown=edge:edge . /app/

# Create directories for runtime data
RUN mkdir -p /app/models /app/certs /app/output /app/logs \
    && chown -R edge:edge /app

# Set permissions
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# Environment variables for runtime
ENV EDGE_APP_NAME="jetson-edge-cv"
ENV EDGE_DEBUG=false
ENV EDGE_INFERENCE__BACKEND=auto
ENV EDGE_INFERENCE__DEVICE=cuda:0
ENV EDGE_VIDEO__WIDTH=1280
ENV EDGE_VIDEO__HEIGHT=720
ENV EDGE_VIDEO__FPS=30
ENV EDGE_OUTPUT__ENABLE_DISPLAY=false

# Expose ports
EXPOSE 8554/tcp
EXPOSE 8080/tcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Switch to non-root user
USER edge

# Default command
ENTRYPOINT ["python3", "-m", "edge.main"]
CMD ["--config", "/app/config/config.yaml"]

# =============================================================================
# Alternative: Development image with additional tools
# =============================================================================
FROM final AS dev

USER root

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    iotop \
    net-tools \
    iputils-ping \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip3 install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    ipython

USER edge

# Override entrypoint for development
ENTRYPOINT ["/bin/bash"]
