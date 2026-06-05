# syntax=docker/dockerfile:1.7

ARG CUDA_VERSION=12.4.0
ARG PYTHON_VERSION=3.10
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-pip \
        python${PYTHON_VERSION}-venv \
        libhdf5-dev \
        ca-certificates \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tomo_image_stitcher

# Install CPU-only Python deps first (better Docker layer caching)
COPY pyproject.toml ./
COPY src ./src
COPY README.md ./
COPY LICENSE ./

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -e . \
    && python -m pip install --no-cache-dir cupy-cuda12x \
    && python -m pip install --no-cache-dir jupyter ipywidgets

# Smoke test
RUN python -c "import tomo_image_stitcher; print('tomo_image_stitcher', tomo_image_stitcher.__version__)"

# Jupyter by default; override with `docker run ... python -c "..."` for batch jobs
EXPOSE 8888
WORKDIR /work
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
