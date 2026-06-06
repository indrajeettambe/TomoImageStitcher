# syntax=docker/dockerfile:1.7
#
# Multi-stage build for tomo-image-stitcher.
# Stage 1 (builder) installs the package and all dependencies into a
# virtualenv; stage 2 (runtime) carries only that venv + the bits of
# CUDA runtime that cupy needs at import time.

# --------------------------------------------------------------------- builder
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.11 matches the version used by the publish-pypi workflow.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only what is needed to resolve and install the package.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip \
    && pip install ".[all]"


# --------------------------------------------------------------------- runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Carry over the prepared virtualenv.
COPY --from=builder /opt/venv /opt/venv

# Quick health-check: importing the package should print the version.
RUN python -c "import tomo_image_stitcher; print(tomo_image_stitcher.__version__)"

# Default to a Python REPL. Override with `docker run --rm <image> python -c "..."`.
ENTRYPOINT ["python"]
CMD ["-c", "import tomo_image_stitcher; print(tomo_image_stitcher.__version__)"]
