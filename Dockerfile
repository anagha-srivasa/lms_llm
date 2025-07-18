```dockerfile
# Use CUDA-enabled base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copy requirements and install (with CUDA support for PyTorch)
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY . /app

# Expose port if serving via REST/API
# EXPOSE 8080

# Default entrypoint (use device cuda if available)
ENV DEVICE=cuda
CMD ["python", "orchestrator.py", "--model_path", "./model", "--num_instances", "4", "--device", "$DEVICE"]
```
