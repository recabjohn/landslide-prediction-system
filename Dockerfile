# ================================================
# AI Landslide Monitoring System — Dockerfile
# ================================================
# Streamlit Dashboard · YOLOv8 · Open-Meteo API
# Optimized for Azure Container Apps
# ================================================

FROM python:3.11-slim

# System dependencies for OpenCV + image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch CPU-only first (smaller, faster)
RUN pip install --no-cache-dir --timeout=300 \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install (without torch since we installed it above)
COPY requirements.txt .
RUN grep -v "torch" requirements.txt | grep -v "extra-index-url" > requirements-notorch.txt && \
    pip install --no-cache-dir --timeout=300 -r requirements-notorch.txt && \
    rm requirements-notorch.txt

# Copy the entire project
COPY . .

# Create dataset directory
RUN mkdir -p dataset/indian_hills

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "ui/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
