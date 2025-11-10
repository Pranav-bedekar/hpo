# Use Python 3.10 slim as base
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements directly into image (UMAP, HDBSCAN need C build tools)
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install boto3 numpy pandas umap-learn hdbscan scikit-learn tqdm

# Copy your script
COPY tune_hdbscan.py .

# Default entrypoint (Katib overrides args)
ENTRYPOINT ["python", "tune_hdbscan.py"]
