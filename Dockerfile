FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential

# Install Python packages
RUN pip install boto3 numpy pandas umap-learn hdbscan tqdm

# Copy script
COPY cluster_final_embeddings.py .

# Default command
CMD ["python", "cluster_final_embeddings.py"]
