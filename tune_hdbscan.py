import argparse
import boto3
import io
import numpy as np
import umap
import hdbscan
from sklearn.metrics import silhouette_score

def load_embeddings(bucket_name, prefix):
    """Download all .npy embeddings from S3 and stack into numpy array."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    all_embeddings = []
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".npy"):
                continue

            file_stream = s3.get_object(Bucket=bucket_name, Key=key)["Body"].read()
            emb = np.load(io.BytesIO(file_stream))
            all_embeddings.append(emb)

    if not all_embeddings:
        raise ValueError("No .npy embeddings found in S3 path!")

    return np.vstack(all_embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name", type=str, required=True)
    parser.add_argument("--input_prefix", type=str, required=True)
    parser.add_argument("--min_cluster_size", type=int, required=True)
    args = parser.parse_args()

    # Load all embeddings
    embeddings = load_embeddings(args.bucket_name, args.input_prefix)

    # UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    # HDBSCAN clustering with tuned hyperparameter
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size)
    labels = clusterer.fit_predict(reduced)

    # Compute noise ratio metric
    noise_ratio = (labels == -1).sum() / len(labels)

    # Katib expects: <metric-name>=<value>
    print(f"metric={noise_ratio}")


if __name__ == "__main__":
    main()
