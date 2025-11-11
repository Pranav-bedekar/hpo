import boto3
import io
import numpy as np
import pandas as pd
import umap
import hdbscan
from tqdm import tqdm

def run(
    bucket_name="video-datasets-kubeflow",
    input_prefix="final_emb2/",
    output_prefix="cluster-results/",
    n_neighbors=15,
    min_cluster_size=4
):

    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=input_prefix)

    all_embeddings = []
    filenames = []

    for page in page_iterator:
        for obj in tqdm(page.get("Contents", []), desc="Downloading embeddings"):
            key = obj["Key"]
            if not key.endswith(".npy"):
                continue

            file_stream = s3.get_object(Bucket=bucket_name, Key=key)["Body"].read()
            emb = np.load(io.BytesIO(file_stream))

            all_embeddings.append(emb)
            filenames.append(key.split("/")[-1])

    if not all_embeddings:
        print("No embeddings found.")
        return

    embeddings = np.vstack(all_embeddings)
    print(f"Loaded embeddings: {embeddings.shape}")

    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    df = pd.DataFrame({
        "filename": filenames,
        "cluster": cluster_labels,
        "e1": reduced_embeddings[:, 0],
        "e2": reduced_embeddings[:, 1]
    })

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    output_key = f"{output_prefix}cluster_results.csv"
    s3.put_object(
        Bucket=bucket_name,
        Key=output_key,
        Body=csv_buffer.getvalue()
    )

    print(f"Uploaded → s3://{bucket_name}/{output_key}")


if __name__ == "__main__":
    run()
