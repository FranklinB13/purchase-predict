import glob

import pandas as pd
from google.cloud import storage


def load_csv_from_bucket(project: str, bucket_path: str) -> pd.DataFrame:
    """
    Loads multiple CSV files from bucket (as generated from Spark).
    """
    storage_client = storage.Client(project=project)
    bucket_name, folder_path = bucket_path.split("/", maxsplit=1)
    folder = f"{folder_path}/part-"
    for blob in storage_client.list_blobs(bucket_name, prefix=folder):
        filename = blob.name.split("/")[-1]
        if filename.endswith("csv"):
            blob.download_to_filename("/tmp/" + filename)

    all_files = glob.glob("/tmp/*.csv")
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, sep=",")
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    return df
