import os
import pandas as pd


def check_index_files(directory: str) -> bool:
    """Function to check whether there is a LlamaIndex index for document retrieval persisted to disk."""
    expected_files = {
        "docstore.json",
        "graph_store.json",
        "index_store.json",
        "vector_store.json",
    }
    actual_files = set(os.listdir(directory)) & expected_files
    return expected_files == actual_files


def get_filename(directory: str) -> str:
    """Function to return single file from a directory"""
    return os.path.join(directory, os.listdir(directory)[0])


def read_file_to_df(filename: str) -> pd.DataFrame:
    """Function that reads a file to a pandas DataFrame based on file extension."""
    if filename.endswith(".csv"):
        return pd.read_csv(filename)
    elif filename.endswith(".json"):
        return pd.read_json(filename)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
