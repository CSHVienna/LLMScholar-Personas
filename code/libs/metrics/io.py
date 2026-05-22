import datetime
import glob
import json
import os

import pandas as pd


def printf(message):
    # Get the current timestamp in the desired format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Print the message with the timestamp prepended
    print(f"[{timestamp}] {message}")


def path_join(*path_segments):
    """
    Joins multiple path components into a single path.

    Args:
        *path_segments (str): Path components to join.

    Returns:
        str: The joined path.
    """
    return os.path.join(*path_segments)


def read_list_of_dicts(file_path):
    """
    Read a list of dictionaries from a text file, where each line is a JSON object.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        list: A list of dictionaries read from the file.
    """
    try:
        with open(file_path, "r") as file:
            data = [json.loads(line) for line in file]
            return data
    except Exception as e:
        printf(f"Error reading list of dicts from {file_path}: {e}")
        return None


def read_json_file(file_path):
    """
    Read a JSON file and return its content.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: Parsed content of the JSON file.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        printf(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        printf(f"Error: Failed to decode JSON. Details: {e}")
    except Exception as e:
        printf(f"An unexpected error occurred: {e}")


def read_csv(fn, **kwargs):
    try:
        return pd.read_csv(fn, **kwargs)
    except Exception as e:
        printf(f"Error: {e}")
        return None


def save_csv(df, fn, **kwargs):
    try:
        verbose = kwargs.pop("verbose", True)
        df.to_csv(fn, **kwargs)
        if verbose:
            printf(f"Data successfully saved to {fn}")
    except Exception as e:
        printf(f"Error: {e}")


def exists(fn):
    return os.path.exists(fn)


def get_files(path, pattern):
    return glob.glob(os.path.join(path, pattern))


def validate_path(path):
    """
    Ensures all directories in the given path exist.
    - If the path is a file, ensures its containing directory exists.
    - If the path is a directory, ensures the entire directory path exists.
    """
    # Check if the path is a directory or a file
    if (
        os.path.isfile(path) or os.path.splitext(path)[1]
    ):  # Assume paths with extensions are files
        dir_path = os.path.dirname(path)
    else:  # Otherwise, treat it as a directory path
        dir_path = path

    # Create directories if they do not exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_text(text, fn):
    try:
        with open(fn, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        printf(f"Error: {e}")
