import os
import glob
import tarfile
import pandas as pd
import json
from io import StringIO
import gzip
import datetime
import zipfile
import numpy as np

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

def read_file_from_zip_file_as_dataframe(zip_file_path, target_file_name, **kwargs):
    """
    Reads a specific file inside a zip file as a pandas DataFrame.

    Args:
        zip_path (str): The path to the zip file.
        target_file_name (str): The name of the file to read within the zip file.
        **kwargs: Additional arguments to pass to pandas.read_csv().

    Returns:
        pd.DataFrame: The content of the specified file as a pandas DataFrame.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        try:
            with zip_ref.open(target_file_name) as file:
                return pd.read_csv(file, **kwargs)
        except Exception as e:
            printf(f"Error: {e}")
            return None 

def read_file_from_tar_gz_as_dataframe(tar_gz_path, target_file_name, **kwargs):
    """
    Reads a specific file inside a folder within a tar.gz archive as a pandas DataFrame.

    Args:
        tar_gz_path (str): The path to the tar.gz file.
        target_file_name (str): The name of the file to read within the folder inside the archive.
        **kwargs: Additional arguments to pass to pandas.read_csv().

    Returns:
        pd.DataFrame: The content of the specified file as a pandas DataFrame.
    """
    try:
        with tarfile.open(tar_gz_path, 'r:gz') as tar:
            # Locate the target file inside the folder
            for member in tar.getmembers():
                if member.name.endswith(f"/{target_file_name}"):
                    with tar.extractfile(member) as file:
                        # Read the file into a DataFrame
                        return pd.read_csv(StringIO(file.read().decode('utf-8')), **kwargs)
            raise FileNotFoundError(f"File '{target_file_name}' not found in the archive.")
    except Exception as e:
        printf(f"Error: {e}")
        return None
    
def save_dicts_to_text_file(dict_list, file_path):
    """
    Saves a list of dictionaries to a text file, with one dictionary per line.

    Args:
        dict_list (list): A list of dictionaries to save.
        file_path (str): The path to the text file where data will be saved.

    Returns:
        None
    """
    try:
        with open(file_path, 'w') as file:
            for dictionary in dict_list:
                # Convert each dictionary to a JSON string and write it to the file
                file.write(json.dumps(dictionary) + '\n')
        printf(f"Data successfully saved to {file_path}")
    except Exception as e:
        printf(f"Error: {e}")

def save_list_to_file(data_list, file_path):
    """
    Save a list to a file, with each item on a new line.

    Parameters:
        data_list (list): The list of items to save.
        file_path (str): The path of the file where the list should be saved.
    """
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")


def read_compressed_json(file_path):
    """
    Reads a compressed JSON file in .json.gz format.

    Parameters:
        file_path (str): Path to the .json.gz file.

    Returns:
        object: The Python object (dict or list) loaded from the JSON file.
    """
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        printf(f"Error reading compressed JSON file: {e}")
        return None
    
def read_list_of_dicts(file_path):

    """
    Read a list of dictionaries from a text file, where each line is a JSON object.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        list: A list of dictionaries read from the file.
    """
    try:
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
            return data
    except Exception as e:
        printf(f"Error reading list of dicts from {file_path}: {e}")
        return None

def save_json_file(data, file_path):
    """
    Save a dictionary to a JSON file.

    Parameters:
        data (dict): The dictionary to save.
        file_path (str): The path to the JSON file.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        printf(f"Error saving JSON file: {e}")


def read_json_file(file_path):
    """
    Read a JSON file and return its content.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: Parsed content of the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
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
        verbose = kwargs.pop('verbose', True)
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
    if os.path.isfile(path) or os.path.splitext(path)[1]:  # Assume paths with extensions are files
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
        
def save_numpy_array(array, fn):
    try:
        np.save(fn, array)
    except Exception as e:
        printf(f"Error: {e}")