"""
io.py — Reading and writing metric files.

  validate_path              create the parent folder of an output file
"""

import os


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
