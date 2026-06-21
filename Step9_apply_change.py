#!/usr/bin/env python
# coding: utf-8
"""Convert all symbolic links under the dataset root to regular files.

Equivalent to the bash workflow:

    for f in $(find MCP/ -type l); do cp --remove-destination $(readlink $f) $f; done

but applied recursively to all subdirectories (MCP, NN, SYN, DphID*,
Quality below 60, etc.) under the dataset root.
"""

import os
import shutil
from pathlib import Path


def convert_symlinks(root_dir):
    """Recursively convert all file symlinks under root_dir to regular files.

    For every symbolic link pointing to a regular file, the symlink is removed
    and replaced by a copy of its target (preserving metadata).

    Parameters
    ----------
    root_dir : str or Path
        Root directory to search for symbolic links.

    Returns
    -------
    tuple[int, int]
        Number of successfully converted symlinks and number of failures.
    """
    root_dir = Path(root_dir)
    converted = 0
    failed = 0

    for path in root_dir.rglob("*"):
        if not path.is_symlink():
            continue

        try:
            real_target = path.resolve()
            if not real_target.exists():
                print(f"Skipping broken symlink: {path} -> {path.readlink()}")
                failed += 1
                continue
            if not real_target.is_file():
                print(f"Skipping non-file symlink: {path} -> {real_target}")
                continue

            # Equivalent to cp --remove-destination $(readlink $f) $f
            path.unlink()
            shutil.copy2(real_target, path)
            converted += 1
            print(f"Converted: {path} <- {real_target}")
        except Exception as e:
            print(f"Failed to convert {path}: {e}")
            failed += 1

    print(f"Done. Converted {converted} symlink(s), failed {failed}.")
    return converted, failed

def delete_orignal_image():
    print("delete orignal images")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert all symbolic links under the dataset root to regular files."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        default="/media/filming/2025-白海豚/20240825-JM_02-3/",
        help="Root directory of the dataset",
    )
    args = parser.parse_args()
    convert_symlinks(args.root_dir)
