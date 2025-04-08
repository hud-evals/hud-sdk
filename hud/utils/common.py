from __future__ import annotations

import io
import logging
import tarfile
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger("hud.utils.common")

class ExecuteResult(TypedDict):
    """
    Result of an execute command.

    Attributes:
        stdout: Standard output from the command
        stderr: Standard error from the command
        exit_code: Exit code of the command
    """
    stdout: bytes
    stderr: bytes
    exit_code: int
    
    
def directory_to_tar_bytes(directory_path: Path) -> bytes:
    """
    Converts a directory to a tar archive and returns it as bytes.
    
    This function creates a tar archive of the specified directory in memory,
    without writing to a temporary file on disk.
    
    Args:
        path: Path to the directory to convert
        
    Returns:
        Bytes of the tar archive
    """
    output = io.BytesIO()
    
    with tarfile.open(fileobj=output, mode="w") as tar:
        # Walk through the directory
        for file_path in directory_path.rglob("*"):
            if file_path.is_file():
                # Calculate relative path for the archive
                rel_path = file_path.relative_to(directory_path)
                logger.debug("Adding %s to tar archive", rel_path)
                tar.add(file_path, arcname=str(rel_path))
    
    # Get the bytes from the BytesIO object
    output.seek(0)
    return output.getvalue()
