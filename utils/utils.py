import os
from pathlib import Path


def get_repo_root(marker_files=(".git", "pyproject.toml", "requirements.txt", "setup.cfg", "README.md")) -> Path:
    """
    Dynamically find the repository root by looking for known marker files.
    Works from any file or notebook location within the repository.
    
    Args:
        marker_files: Tuple of filenames to look for that indicate the repo root
        
    Returns:
        Path: The repository root directory path
        
    Raises:
        FileNotFoundError: If repository root cannot be found
        
    Example:
        >>> from utils import get_repo_root
        >>> repo_root = get_repo_root()
        >>> data_path = repo_root / "data" / "customers.csv"
    """
    # Handle both regular Python files and Jupyter notebooks
    if "__file__" in globals():
        current_path = Path(__file__).resolve()
    else:
        # In Jupyter notebooks, __file__ is not defined
        current_path = Path.cwd()
    
    # Traverse up the directory tree to find the repository root
    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    
    raise FileNotFoundError(
        f"❌ Repository root not found. Make sure one of these files exists at the root: {', '.join(marker_files)}"
    )
