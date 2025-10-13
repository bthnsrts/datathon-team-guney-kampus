"""
utils package

This module provides utility functions for use throughout the repository.
Currently includes:
- get_repo_root: Dynamically locate the root directory of the repository.
"""

from .utils import get_repo_root

__all__ = ["get_repo_root"]