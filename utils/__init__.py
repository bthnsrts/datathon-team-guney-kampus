"""
utils package

This module provides utility functions for use throughout the repository.
Currently includes:
- get_repo_root: Dynamically locate the root directory of the repository.
- bucketize_sum_month_lags: Create lagged features over specified month buckets using SUM aggregation.
"""

from .utils import get_repo_root, bucketize_sum_month_lags

__all__ = ["get_repo_root", "bucketize_sum_month_lags"]