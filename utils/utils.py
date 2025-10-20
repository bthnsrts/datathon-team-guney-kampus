from pathlib import Path
import duckdb
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score


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

def bucketize_sum_month_lags(    
    table: pd.DataFrame,
    value_col: str,
    buckets: Dict[str, Tuple[int, int]],
    *,
    cust_col: str = "cust_id",
    date_col: str = "date",
    ref_col: str = "ref_date",                      
):
    """
    Create disjoint lagged features over whole-month differences (ref_date - date),
    returning one row per (cust_id, ref_date) with columns like <value_col>_<bucketlabel>.

    buckets: dict label -> (lo_months, hi_months), inclusive, non-overlapping.
             Example: {"l1m": (1,1), "l2m": (2,2), "l5m": (3,5), "l8m": (6,8), ...}
    """

    con = duckdb.connect(":memory:")
    con.register("df", table) 

    proj_parts = []
    for label, (lo, hi) in buckets.items():
        alias = f"{value_col}_{label}"        
        # Outside bucket contributes 0; inside bucket use 0 if value is NULL (so sum is numeric)
        expr = (
            f"SUM(CASE WHEN months_diff BETWEEN {lo} AND {hi} "
            f"THEN COALESCE({value_col}, 0) ELSE 0 END) AS {alias}"
        )    
        proj_parts.append(expr)

    proj_sql = ",\n       ".join(proj_parts)

    sql = f"""
    WITH typed AS (
        SELECT
            {cust_col}      AS cust_id,
            CAST({date_col} AS DATE)     AS dt,
            CAST({ref_col}  AS DATE)     AS ref_dt,
            {value_col}
        FROM df
    ),
    with_lag AS (
        SELECT
            cust_id,
            {value_col},
            ref_dt,
            date_diff('month', dt, ref_dt) AS months_diff
        FROM typed
        WHERE dt <= ref_dt
    )
    SELECT
        cust_id,
        ref_dt AS ref_date,
        {proj_sql}
    FROM with_lag
    GROUP BY cust_id, ref_date
    ORDER BY cust_id, ref_date
    """

    return con.execute(sql).df()
