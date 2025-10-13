import pandas.testing as pdt
import pandas as pd

from utils import bucketize_sum_month_lags, get_repo_root

INPUT_PATH = str(get_repo_root()) + "/tests/data/bucketize_sum_month_lags_input.csv"
EXPECTED_PATH = str(get_repo_root()) + "/tests/data/bucketize_sum_month_lags_expected.csv"

BUCKETS = {
    "l0m": (0, 0),
    "l2m": (1, 2),
    "l5m": (3, 5),
    "l8m": (6, 8),
    "l1year": (9, 12),
}

def _normalize_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make result & expected comparable:
    - cast dates to string (CSV stores them as 'YYYY-MM-DD')
    - ensure column order is stable (cust_id, ref_date, then the feature columns)
    - sort rows deterministically
    """
    out = df.copy()

    # 1) Make sure the identifier columns exist as expected
    assert "cust_id" in out.columns, "Expected 'cust_id' in output"
    assert "ref_date" in out.columns, "Expected 'ref_date' in output"

    # 2) Dates as ISO strings so CSV ↔ DataFrame comparisons are apples-to-apples
    out["ref_date"] = pd.to_datetime(out["ref_date"]).dt.date.astype(str)

    # 3) Stable column order: id cols first, then feature columns (sorted)
    id_cols = ["cust_id", "ref_date"]
    feat_cols = [c for c in out.columns if c not in id_cols]
    feat_cols_sorted = sorted(feat_cols)
    out = out[id_cols + feat_cols_sorted]

    # 4) Stable row order
    out = out.sort_values(id_cols).reset_index(drop=True)

    return out

def test_bucketize_sum_month_lags_matches_expected():
    # --- Arrange ---
    # Read input fixture; parse dates so your function gets clean inputs
    df_in = pd.read_csv(
        INPUT_PATH,
        dtype={"cust_id": "Int64"},
        parse_dates=["date", "ref_date"],
    )
    
    # Read expected fixture (kept as CSV for easy diffing in PRs)
    df_expected = pd.read_csv(
        EXPECTED_PATH,
        dtype={"cust_id": "Int64"},
    )

    # --- Act ---
    df_out = bucketize_sum_month_lags(
        table=df_in,
        value_col="cc_transaction_all_cnt",
        buckets=BUCKETS,
        # (cust_col/date_col/ref_col defaults match your function signature)
    )

    # --- Assert ---
    # Normalize both sides (dates to strings, stable ordering)
    df_out_norm = _normalize_for_compare(df_out)
    df_exp_norm = _normalize_for_compare(df_expected)

    # Ensure same columns (exact set & order)
    assert list(df_out_norm.columns) == list(df_exp_norm.columns), (
        f"Output columns differ.\nGot: {list(df_out_norm.columns)}\n"
        f"Exp: {list(df_exp_norm.columns)}"
    )

    # Compare values (dtype differences are fine; NaNs considered equal)
    pdt.assert_frame_equal(
        df_out_norm, df_exp_norm,
        check_dtype=False,
        check_like=False,
    )