"""
Sanity tests on the data pipeline.

These guard the two failure modes that matter most in credit modeling:
the target must be clean binary, and known leakage columns must never
re-enter the feature set.
"""
import pandas as pd
from src import data


def test_target_is_binary():
    df = pd.DataFrame({"loan_status": ["Fully Paid", "Charged Off"]})
    out = data.build_target(df)
    assert set(out["default"].unique()).issubset({0, 1})


def test_leakage_columns_removed():
    df = pd.DataFrame({
        "loan_amnt": [1000], "total_pymnt": [500],
        "recoveries": [0], "out_prncp": [0],
    })
    out = data.drop_leakage_columns(df)
    for col in ["total_pymnt", "recoveries", "out_prncp"]:
        assert col not in out.columns


def test_loan_amnt_preserved():
    df = pd.DataFrame({"loan_amnt": [1000], "total_pymnt": [500]})
    out = data.drop_leakage_columns(df)
    assert "loan_amnt" in out.columns
