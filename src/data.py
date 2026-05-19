"""
Data loading & cleaning.

The notebook will CALL these functions. It won't redefine them.
That way the cleaning your model trained on is byte-for-byte the
same cleaning your demo app uses. Mismatch here = the #1 cause of
"works in notebook, broken in production" — a classic interview topic.
"""
import pandas as pd

from src import config


def load_raw_loans() -> pd.DataFrame:
    """Load the raw Lending Club CSV exactly as downloaded."""
    if not config.RAW_LOANS_CSV.exists():
        raise FileNotFoundError(
            f"Expected raw data at {config.RAW_LOANS_CSV}. "
            "See README section 'Getting the dataset'."
        )
    # low_memory=False: Lending Club has mixed-type columns;
    # this avoids pandas' chunked-inference warnings.
    return pd.read_csv(config.RAW_LOANS_CSV, low_memory=False)


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the binary target from loan_status.

    CREDIT-RISK LEAKAGE WARNING (say this in interviews):
    'Current' loans have no known outcome yet — including them
    leaks the future. We keep only RESOLVED loans.
    """
    resolved_bad = {
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
    }
    resolved_good = {
        "Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid",
    }
    df = df[df["loan_status"].isin(resolved_bad | resolved_good)].copy()
    df[config.TARGET_COL] = df["loan_status"].isin(resolved_bad).astype(int)
    return df.drop(columns=["loan_status"])


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that would NOT exist at loan-application time.

    e.g. 'total_pymnt', 'recoveries' — these are only known AFTER
    the loan plays out. A model using them scores 0.99 AUC and is
    completely useless in reality. Catching this is what separates
    a real candidate from someone who just ran sklearn.
    """
    post_origination = [
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
        "total_rec_int", "total_rec_late_fee", "recoveries",
        "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
        "next_pymnt_d", "out_prncp", "out_prncp_inv",
    ]
    return df.drop(columns=[c for c in post_origination if c in df.columns])
