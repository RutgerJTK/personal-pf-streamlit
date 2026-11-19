"""
Live data ingestion from AWS S3.

Reads ``cleaned/portfolio_transactions.csv`` with the read-only IAM user whose
keys live in Streamlit secrets:

    [aws]
    aws_access_key_id     = "AKIA..."
    aws_secret_access_key = "..."
    bucket_name           = "personal-pf-streamlit-s3-bucket"
    region_name           = "eu-central-1"   # optional but recommended

Only call load_transactions() from pages that run after require_login().
"""
from __future__ import annotations

import io

import boto3
import pandas as pd
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

CLEANED_KEY = "cleaned/portfolio_transactions.csv"
EXPECTED_COLUMNS = ["datetime", "isin", "action", "quantity", "net_amount"]
ALLOWED_ACTIONS = {"BUY", "SELL", "DIVIDEND", "DEPOSIT", "CASH"}


def parse_transactions(raw: bytes | str) -> pd.DataFrame:
    """Parse + validate the cleaned CSV. Pure function (no Streamlit, no AWS)."""
    buf = io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(raw)
    df = pd.read_csv(buf, keep_default_na=False, dtype={"isin": str})
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    df["datetime"] = pd.to_datetime(df["datetime"], format="%Y-%m-%d %H:%M:%S")
    df["quantity"] = df["quantity"].astype(float)
    df["net_amount"] = df["net_amount"].astype(float)
    bad = set(df["action"].unique()) - ALLOWED_ACTIONS
    if bad:
        raise ValueError(f"Unexpected actions: {bad}")
    return df.sort_values("datetime", kind="stable").reset_index(drop=True)


@st.cache_resource
def _s3_client():
    cfg = st.secrets["aws"]
    return boto3.client(
        "s3",
        aws_access_key_id=cfg["aws_access_key_id"],
        aws_secret_access_key=cfg["aws_secret_access_key"],
        region_name=cfg.get("region_name"),
    )


@st.cache_data(ttl=3600, show_spinner="Loading transactions…")
def load_transactions() -> pd.DataFrame:
    """Fetch the cleaned transactions. Raises on failure (callers show a message)."""
    cfg = st.secrets["aws"]
    obj = _s3_client().get_object(Bucket=cfg["bucket_name"], Key=CLEANED_KEY)
    return parse_transactions(obj["Body"].read())


def load_transactions_or_stop() -> pd.DataFrame:
    """Page helper: load the data, or show a short error and halt the page."""
    try:
        return load_transactions()
    except (KeyError, FileNotFoundError):
        print("[s3] [aws] secrets missing or incomplete")
        st.error("AWS settings are missing. Check the app's secrets.")
    except (ClientError, BotoCoreError) as exc:
        print(f"[s3] fetch failed: {type(exc).__name__}: {exc}")  # server log only
        st.error("Could not load data from S3. Check the bucket, key and IAM permissions.")
    except ValueError as exc:
        print(f"[s3] bad data: {exc}")
        st.error("The cleaned file on S3 does not have the expected format. Re-run the pipeline.")
    st.stop()


def refresh() -> None:
    load_transactions.clear()