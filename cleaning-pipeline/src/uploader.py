"""
S3 uploader
===========

Pushes two things to your private bucket, using YOUR local AWS credentials
(not the read-only key that the Streamlit app uses):

    raw/<YYYY-MM-DD>/Account.csv         dated archive of the unedited exports
    raw/<YYYY-MM-DD>/Transactions.csv
    cleaned/portfolio_transactions.csv   the anonymised file the app reads

Guards: the cleaned file must have exactly the five expected columns, and
nothing from raw_data/ can ever be written under cleaned/.

After a real (non dry-run) upload, the raw exports in raw_data/ are checked
against their S3 copies (same size) and then deleted locally, so S3 is the
only place they live. Pass --keep-local to keep them.

Credentials come from the normal boto3 chain (env vars, ~/.aws, AWS_PROFILE).

Usage:
    python cleaning-pipeline/src/uploader.py --dry-run
    python cleaning-pipeline/src/uploader.py
    python cleaning-pipeline/src/uploader.py --bucket my-bucket --profile pf-admin
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "cleaning-pipeline" / "raw_data"
DEFAULT_CLEAN_FILE = REPO_ROOT / "cleaning-pipeline" / "output" / "portfolio_transactions.csv"
DEFAULT_BUCKET = "personal-pf-streamlit-s3-bucket"  # matches infra/iam/reader-policy.json

RAW_FILES = ("Account.csv", "Transactions.csv")
CLEANED_KEY = "cleaned/portfolio_transactions.csv"
EXPECTED_COLUMNS = ["datetime", "isin", "action", "quantity", "net_amount"]
ALLOWED_ACTIONS = {"BUY", "SELL", "DIVIDEND", "DEPOSIT", "FEE", "CASH"}


class UploadError(Exception):
    pass


def validate_cleaned_file(path: Path) -> int:
    """Return the row count, or raise if the file is not the anonymised format."""
    if not path.exists():
        raise UploadError(f"{path} not found. Run cleaner.py first.")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != EXPECTED_HEADER:
            raise UploadError(f"Refusing to upload: header is {header}, expected {EXPECTED_HEADER}")
        rows = 0
        for i, row in enumerate(reader, start=2):
            if len(row) != len(EXPECTED_HEADER) or row[2] not in ALLOWED_ACTIONS:
                raise UploadError(f"Refusing to upload: malformed row at line {i}: {row}")
            rows += 1
    if rows == 0:
        raise UploadError("Refusing to upload an empty cleaned file")
    return rows


def _put(s3, bucket: str, key: str, path: Path, dry_run: bool) -> None:
    size_kb = path.stat().st_size / 1024
    if dry_run:
        print(f"  [dry-run] {path.name} ({size_kb:.1f} KB) -> s3://{bucket}/{key}")
        return
    s3.upload_file(
        str(path), bucket, key,
        ExtraArgs={"ContentType": "text/csv", "ServerSideEncryption": "AES256"},
    )
    print(f"  uploaded {path.name} ({size_kb:.1f} KB) -> s3://{bucket}/{key}")


def remove_local_raw(s3, bucket: str, uploads: list[tuple[Path, str]]) -> None:
    """Delete each local raw file, but only once S3 holds a copy of the same size."""
    for path, key in uploads:
        remote = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        local = path.stat().st_size
        if remote != local:
            raise UploadError(f"s3://{bucket}/{key} is {remote} bytes, local {path.name} is {local}; "
                              "kept the local file")
    for path, _ in uploads:
        path.unlink()
        print(f"  removed local {path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Upload raw archive + cleaned file to S3")
    p.add_argument("--bucket", default=os.environ.get("PF_BUCKET", DEFAULT_BUCKET))
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--clean-file", type=Path, default=DEFAULT_CLEAN_FILE)
    p.add_argument("--skip-raw", action="store_true", help="only upload the cleaned file")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-local", action="store_true",
                   help="keep the raw exports in raw_data/ after uploading (default: delete them)")
    args = p.parse_args(argv)

    try:
        rows = validate_cleaned_file(args.clean_file)
        raw_paths = [] if args.skip_raw else [args.raw_dir / name for name in RAW_FILES]
        for rp in raw_paths:
            if not rp.exists():
                raise UploadError(f"{rp} not found (use --skip-raw to upload only the cleaned file)")

        session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
        s3 = session.client("s3")
        if not args.dry_run:
            s3.head_bucket(Bucket=args.bucket)  # fails early on bad credentials / name

        print(f"Bucket: {args.bucket}   cleaned rows: {rows}   dry-run: {args.dry_run}")
        stamp = date.today().isoformat()
        uploads = [(rp, f"raw/{stamp}/{rp.name}") for rp in raw_paths]
        for rp, key in uploads:
            _put(s3, args.bucket, key, rp, args.dry_run)
        _put(s3, args.bucket, CLEANED_KEY, args.clean_file, args.dry_run)
        if uploads and not args.dry_run and not args.keep_local:
            remove_local_raw(s3, args.bucket, uploads)
        elif uploads and args.dry_run and not args.keep_local:
            print("  [dry-run] would remove the local raw exports after upload")
    except UploadError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except (ClientError, BotoCoreError) as exc:
        print(f"\nAWS ERROR: {exc}", file=sys.stderr)
        return 2

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())