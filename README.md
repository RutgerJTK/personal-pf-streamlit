# personal-pf-streamlit

A password-protected Streamlit dashboard for tracking my personal investment
portfolio and watchlist, replacing a legacy Google Sheets setup.

## How it works

1. DEGIRO exports are cleaned and anonymized locally (no PII, no broker metadata).
2. The cleaned data is uploaded to a private AWS S3 bucket.
3. The Streamlit app fetches the cleaned data from S3 with read-only credentials,
   after the user has authenticated.

## Repository layout

- `infra/`: documentation of the AWS setup (IAM policies)
- `cleaning-pipeline/`: local cleaning and upload scripts (coming soon)
- `streamlit-app/`: the dashboard (coming soon)
- `shared-config/`: static config such as the ISIN to ticker map (coming soon)

## Security notes

- The S3 bucket is private with all public access blocked.
- The app uses a dedicated IAM user that can only read the `cleaned/` prefix.
- Secrets live in `.streamlit/secrets.toml` locally and in Streamlit Cloud's
  secrets manager when deployed. They are never committed.
- Raw broker exports never leave the local machine except to the private `raw/` prefix.

## Status

Work in progress. Infrastructure is set up; application code is next.