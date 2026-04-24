import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import requests


SOURCE_URL = "https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD"
SOURCE_PAGE_URL = "https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/g62h-syeh"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fetch the official HHS state-level COVID hospital timeseries CSV."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="us_states_hhs_covid_admissions_raw",
        help="Dataset directory name under output_root.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="dataset",
        help="Root directory under which raw files will be written.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="hhs_state_timeseries_raw.csv",
        help="Filename for the downloaded CSV.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds.",
    )
    return parser


def fetch_csv(timeout: int) -> bytes:
    response = requests.get(
        SOURCE_URL,
        timeout=timeout,
        headers={"User-Agent": "ST-LLM-Plus-HHS-Fetch/1.0"},
    )
    response.raise_for_status()
    return response.content


def main():
    args = build_parser().parse_args()

    dataset_root = Path(args.output_root).resolve() / args.dataset_name
    raw_dir = ensure_dir(dataset_root / "raw")
    output_path = raw_dir / args.output_filename

    csv_bytes = fetch_csv(args.timeout)
    output_path.write_bytes(csv_bytes)

    manifest = {
        "source_name": "HHS/HealthData.gov state timeseries hospital dataset",
        "source_url": SOURCE_URL,
        "source_page_url": SOURCE_PAGE_URL,
        "fetch_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_filename": output_path.name,
        "selected_metric_definition": (
            "daily_hospital_admissions = "
            "previous_day_admission_adult_covid_confirmed + "
            "previous_day_admission_pediatric_covid_confirmed"
        ),
    }
    with open(raw_dir / "fetch_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Downloaded raw CSV to: {output_path}")
    print(f"Saved manifest to: {raw_dir / 'fetch_manifest.json'}")


if __name__ == "__main__":
    main()
