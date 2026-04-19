import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


STATE_TO_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}

STATE_FIPS_TO_ABBR = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

ABBR_TO_STATE = {abbr: state for state, abbr in STATE_TO_ABBR.items()}

ILI_API_URL = "https://api.delphi.cmu.edu/epidata/fluview/"
GITHUB_GIST_API_URL = "https://api.github.com/gists/3906059"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
FALLBACK_STATE_ADJACENCY = {
    "WA": ["ID", "OR"],
    "DE": ["MD", "PA", "NJ"],
    "DC": ["MD", "VA"],
    "WI": ["MI", "MN", "IA", "IL"],
    "WV": ["OH", "PA", "MD", "VA", "KY"],
    "HI": [],
    "FL": ["AL", "GA"],
    "WY": ["MT", "SD", "NE", "CO", "UT", "ID"],
    "NH": ["VT", "ME", "MA"],
    "NJ": ["DE", "PA", "NY"],
    "NM": ["AZ", "UT", "CO", "OK", "TX"],
    "TX": ["NM", "OK", "AR", "LA"],
    "LA": ["TX", "AR", "MS"],
    "NC": ["VA", "TN", "GA", "SC"],
    "ND": ["MN", "SD", "MT"],
    "NE": ["SD", "IA", "MO", "KS", "CO", "WY"],
    "TN": ["KY", "VA", "NC", "GA", "AL", "MS", "AR", "MO"],
    "NY": ["NJ", "PA", "VT", "MA", "CT"],
    "PA": ["NY", "NJ", "DE", "MD", "WV", "OH"],
    "CA": ["OR", "NV", "AZ"],
    "NV": ["ID", "UT", "AZ", "CA", "OR"],
    "VA": ["NC", "TN", "KY", "WV", "MD", "DC"],
    "CO": ["WY", "NE", "KS", "OK", "NM", "AZ", "UT"],
    "AK": [],
    "AL": ["MS", "TN", "GA", "FL"],
    "AR": ["MO", "TN", "MS", "LA", "TX", "OK"],
    "VT": ["NY", "NH", "MA"],
    "IL": ["IN", "KY", "MO", "IA", "WI"],
    "GA": ["FL", "AL", "TN", "NC", "SC"],
    "IN": ["MI", "OH", "KY", "IL"],
    "IA": ["MN", "WI", "IL", "MO", "NE", "SD"],
    "OK": ["KS", "MO", "AR", "TX", "NM", "CO"],
    "AZ": ["CA", "NV", "UT", "CO", "NM"],
    "ID": ["MT", "WY", "UT", "NV", "OR", "WA"],
    "CT": ["NY", "MA", "RI"],
    "ME": ["NH"],
    "MD": ["VA", "WV", "PA", "DC", "DE"],
    "MA": ["RI", "CT", "NY", "NH", "VT"],
    "OH": ["PA", "WV", "KY", "IN", "MI"],
    "UT": ["ID", "WY", "CO", "NM", "AZ", "NV"],
    "MO": ["IA", "IL", "KY", "TN", "AR", "OK", "KS", "NE"],
    "MN": ["WI", "IA", "SD", "ND"],
    "MI": ["WI", "IN", "OH"],
    "RI": ["CT", "MA"],
    "KS": ["NE", "MO", "OK", "CO"],
    "MT": ["ND", "SD", "WY", "ID"],
    "MS": ["LA", "AR", "TN", "AL"],
    "SC": ["GA", "NC"],
    "KY": ["IN", "OH", "WV", "VA", "TN", "MO", "IL"],
    "OR": ["CA", "NV", "ID", "WA"],
    "SD": ["ND", "MN", "IA", "NE", "WY", "MT"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch raw weekly US state ILI data and state adjacency edges."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/ili_us_states/raw",
        help="Directory where raw fetched files will be written.",
    )
    parser.add_argument("--start_epiweek", type=str, default="2013W40")
    parser.add_argument("--end_epiweek", type=str, default="2023W40")
    parser.add_argument(
        "--include_dc",
        action="store_true",
        default=True,
        help="Include Washington, DC in both ILI output and adjacency edges.",
    )
    parser.add_argument(
        "--skip_prepare",
        action="store_true",
        help="Only fetch raw inputs and do not call prepare_cdc_ili.py.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ili_us_states",
        help="Dataset name passed to prepare_cdc_ili.py if prepare is run.",
    )
    return parser.parse_args()


def epiweek_to_delphi(value):
    if len(value) != 7 or value[4] != "W":
        raise ValueError(f"Invalid epiweek format: {value}. Expected YYYYWww.")
    year = value[:4]
    week = value[5:]
    if not (year.isdigit() and week.isdigit()):
        raise ValueError(f"Invalid epiweek format: {value}.")
    return f"{year}{int(week):02d}"


def delphi_to_year_week(epiweek):
    text = str(int(epiweek))
    year = int(text[:4])
    week = int(text[4:])
    return year, week


def fetch_json(url, params):
    query = urllib.parse.urlencode(params)
    request_url = f"{url}?{query}"
    payload = download_text(
        request_url,
        label="ILI weekly data",
        referer="https://cmu-delphi.github.io/delphi-epidata/api/fluview.html",
        accept="application/json,text/plain,*/*",
    )
    return json.loads(payload), request_url


def fetch_ili_data(start_epiweek, end_epiweek, include_dc):
    regions = sorted(
        abbr.lower()
        for abbr in STATE_TO_ABBR.values()
        if include_dc or abbr != "DC"
    )
    params = {
        "regions": ",".join(regions),
        "epiweeks": f"{epiweek_to_delphi(start_epiweek)}-{epiweek_to_delphi(end_epiweek)}",
    }
    payload, request_url = fetch_json(ILI_API_URL, params)
    if payload.get("result") != 1:
        raise RuntimeError(f"ILI API request failed: {payload}")

    rows = []
    for item in payload.get("epidata", []):
        region_abbr = str(item["region"]).upper()
        if region_abbr not in ABBR_TO_STATE:
            continue
        year, week = delphi_to_year_week(item["epiweek"])
        rows.append(
            {
                "REGION": ABBR_TO_STATE[region_abbr],
                "REGION_ABBR": region_abbr,
                "YEAR": year,
                "WEEK": week,
                "EPIWEEK": f"{year:04d}W{week:02d}",
                "% WEIGHTED ILI": item.get("wili"),
                "% UNWEIGHTED ILI": item.get("ili"),
                "ISSUE": item.get("issue"),
                "RELEASE_DATE": item.get("release_date"),
                "NUM_PROVIDERS": item.get("num_providers"),
                "NUM_PATIENTS": item.get("num_patients"),
                "NUM_ILI": item.get("num_ili"),
                "SOURCE": "Delphi Epidata fluview (sourced from CDC FluView)",
            }
        )

    if not rows:
        raise RuntimeError("ILI API returned no rows.")

    ili_df = pd.DataFrame(rows).sort_values(["YEAR", "WEEK", "REGION_ABBR"]).reset_index(drop=True)
    return ili_df, request_url


def fetch_github_adjacency_data():
    metadata_text = download_text(
        GITHUB_GIST_API_URL,
        label="GitHub adjacency gist metadata",
        referer="https://gist.github.com/Glench/3906059",
        accept="application/vnd.github+json,application/json,text/plain,*/*",
    )
    metadata = json.loads(metadata_text)
    files = metadata.get("files", {})
    if "adjacent_us_states.json" not in files:
        raise RuntimeError("GitHub gist metadata did not contain adjacent_us_states.json")

    raw_url = files["adjacent_us_states.json"].get("raw_url")
    if not raw_url:
        raise RuntimeError("GitHub gist metadata did not expose a raw_url")

    adjacency_json_text = download_text(
        raw_url,
        label="GitHub adjacency gist raw JSON",
        referer="https://gist.github.com/Glench/3906059",
        accept="application/json,text/plain,*/*",
    )
    return json.loads(adjacency_json_text), raw_url


def download_text(url, label, referer=None, accept="*/*"):
    print(f"[Download] Starting {label}")
    print(f"[Download] URL: {url}")
    try:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Accept": accept,
                **({"Referer": referer} if referer else {}),
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            total = response.headers.get("Content-Length")
            total = int(total) if total and total.isdigit() else None
            chunks = []
            downloaded = 0
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    print(
                        f"\r[Download] {label}: {downloaded}/{total} bytes ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )
            if total:
                print()
            else:
                print(f"[Download] {label}: {downloaded} bytes")
            print(f"[Download] Completed {label} via urllib")
            return b"".join(chunks).decode("utf-8")
    except urllib.error.URLError:
        curl_path = shutil.which("curl.exe") or shutil.which("curl")
        if not curl_path:
            raise
        print(f"[Download] urllib failed for {label}, falling back to curl")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            cmd = [
                curl_path,
                "-fL",
                "-A",
                DEFAULT_USER_AGENT,
                "--progress-bar",
                "-o",
                str(tmp_path),
            ]
            if referer:
                cmd.extend(["-e", referer])
            if accept:
                cmd.extend(["-H", f"Accept: {accept}"])
            cmd.append(url)
            subprocess.run(cmd, check=True)
            text = tmp_path.read_text(encoding="utf-8")
            print(f"[Download] Completed {label} via curl")
            return text
        finally:
            tmp_path.unlink(missing_ok=True)


def build_state_edges_from_census(raw_text, include_dc):
    from io import StringIO

    county_df = pd.read_csv(StringIO(raw_text), sep="|", dtype=str)
    county_df = county_df.rename(columns=lambda c: str(c).strip())
    county_df["County GEOID"] = county_df["County GEOID"].fillna("").str.zfill(5)
    county_df["Neighbor GEOID"] = county_df["Neighbor GEOID"].fillna("").str.zfill(5)
    county_df["state_a"] = county_df["County GEOID"].str[:2].map(STATE_FIPS_TO_ABBR)
    county_df["state_b"] = county_df["Neighbor GEOID"].str[:2].map(STATE_FIPS_TO_ABBR)
    county_df = county_df.dropna(subset=["state_a", "state_b"])
    county_df = county_df.loc[county_df["state_a"] != county_df["state_b"]].copy()

    if not include_dc:
        county_df = county_df.loc[
            (county_df["state_a"] != "DC") & (county_df["state_b"] != "DC")
        ].copy()

    edge_df = county_df[["state_a", "state_b"]].drop_duplicates().copy()
    edge_df["pair"] = edge_df.apply(
        lambda row: tuple(sorted((row["state_a"], row["state_b"]))), axis=1
    )
    edge_df = edge_df.drop_duplicates(subset=["pair"]).drop(columns="pair")
    edge_df = edge_df.sort_values(["state_a", "state_b"]).reset_index(drop=True)
    edge_df["state_name"] = edge_df["state_a"].map(ABBR_TO_STATE)
    edge_df["neighbor_name"] = edge_df["state_b"].map(ABBR_TO_STATE)
    edge_df = edge_df[
        ["state_name", "state_a", "neighbor_name", "state_b"]
    ].rename(columns={"state_a": "state_abbr", "state_b": "neighbor_abbr"})
    return edge_df


def build_state_edges_from_fallback(include_dc):
    rows = []
    seen = set()
    for state_abbr, neighbors in FALLBACK_STATE_ADJACENCY.items():
        if not include_dc and state_abbr == "DC":
            continue
        for neighbor_abbr in neighbors:
            if not include_dc and neighbor_abbr == "DC":
                continue
            pair = tuple(sorted((state_abbr, neighbor_abbr)))
            if pair in seen:
                continue
            seen.add(pair)
            rows.append(
                {
                    "state_name": ABBR_TO_STATE[state_abbr],
                    "state_abbr": state_abbr,
                    "neighbor_name": ABBR_TO_STATE[neighbor_abbr],
                    "neighbor_abbr": neighbor_abbr,
                }
            )
    return pd.DataFrame(rows).sort_values(["state_abbr", "neighbor_abbr"]).reset_index(drop=True)


def build_state_edges_from_github(adjacency_data, include_dc):
    rows = []
    seen = set()
    for state_abbr, neighbors in adjacency_data.items():
        state_abbr = str(state_abbr).upper()
        if state_abbr not in ABBR_TO_STATE:
            continue
        if not include_dc and state_abbr == "DC":
            continue
        for neighbor_abbr in neighbors:
            neighbor_abbr = str(neighbor_abbr).upper()
            if neighbor_abbr not in ABBR_TO_STATE:
                continue
            if not include_dc and neighbor_abbr == "DC":
                continue
            pair = tuple(sorted((state_abbr, neighbor_abbr)))
            if pair in seen:
                continue
            seen.add(pair)
            rows.append(
                {
                    "state_name": ABBR_TO_STATE[state_abbr],
                    "state_abbr": state_abbr,
                    "neighbor_name": ABBR_TO_STATE[neighbor_abbr],
                    "neighbor_abbr": neighbor_abbr,
                }
            )
    if not rows:
        raise RuntimeError("GitHub adjacency JSON produced no usable state edges")
    return pd.DataFrame(rows).sort_values(["state_abbr", "neighbor_abbr"]).reset_index(drop=True)


def run_prepare(output_dir, ili_csv_path, adj_csv_path, dataset_name, start_epiweek, end_epiweek):
    cmd = [
        sys.executable,
        "prepare_cdc_ili.py",
        "--ili_csv",
        str(ili_csv_path),
        "--adj_csv",
        str(adj_csv_path),
        "--dataset_name",
        dataset_name,
        "--start_epiweek",
        start_epiweek,
        "--end_epiweek",
        end_epiweek,
    ]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adjacency_source_info = None

    try:
        print("[Stage 1/2] Fetching weekly state ILI data")
        ili_df, ili_request_url = fetch_ili_data(
            start_epiweek=args.start_epiweek,
            end_epiweek=args.end_epiweek,
            include_dc=args.include_dc,
        )
        print("[Stage 2/2] Fetching GitHub state adjacency data")
        try:
            adjacency_data, raw_url = fetch_github_adjacency_data()
            edge_df = build_state_edges_from_github(adjacency_data, include_dc=args.include_dc)
            adjacency_source_info = {
                "kind": "GitHub gist adjacency JSON",
                "request_url": GITHUB_GIST_API_URL,
                "raw_url": raw_url,
            }
            print("[Stage 2/2] Completed adjacency build from GitHub source")
        except Exception as exc:
            print(f"[Stage 2/2] GitHub adjacency download failed: {exc}")
            print("[Stage 2/2] Falling back to built-in state adjacency table")
            edge_df = build_state_edges_from_fallback(include_dc=args.include_dc)
            adjacency_source_info = {
                "kind": "Built-in fallback state adjacency table",
                "source_note": "Fallback used because GitHub adjacency download returned an error in this environment.",
            }
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network fetch failed: {exc}") from exc

    ili_filename = f"cdc_ili_states_{args.start_epiweek}_{args.end_epiweek}.csv"
    ili_csv_path = output_dir / ili_filename
    adj_csv_path = output_dir / "us_state_adjacency_edges.csv"

    ili_df.to_csv(ili_csv_path, index=False)
    edge_df.to_csv(adj_csv_path, index=False)

    manifest = {
        "ili_source": {
            "kind": "Delphi Epidata fluview",
            "source_data": "CDC FluView",
            "request_url": ili_request_url,
        },
        "adjacency_source": adjacency_source_info,
        "time_range": {
            "start_epiweek": args.start_epiweek,
            "end_epiweek": args.end_epiweek,
        },
        "files": {
            "ili_csv": str(ili_csv_path),
            "adj_csv": str(adj_csv_path),
        },
        "counts": {
            "ili_rows": int(len(ili_df)),
            "state_edges": int(len(edge_df)),
        },
    }
    with open(output_dir / "fetch_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote ILI CSV: {ili_csv_path}")
    print(f"Wrote adjacency CSV: {adj_csv_path}")
    print(f"ILI rows: {len(ili_df)}")
    print(f"State edges: {len(edge_df)}")

    if not args.skip_prepare:
        print("[Stage 3/3] Building train/val/test package with prepare_cdc_ili.py")
        run_prepare(
            output_dir=output_dir,
            ili_csv_path=ili_csv_path,
            adj_csv_path=adj_csv_path,
            dataset_name=args.dataset_name,
            start_epiweek=args.start_epiweek,
            end_epiweek=args.end_epiweek,
        )
        print("[Stage 3/3] Completed dataset packaging")


if __name__ == "__main__":
    main()
