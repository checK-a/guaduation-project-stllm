# CDC ILI Data Minimal Checklist

## Required inputs

1. CDC ILINet weekly state data
   - Preferred manual source: CDC FluView Interactive
   - Page: https://www.cdc.gov/fluview/overview/fluview-interactive.html
   - Data app: https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html
   - Scope:
     - Geography: all 50 states + Washington, DC
     - Time range: 2013W40 to 2023W40
     - Metric: weighted ILI percentage
   - Required fields after download or API pull:
     - `REGION`
     - `YEAR`
     - `WEEK`
     - `% WEIGHTED ILI`

2. U.S. state adjacency
   - Default source used by the fetch script: GitHub gist adjacency JSON
   - Gist page: https://gist.github.com/Glench/3906059
   - The script fetches gist metadata from GitHub API, then downloads the raw `adjacent_us_states.json`.
   - If GitHub download fails, the script falls back to the same adjacency semantics embedded locally in the repo.

## Practical note on automation

- CDC FluView Interactive is the authoritative public source.
- The repo fetch script pulls weekly state ILI programmatically from Delphi Epidata, which documents that its `fluview` endpoint is sourced from CDC FluView:
  - Docs: https://cmu-delphi.github.io/delphi-epidata/api/fluview.html
  - API: https://api.delphi.cmu.edu/epidata/fluview/
- This is a pragmatic automation path because the CDC interactive export is browser-oriented rather than a stable public static file URL.

## Outputs produced by the fetch script

- `dataset/ili_us_states/raw/cdc_ili_states_*.csv`
- `dataset/ili_us_states/raw/us_state_adjacency_edges.csv`
- `dataset/ili_us_states/raw/fetch_manifest.json`

## Fallback behavior

- The fetch script tries the GitHub adjacency gist first.
- If GitHub returns an error or is unreachable, the script falls back to a built-in U.S. state adjacency table and records that choice in `fetch_manifest.json`.
- This fallback keeps the data pipeline runnable in restricted network environments.

## Next step after raw fetch

- Build the training package with:

```bash
python prepare_cdc_ili.py --ili_csv <raw_ili_csv> --adj_csv <adj_edges_csv>
```
