# milan-traffic-forecasting

Internet traffic forecasting over Milan's 10,000-square grid using the
[Telecom Italia Big Data Challenge](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV) dataset (Nov 2013 – Jan 2014, 62 daily files).

## Project structure

```
src/
  loader.py   — one-time ingestion pipeline (raw → Parquet)
  data.py     — canonical load function for all later analysis
data/
  raw/        — staging area (files deleted after processing)
  processed/
    milan_internet_traffic.parquet
notes/
  loader_memory_benchmark.md
```

## Data pipeline

### Step 1 — run once
```bash
python src/loader.py
```
Reads all 62 raw `.txt` files one at a time, aggregates internet traffic per
`(square_id, time_interval)`, and writes a single Parquet file.

| | Size |
|---|---|
| Raw `.txt` files (62 days) | 20.8 GB |
| `milan_internet_traffic.parquet` | 408.8 MB |
| Reduction | 98% (50.9× smaller) |

### Step 2 — all later scripts and notebooks
```python
from src.data import load_traffic
df = load_traffic()   # 89M rows, 2.4 s, never touches raw files
```

## Memory strategy

- `loader.py` uses `usecols`, `dtype` downcasting (int32/float32), and immediate
  groupby aggregation to keep each file's retained footprint to ~23 MB.
- Raw files are deleted from `data/raw/` immediately after processing so only
  one 322 MB file lives on disk at a time during ingestion.
- The Parquet file is the single source of truth for Sections 2–4.
  Raw files are never read again after ingestion.
