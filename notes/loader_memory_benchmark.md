# Loader Memory Benchmark

## Parquet vs Raw size comparison

| | Size |
|---|---|
| Raw `.txt` files (62 days) | 20.805 GB |
| `milan_internet_traffic.parquet` | 408.79 MB |
| Reduction | 98.0% (50.9× smaller) |

Parquet reload: 89,245,318 rows × 3 columns in 2.82 s, 1,428 MB in-memory.


**File tested:** `data/raw/sms-call-internet-mi-2013-11-01.txt`

## Results

| Method         | Peak Memory (MB) | Rows      |
|----------------|-----------------|-----------|
| Naive load     | 348.95 MB       | 4,842,625 |
| Optimized load | 355.99 MB       | 1,439,982 |
| Reduction      | -2.0%           | —         |

## Full-pipeline run (all 62 files)

| Metric | Value |
|---|---|
| Files processed | 62 |
| Total rows | 89,245,318 |
| Per-file retained memory | ~23 MB |
| Combined DataFrame (retained) | 1,427.93 MB |
| Peak memory (whole loop, tracemalloc) | 2,857.20 MB |
| Total time | 298.2 s (~5 min) |
| Output | `data/processed/milan_internet_traffic.parquet` |

Note: peak of 2,857 MB reflects the growing `frames` list accumulating all 62
aggregated DataFrames in memory before `pd.concat`. The per-file load spike
(~356 MB) is transient; the dominant cost is the combined retained data (1,428 MB).

## Scale-up Check (single file)

| Metric | Value |
|---|---|
| File size on disk | 322 MB |
| Load time | 3.69 s |
| Peak memory (during load) | 356 MB |
| Retained memory (post-load) | 23 MB |

Conclusion: no chunking needed. Sequential processing of all 62 files is safe —
peak spikes to ~356 MB per file then drops to ~23 MB retained.

## Observations

- The optimized loader reduces rows by ~70% (4.8M → 1.4M) by aggregating across
  country codes, collapsing to one row per (square_id, time_interval).
- Peak memory did not decrease as expected. `tracemalloc` measures the Python-level
  peak during loading — pandas allocates the full DataFrame before aggregation, so
  the peak is captured before memory is freed.
- The real benefit is in the **retained** DataFrame size: ~70% fewer rows plus
  float32/int32 vs float64/int64 roughly halves per-cell memory cost.
- Across the full pipeline (62 daily files), the optimized loader keeps
  significantly less data in memory at any one time.
