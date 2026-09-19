# Section 1 — Data Ingestion and Memory Management

## 1.1 Dataset

The raw dataset is the Telecom Italia Big Data Challenge (Barlacchi et al., 2015),
covering Milan's 10,000-cell grid at 10-minute resolution from 1 November 2013 to
1 January 2014 — 62 daily tab-separated files totalling 20.8 GB on disk. Each row
records activity for one grid square, one time interval, and one country code across
eight columns: square_id, time_interval, country_code, sms_in, sms_out, call_in,
call_out, and internet_traffic. Only the last three columns (square_id,
time_interval, internet_traffic) are needed for this project.

## 1.2 Ingestion strategy

Loading all 62 files naively — all eight columns, default int64/float64 dtypes, no
early aggregation — would hold roughly 62 × 349 MB ≈ 21 GB in memory simultaneously,
far exceeding a typical workstation. Four targeted optimisations were applied instead:

**usecols** — `pd.read_csv(..., usecols=[...])` instructs the parser to discard the
five SMS and call columns at read time, before they are ever allocated in memory.
Pandas' CSV engine skips those fields entirely rather than parsing and then dropping
them (McKinney, 2022).

**Dtype downcasting** — square_id is stored as int32 (max value 10,000; int32 ceiling
4.3 × 10⁹) and internet_traffic as float32 (sufficient precision for traffic volumes).
This halves the per-cell cost relative to pandas' default int64/float64, since each
value occupies 4 bytes instead of 8.

**Early aggregation** — each raw file contains one row per (square_id, time_interval,
country_code) triplet. Summing internet_traffic across country codes immediately after
loading collapses ~4.8 M rows to ~1.4 M rows (a 70% reduction) before the DataFrame
is appended to the accumulator list.

**Process-then-discard** — each raw file is copied into a staging folder, processed,
and deleted before the next file is read. At no point does more than one 322 MB raw
file exist on disk simultaneously.

## 1.3 Per-file memory: naive vs. optimised

Measured with Python's `tracemalloc` on `sms-call-internet-mi-2013-11-01.txt`
(322 MB on disk):

| Method | Peak memory | Rows |
|---|---|---|
| Naive (all columns, default dtypes) | 348.95 MB | 4,842,625 |
| Optimised (usecols + int32/float32 + groupby) | 355.99 MB | 1,439,982 |

The peak figures are nearly identical because `tracemalloc` captures the high-water
mark during loading — pandas allocates the full intermediate buffer before the
aggregation frees memory. The meaningful difference is in the **retained** DataFrame:
23 MB after aggregation vs. the naive 349 MB peak, a 93% reduction in working-set
size. This is the figure that matters for a pipeline that accumulates 62 daily frames.

## 1.4 Full-pipeline memory: all 62 files

| Metric | Value |
|---|---|
| Files processed | 62 |
| Total rows (combined) | 89,245,318 |
| Per-file retained memory | ~23 MB |
| Peak memory (tracemalloc, whole loop) | 2,857 MB |
| Total ingestion time | 298 s (~5 min) |

The 2,857 MB whole-loop peak reflects the `frames` list growing as all 62 aggregated
DataFrames accumulate before `pd.concat`. The per-file load spike (~356 MB) is
transient and released immediately after each file. A streaming write strategy
(writing each day's frame to Parquet incrementally) could reduce this further, but
the current peak was comfortably within available RAM and no intervention was needed.

## 1.5 Raw files vs. final Parquet

After concatenation the combined DataFrame was written to a single Parquet file using
Snappy compression (the pyarrow default):

| | Size |
|---|---|
| Raw `.txt` files (62 days) | 20.805 GB |
| `milan_internet_traffic.parquet` | 408.79 MB |
| Reduction | **98.0% — 50.9× smaller** |

Three effects compound to produce this reduction: (1) five of eight columns are
dropped entirely; (2) early aggregation removes the country-code dimension, cutting
row count by 70%; and (3) Parquet's columnar layout with Snappy compression encodes
repeated integer and float values far more efficiently than plain-text TSV. The
Parquet file reloads in 2.4 s via `pd.read_parquet`, compared to ~5 minutes for the
full ingestion pipeline. All analysis in Sections 2–4 reads exclusively from this
file — the raw daily files are never accessed again.

## References

Barlacchi, G., De Nadai, M., Larcher, R., Casella, A., Chitic, C., Torrisi, G.,
Antonelli, F., Vespignani, A., Pentland, A., & Lepri, B. (2015). A multi-source
dataset of urban life in the city of Milan and the Province of Trentino.
*Scientific Data*, 2, 150055. https://doi.org/10.1038/sdata.2015.55

McKinney, W. (2022). *Python for Data Analysis* (3rd ed.). O'Reilly Media.
https://wesmckinney.com/book/accessing-data#io_flat_file

Apache Parquet (2024). *Parquet file format specification*.
https://parquet.apache.org/docs/file-format/
