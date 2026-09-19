"""
Loader for the Milan Telecom Italia "Internet traffic" dataset.

Raw daily files are tab-separated, no header, with columns:
    square_id, time_interval, country_code, sms_in, sms_out, call_in, call_out, internet_traffic

We only need total Internet traffic per (square_id, time_interval), summed
across all country codes. This module provides:
    - load_and_aggregate_day(): the memory-optimized loader (use this)
    - naive_load_day():         a naive loader, kept only to measure the
                                 memory saved by the optimized version
    - build_dataset():          loops all source folders, processes every
                                 daily file one at a time, and saves a single
                                 combined Parquet file
"""

import os
import shutil
from pathlib import Path

import pandas as pd

RAW_COLUMNS = [
    "square_id", "time_interval", "country_code",
    "sms_in", "sms_out", "call_in", "call_out", "internet_traffic",
]


def load_and_aggregate_day(filepath: str) -> pd.DataFrame:
    """
    Memory-efficient load of one day's raw file.

    Optimizations:
      - usecols: never parse the SMS/call columns we don't need
      - dtype:   int32 / float32 instead of pandas' default int64 / float64
      - aggregate immediately, so the returned table is one row per
        (square_id, time_interval) instead of one row per
        (square_id, time_interval, country_code)

    min_count=1 in the groupby-sum means an interval where EVERY country
    code was missing stays NaN (genuinely no data), rather than silently
    becoming 0.0 (which would misleadingly say "zero traffic").
    """
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        usecols=["square_id", "time_interval", "internet_traffic"],
        dtype={"square_id": "int32", "time_interval": "int64", "internet_traffic": "float32"},
    )
    agg = (
        df.groupby(["square_id", "time_interval"], as_index=False)["internet_traffic"]
        .sum(min_count=1)
    )
    return agg


def naive_load_day(filepath: str) -> pd.DataFrame:
    """
    Naive load: all 8 columns, pandas' default dtypes (int64/float64),
    no early aggregation. Used ONLY to produce a before/after memory
    comparison for the report -- do not use this for the real pipeline.
    """
    return pd.read_csv(filepath, sep="\t", header=None, names=RAW_COLUMNS)


# Source folders that contain the raw daily files
_SOURCE_DIRS = [
    Path("dataverse_files"),
    Path("dataverse_files (1)"),
    Path("dataverse_files (2)"),
]


def build_dataset(raw_dir: Path, out_path: Path) -> pd.DataFrame:
    """
    Loop over every daily file across all source folders, aggregate each one,
    then delete the copy in raw_dir so only one raw file lives on disk at a time.
    Saves the combined DataFrame to out_path as Parquet and returns it.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all source files in chronological order
    source_files = sorted(
        f for d in _SOURCE_DIRS if d.exists() for f in d.glob("*.txt")
    )

    frames = []
    for i, src in enumerate(source_files, 1):
        # Copy into raw_dir, process, then delete
        staging = raw_dir / src.name
        shutil.copy2(src, staging)

        day_df = load_and_aggregate_day(staging)
        retained_mb = day_df.memory_usage(deep=True).sum() / 1e6
        print(f"[{i:02d}/{len(source_files)}] {src.name}: "
              f"{day_df['square_id'].nunique():,} squares, {retained_mb:.2f} MB retained")

        frames.append(day_df)
        staging.unlink()  # delete raw copy immediately

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(out_path, index=False)
    print(f"\nSaved {len(combined):,} rows -> {out_path}")
    return combined


if __name__ == "__main__":
    import sys
    import time
    import tracemalloc

    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "single" and len(sys.argv) == 3:
        # --- per-file naive vs optimised comparison (Step 1) ---
        filepath = sys.argv[2]

        tracemalloc.start()
        naive_df = naive_load_day(filepath)
        _, naive_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        opt_df = load_and_aggregate_day(filepath)
        _, opt_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"Naive load     : peak memory = {naive_peak / 1e6:.2f} MB, rows = {len(naive_df):,}")
        print(f"Optimized load : peak memory = {opt_peak / 1e6:.2f} MB, rows = {len(opt_df):,}")
        print(f"Reduction      : {(1 - opt_peak / naive_peak) * 100:.1f}%")

    else:
        # --- full pipeline run with whole-loop tracemalloc (Steps 3 & 4) ---
        raw_dir  = Path("data/raw")
        out_path = Path("data/processed/milan_internet_traffic.parquet")

        tracemalloc.start()
        t0 = time.time()

        combined = build_dataset(raw_dir, out_path)

        elapsed = time.time() - t0
        _, full_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        combined_mb = combined.memory_usage(deep=True).sum() / 1e6
        print(f"\n--- Full-pipeline summary ---")
        print(f"Files processed : {len(_SOURCE_DIRS)} folders")
        print(f"Total rows      : {len(combined):,}")
        print(f"Combined size   : {combined_mb:.2f} MB (retained)")
        print(f"Peak memory     : {full_peak / 1e6:.2f} MB (tracemalloc, whole loop)")
        print(f"Total time      : {elapsed:.1f} s")