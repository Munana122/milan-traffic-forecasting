"""
Single entry point for loading the processed dataset.

All scripts and notebooks in Sections 2-4 import from here:

    from src.data import load_traffic

Never load raw daily files after the ingestion pipeline (src/loader.py) has run.
"""

from pathlib import Path

import pandas as pd

PARQUET_PATH = Path(__file__).parent.parent / "data" / "processed" / "milan_internet_traffic.parquet"


def load_traffic(columns: list[str] | None = None) -> pd.DataFrame:
    """
    Load the processed Milan internet-traffic dataset from Parquet.

    Parameters
    ----------
    columns : list of str, optional
        Subset of columns to load (e.g. ['square_id', 'internet_traffic']).
        Omit to load all three columns: square_id, time_interval, internet_traffic.

    Returns
    -------
    pd.DataFrame with shape (89_245_318, 3) by default.
    """
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(
            f"Parquet file not found at {PARQUET_PATH}.\n"
            "Run `python src/loader.py` first to build it."
        )
    return pd.read_parquet(PARQUET_PATH, columns=columns)


if __name__ == "__main__":
    import time

    t0 = time.time()
    df = load_traffic()
    elapsed = time.time() - t0

    mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"Shape   : {df.shape}")
    print(f"Columns : {list(df.columns)}")
    print(f"Loaded  : {elapsed:.2f} s, {mb:.2f} MB in-memory")
    print(df.head())
