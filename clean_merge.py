"""Clean and merge three CSV files into one analysis-ready table.

Command line:  python clean_merge.py a.csv b.csv c.csv key merged.csv
"""
import sys

import numpy as np
import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise names, fix types, drop duplicates and fill gaps."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_")
    )
    df = df.drop_duplicates()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.strip()
        if "date" in col:
            df[col] = pd.to_datetime(s, errors="coerce")
            continue
        conv = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() and conv.notna().sum() >= 0.9 * s.notna().sum():
            df[col] = conv.astype(float)
        else:
            df[col] = s.fillna("Unknown")
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].fillna(df[num].median())
    return df


def common_keys(dfs):
    """Column names that exist in every table (candidate join keys)."""
    cols = set(dfs[0].columns)
    for d in dfs[1:]:
        cols &= set(d.columns)
    return sorted(cols)


def merge_all(dfs, key, how="left"):
    """Merge several tables on one key (or list of keys)."""
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on=key, how=how, suffixes=("", "_dup"))
    return out.loc[:, ~out.columns.str.endswith("_dup")]


def demo_data(n=24, seed=7):
    """Small made-up farm dataset so the dashboard works out of the box."""
    rng = np.random.default_rng(seed)
    ids = [f"F{i:03d}" for i in range(1, n + 1)]
    farms = pd.DataFrame({
        "Farm ID": ids,
        "Farm Name": [f"Farm {i}" for i in range(1, n + 1)],
        "County": rng.choice(["Kiambu", "Nakuru", "Kisumu", "Meru"], n),
        "Crop": rng.choice(["Maize", "Beans", "Tomatoes", "Kale"], n),
        "Acres": rng.integers(1, 12, n),
    })
    harvest = pd.DataFrame({
        "Farm ID": ids,
        "Harvest Date": pd.date_range("2025-01-15", periods=n, freq="9D").astype(str),
        "Yield (kg)": rng.integers(600, 4200, n),
        "Price per kg": rng.integers(25, 90, n),
    })
    expenses = pd.DataFrame({
        "Farm ID": ids,
        "Seed Cost": rng.integers(8000, 40000, n),
        "Labour Cost": rng.integers(15000, 90000, n),
        "Fertiliser Cost": rng.integers(10000, 70000, n),
    })
    return farms, harvest, expenses


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    *paths, key, out_path = sys.argv[1:]
    tables = [clean(pd.read_csv(p)) for p in paths]
    merge_all(tables, key).to_csv(out_path, index=False)
    print(f"Saved {out_path}")