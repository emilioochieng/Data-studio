"""Automatic cleaning and merging for messy data files.

Command line:  python clean_merge.py messy.csv cleaned.csv
"""
import difflib
import io
import sys

import numpy as np
import pandas as pd

PLACEHOLDERS = {"", "na", "n/a", "#n/a", "nan", "null", "none", "nil", "-", "--", "?",
                "missing", "unknown", "not available"}
ID_WORDS = {"id", "phone", "tel", "code", "zip", "sku", "ref", "no"}
DATE_RE = (r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}"
           r"|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}")


# ---------- reading ----------
def read_any(file, name=""):
    """Read CSV/TSV/TXT (any delimiter or encoding) or Excel as plain text."""
    name = (name or getattr(file, "name", "")).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file, dtype=str)
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return _read_text(text)


def _read_text(text):
    """Detect the delimiter and the real header row, even with junk lines above the table."""
    lines = [ln for ln in text.splitlines() if ln.strip()][:60]
    best, score = ",", -1
    for d in [",", ";", "\t", "|"]:
        counts = [ln.count(d) for ln in lines]
        top = max(set(counts), key=counts.count)
        if top and top * counts.count(top) > score:
            best, score = d, top * counts.count(top)
    width = max(ln.count(best) for ln in lines) + 1
    df = pd.read_csv(io.StringIO(text), sep=best, header=None, names=list(range(width)),
                     dtype=str, skip_blank_lines=True, engine="python", on_bad_lines="skip")
    df = df.dropna(how="all").reset_index(drop=True)
    counts = df.notna().sum(axis=1).values
    pos = int(np.argmax(counts >= 0.8 * counts.max()))
    df.columns = [str(v) if pd.notna(v) else "" for v in df.iloc[pos]]
    out = df.iloc[pos + 1:].reset_index(drop=True)
    out.attrs["junk_rows"] = pos
    return out


# ---------- helpers ----------
def _id_like(col):
    return bool(set(col.split("_")) & ID_WORDS)


def _fix_header(df, log):
    unnamed = sum(str(c).lower().startswith("unnamed") for c in df.columns)
    if unnamed >= max(1, len(df.columns) // 2) and len(df):
        counts = df.notna().sum(axis=1).values
        pos = int(np.argmax(counts >= 0.8 * counts.max()))
        df.columns = [str(v) if pd.notna(v) else "" for v in df.iloc[pos]]
        df = df.iloc[pos + 1:].reset_index(drop=True)
        log.append(f"Found the real header row (row {pos + 2}) and removed {pos + 1} junk rows above it")
    return df


def _tidy_columns(df, log):
    new = (df.columns.astype(str).str.strip().str.lower()
           .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_"))
    seen, out = {}, []
    for i, c in enumerate(new):
        c = c or f"column_{i + 1}"
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    if out != [str(c) for c in df.columns]:
        log.append("Standardised column names (lowercase, no spaces or symbols)")
    df.columns = out
    return df


def _to_number(s):
    t = s.astype("string").str.strip()
    neg = t.str.match(r"^\(.*\)$", na=False)
    t = (t.str.replace(r"(?i)(kes|ksh|usd|kgs?|acres?)\.?|[$€£%]", "", regex=True)
          .str.replace(r"[,\s()]", "", regex=True))
    num = pd.to_numeric(t, errors="coerce")
    return num.where(~neg, -num.abs())


def _to_date(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")


def _harmonise(s, fuzzy=True):
    """Make 'nakuru', 'NAKURU ' and 'Nakuru' one value; fold rare typos into common spellings."""
    vals = s.dropna()
    low = vals.str.lower()
    canon = vals.groupby(low).agg(lambda x: x.value_counts().idxmax())
    canon = canon.map(lambda v: v.title() if (v.islower() or v.isupper()) else v)
    m = low.map(canon)
    if fuzzy:
        counts = m.value_counts()
        limit = max(2, int(0.05 * len(m)))
        common = [v for v, c in counts.items() if c > limit]
        remap = {}
        for v, c in counts.items():
            if c <= limit:
                hit = difflib.get_close_matches(v.lower(), [x.lower() for x in common], n=1, cutoff=0.85)
                if hit:
                    remap[v] = next(x for x in common if x.lower() == hit[0])
        m = m.replace(remap)
    out = s.copy()
    out.loc[vals.index] = m
    return out


# ---------- main cleaning ----------
def clean(df, fuzzy=True):
    """Structure, types, text and dates. Returns (clean_df, log)."""
    log = []
    df = df.copy()
    if df.attrs.get("junk_rows"):
        log.append(f"Skipped {df.attrs['junk_rows']} title or note rows above the real header")
    n_rows, n_cols = df.shape
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if df.shape != (n_rows, n_cols):
        log.append(f"Removed {n_rows - len(df)} empty rows and {n_cols - df.shape[1]} empty columns")
    df = _fix_header(df, log)
    df = _tidy_columns(df, log)

    blanked = 0
    for col in list(df.columns):
        s = df[col].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
        before = s.notna().sum()
        s = s.mask(s.str.lower().isin(PLACEHOLDERS))
        blanked += int(before - s.notna().sum())
        nn = int(s.notna().sum())
        idl = _id_like(col)
        if nn == 0:
            df[col] = s
            continue
        if "date" in col or "time" in col or (not idl and s.dropna().str.contains(DATE_RE).mean() >= 0.8):
            d = _to_date(s)
            if d.notna().sum() >= 0.8 * nn:
                bad = int(nn - d.notna().sum())
                df[col] = d
                log.append(f"Converted '{col}' to dates" + (f" ({bad} unreadable dates left blank)" if bad else ""))
                continue
        if not idl:
            num = _to_number(s)
            if num.notna().sum() >= 0.8 * nn:
                bad = int(nn - num.notna().sum())
                df[col] = num.astype(float)
                log.append(f"Converted '{col}' to numbers (removed commas, currency and units)"
                           + (f"; {bad} unreadable values left blank" if bad else ""))
                continue
        if not idl and s.nunique() <= 60 and s.dropna().str.len().mean() < 40:
            new = _harmonise(s, fuzzy)
            diff = (new.dropna() != s.dropna())
            if diff.sum():
                pairs = sorted({(a, b) for a, b in zip(s.dropna()[diff], new.dropna()[diff])})[:3]
                eg = ", ".join(f"'{a}' → '{b}'" for a, b in pairs)
                log.append(f"Unified {int(diff.sum())} inconsistent spellings in '{col}' (e.g. {eg})")
            s = new
        df[col] = s
    if blanked:
        log.append(f"Turned {blanked} placeholder values (N/A, -, null, ...) into proper blanks")
    return df, log


def finalise(df, log, sparse=0.6, missing="median", outliers="report"):
    """Duplicates, mostly-empty columns, missing values and extreme values."""
    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n:
        log.append(f"Removed {n - len(df)} duplicate rows")
    if len(df) >= 5:
        empty = [c for c in df.columns if df[c].isna().mean() > sparse]
        if empty:
            df = df.drop(columns=empty)
            log.append(f"Dropped columns that were more than {int(sparse * 100)}% empty: {', '.join(empty)}")
    for c in df.select_dtypes(include="number").columns:
        k = int(df[c].isna().sum())
        if k and missing != "leave blank":
            val = {"median": df[c].median(), "mean": df[c].mean(), "zero": 0}[missing]
            df[c] = df[c].fillna(val)
            log.append(f"Filled {k} missing values in '{c}' with the {missing}")
    for c in df.select_dtypes(exclude=["number", "datetime", "datetimetz", "bool"]).columns:
        k = int(df[c].isna().sum())
        if k:
            df[c] = df[c].fillna("Unknown")
            log.append(f"Filled {k} missing entries in '{c}' with 'Unknown'")
    for c in df.select_dtypes(include="datetime").columns:
        k = int(df[c].isna().sum())
        if k:
            log.append(f"{k} missing dates in '{c}' were left blank")
    if outliers != "ignore":
        for c in df.select_dtypes(include="number").columns:
            if _id_like(c) or df[c].nunique() < 8:
                continue
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            k = int(((df[c] < lo) | (df[c] > hi)).sum())
            if iqr and k:
                if outliers == "cap":
                    df[c] = df[c].clip(lo, hi)
                    log.append(f"Capped {k} extreme values in '{c}' to the range {lo:,.0f} to {hi:,.0f}")
                else:
                    log.append(f"Warning: {k} extreme values in '{c}' (outside {lo:,.0f} to {hi:,.0f}), please check them")
    return df, log


# ---------- merging ----------
def common_keys(dfs):
    cols = set(dfs[0].columns)
    for d in dfs[1:]:
        cols &= set(d.columns)
    return sorted(cols)


def merge_all(dfs, key, how="left"):
    keys = [key] if isinstance(key, str) else list(key)
    prepared = []
    for d in dfs:
        d = d.copy()
        for k in keys:
            if not pd.api.types.is_numeric_dtype(d[k]):
                d[k] = d[k].astype("string").str.strip().str.upper()
        prepared.append(d)
    out = prepared[0]
    for d in prepared[1:]:
        out = out.merge(d, on=key, how=how, suffixes=("", "_dup"))
    return out.loc[:, ~out.columns.str.endswith("_dup")]


# ---------- demo data ----------
def demo_data(n=24, seed=7):
    """Three tidy tables of made-up farm data."""
    rng = np.random.default_rng(seed)
    ids = [f"F{i:03d}" for i in range(1, n + 1)]
    farms = pd.DataFrame({"Farm ID": ids, "Farm Name": [f"Farm {i}" for i in range(1, n + 1)],
                          "County": rng.choice(["Kiambu", "Nakuru", "Kisumu", "Meru"], n),
                          "Crop": rng.choice(["Maize", "Beans", "Tomatoes", "Kale"], n),
                          "Acres": rng.integers(1, 12, n)})
    harvest = pd.DataFrame({"Farm ID": ids,
                            "Harvest Date": pd.date_range("2025-01-15", periods=n, freq="9D").astype(str),
                            "Yield (kg)": rng.integers(600, 4200, n),
                            "Price per kg": rng.integers(25, 90, n)})
    expenses = pd.DataFrame({"Farm ID": ids, "Seed Cost": rng.integers(8000, 40000, n),
                             "Labour Cost": rng.integers(15000, 90000, n),
                             "Fertiliser Cost": rng.integers(10000, 70000, n)})
    return [farms, harvest, expenses]


def messy_demo():
    """One deliberately messy farm file, to show the cleaning at work."""
    cols = ["Farm ID", "Farm Name ", "COUNTY", "Crop", "Acres", "Yield (Kg)", "Price per Kg",
            "Harvest Date", "Seed Cost", "Labour Cost"]
    rows = [
        ["F001 ", "Wanjiru Farm", "nakuru", "maize", "5 acres", "1,850 kg", "KES 45", "12/03/2025", "28,000", "42,000"],
        ["F002", "Otieno Farm", "Nakuru ", "Maize", "3", "1,200", "KES 48", "2025-03-15", "18,500", "30,000"],
        ["F003", "Kamau Farm", "NAKURU", "MAIZE", "8", "N/A", "ksh 44", "3 Apr 2025", "35,000", "-"],
        ["F004", "Achieng Farm", "Nkuru", "Beans", "2", "640", "KES 90", "20/03/2025", "9,000", "12,000"],
        ["F005", "Mwangi Farm", "kiambu", "Tomatoes", "4", "3,900", "KES 60", "25/03/2025", "30,000", "55,000"],
        ["F005", "Mwangi Farm", "kiambu", "Tomatoes", "4", "3,900", "KES 60", "25/03/2025", "30,000", "55,000"],
        [None] * 10,
        ["F006", "Njeri Farm", "Kiambu ", "tomatos", "6", "4,100", "KES 58", "02/04/2025", "42,000", "61,000"],
        ["F007", "Odhiambo Farm", "Kisumu", "Beans", "3", "700", "-", "5 Apr 2025", "11,000", "14,500"],
        ["F008", "Kiprop Farm", "kisumu", "Beans", "(2)", "720", "KES 85", "10/04/2025", "10,500", "13,000"],
        ["F009", "Wafula Farm", "Kisumu", "Kale", "1", "999,999", "KES 30", "12/04/2025", "4,000", "8,000"],
        ["F010", "Chebet Farm", "Meru", "Kale", "2", "900", "KES 32", "not sure", "5,500", "9,500"],
        ["F011", "Mutua Farm", "Meru", "kale", "2", "950", "KES 31", "18/04/2025", "5,800", "N/A"],
        ["F012", "Naliaka Farm", "meru", "Maize", "7", "2,300", "KES 46", "22/04/2025", "33,000", "47,000"],
        ["F013", "Barasa Farm", "Nakuru", "Maize", "6", "2,050", "KES 47", "28/04/2025", "31,000", "44,000"],
        ["F014", "Rono Farm", "Kiambu", "Tomatoes", "5", "3,500", "KES 62", "30/04/2025", "36,000", "52,000"],
    ]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    cleaned, steps = clean(read_any(sys.argv[1]))
    cleaned, steps = finalise(cleaned, steps)
    cleaned.to_csv(sys.argv[2], index=False)
    print("\n".join(f"- {s}" for s in steps))