"""Streamlit dashboard: upload messy files, clean and merge them automatically, then explore."""
import numpy as np
import pandas as pd
import streamlit as st

from clean_merge import (clean, common_keys, demo_data, finalise, merge_all,
                         messy_demo, read_any)

st.set_page_config(page_title="EO Data Studio | Dashboard", page_icon="🌱", layout="wide")
st.title("🌱 Data Dashboard")
st.caption("Upload messy files and they are cleaned, merged and charted automatically.")

# ---------- 1. Upload and options ----------
with st.sidebar:
    st.header("1. Upload")
    files = st.file_uploader("Data files (1 to 3)", type=["csv", "tsv", "txt", "xlsx"],
                             accept_multiple_files=True)
    demo = st.radio("No files? Try a demo", ["Messy single file", "Three tidy files"])
    st.header("2. Cleaning options")
    fuzzy = st.checkbox("Fix typos and inconsistent spelling", value=True)
    sparse = st.slider("Drop columns more than this % empty", 30, 100, 60)
    missing = st.selectbox("Fill missing numbers with", ["median", "mean", "zero", "leave blank"])
    outliers = st.selectbox("Extreme values", ["report", "cap", "ignore"])

if files:
    try:
        raw = [(f.name, read_any(f, f.name)) for f in files[:3]]
    except Exception as e:
        st.error(f"Could not read a file: {e}. Excel files need the openpyxl package; CSV always works.")
        st.stop()
else:
    st.info("Showing demo data. Upload your own files in the sidebar to replace it.")
    raw = ([("messy_demo.csv", messy_demo())] if demo == "Messy single file"
           else [(f"demo_file_{i}.csv", t) for i, t in enumerate(demo_data(), 1)])

# ---------- 2. Clean each file ----------
tables, report = [], []
for name, r in raw:
    t, log = clean(r, fuzzy)
    tables.append(t)
    report.append((name, log))

# ---------- 3. Combine ----------
final_log = []
if len(tables) == 1:
    df = tables[0]
else:
    with st.sidebar:
        st.header("3. Combine files")
        mode = st.radio("How", ["Join on a shared column", "Stack rows (same columns)"])
        keys = common_keys(tables)
        if mode.startswith("Join"):
            if not keys:
                st.error("These files share no column name. Use 'Stack rows' or rename an ID column.")
                st.stop()
            key = st.selectbox("Join on", keys)
            how = st.selectbox("Join type", ["left", "inner", "outer"])
    df = merge_all(tables, key, how) if mode.startswith("Join") else pd.concat(tables, ignore_index=True)
    final_log.append(f"Combined {len(tables)} files ({mode.split(' (')[0].lower()})")

rows_in = sum(len(r) for _, r in raw)
df, final_log = finalise(df, final_log, sparse / 100, missing, outliers)
report.append(("Combined result", final_log))

# ---------- Cleaning report ----------
fixes = sum(len(l) for _, l in report)
m = st.columns(4)
m[0].metric("Rows in", f"{rows_in:,}")
m[1].metric("Rows out", f"{len(df):,}")
m[2].metric("Columns out", f"{df.shape[1]}")
m[3].metric("Cleaning steps", fixes)
with st.expander("What was cleaned", expanded=True):
    text = []
    for name, log in report:
        st.markdown(f"**{name}**")
        st.markdown("\n".join(f"- {x}" for x in log) or "- Nothing needed fixing")
        text.append(name + "\n" + "\n".join(f"- {x}" for x in log))
    st.download_button("Download cleaning report", "\n\n".join(text), "cleaning_report.txt")

# ---------- Derived farm metrics (only if the columns exist) ----------
if {"yield_kg", "price_per_kg"} <= set(df.columns):
    df["revenue"] = df["yield_kg"] * df["price_per_kg"]
cost_cols = [c for c in df.columns if c.endswith("_cost")]
if cost_cols:
    df["total_cost"] = df[cost_cols].sum(axis=1)
if {"revenue", "total_cost"} <= set(df.columns):
    df["profit"] = df["revenue"] - df["total_cost"]

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.select_dtypes(exclude=["number", "datetime", "datetimetz", "bool"]).columns
            if df[c].nunique() <= 30]
date_cols = df.select_dtypes(include="datetime").columns.tolist()

# ---------- Filter ----------
with st.sidebar:
    st.header("Filter")
    if cat_cols:
        fcol = st.selectbox("Filter column", ["(none)"] + cat_cols)
        if fcol != "(none)":
            opts = sorted(df[fcol].dropna().unique())
            df = df[df[fcol].isin(st.multiselect(fcol, opts, default=opts))]

# ---------- KPIs ----------
st.subheader("Key numbers")
k = st.columns(4)
k[0].metric("Rows", f"{len(df):,}")
names = [c for c in ("revenue", "total_cost", "profit") if c in df.columns] or num_cols[:3]
for slot, name in zip(k[1:], names):
    slot.metric(name.replace("_", " ").title(), f"{df[name].sum():,.0f}")

# ---------- Charts ----------
st.subheader("Explore")
if cat_cols and num_cols:
    c1, c2 = st.columns(2)
    with c1:
        g = st.selectbox("Group by", cat_cols, key="g")
        v = st.selectbox("Measure", num_cols, index=num_cols.index("profit") if "profit" in num_cols else 0, key="v")
        agg = st.radio("Aggregation", ["sum", "mean"], horizontal=True)
        st.caption(f"{agg.title()} of {v} by {g}")
        st.bar_chart(df.groupby(g)[v].agg(agg).sort_values(ascending=False))
    with c2:
        counts, edges = np.histogram(df[v].dropna(), bins=20)
        st.caption(f"Distribution of {v}")
        st.bar_chart(pd.DataFrame({"count": counts}, index=[f"{e:,.0f}" for e in edges[:-1]]))
if len(num_cols) >= 2:
    a, b = st.columns(2)
    x = a.selectbox("X axis", num_cols, key="x")
    y = b.selectbox("Y axis", num_cols, index=1, key="y")
    st.caption(f"{y} vs {x}")
    st.scatter_chart(df, x=x, y=y, color=cat_cols[0] if cat_cols else None)
if date_cols and num_cols:
    d = date_cols[0]
    tdf = df.dropna(subset=[d])
    month = tdf[d].dt.to_period("M").dt.to_timestamp()
    trend = tdf.groupby(month)[num_cols[0]].sum().reset_index()
    trend.columns = [d, num_cols[0]]
    st.caption(f"{num_cols[0]} over time")
    st.line_chart(trend, x=d, y=num_cols[0])

# ---------- Table and export ----------
st.subheader("Cleaned data")
st.dataframe(df, width="stretch")
st.download_button("Download cleaned CSV", df.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")