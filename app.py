"""Streamlit dashboard: upload three CSVs, clean, merge and explore them."""
import pandas as pd
import plotly.express as px
import streamlit as st

from clean_merge import clean, common_keys, demo_data, merge_all

st.set_page_config(page_title="EO Data Studio | Dashboard", page_icon="🌱", layout="wide")
st.title("🌱 Data Dashboard")
st.caption("Upload three CSV files or use the demo data. They are cleaned, merged and visualised automatically.")

# ---------- 1. Load ----------
with st.sidebar:
    st.header("1. Data")
    uploads = [st.file_uploader(f"CSV file {i}", type="csv", key=f"f{i}") for i in (1, 2, 3)]
    use_demo = st.toggle("Use demo data", value=not all(uploads))

if use_demo or not all(uploads):
    raw = list(demo_data())
    st.info("Showing demo farm data. Turn off the demo toggle and upload your three files to use your own.")
else:
    raw = [pd.read_csv(u) for u in uploads]

tables = [clean(t) for t in raw]
with st.expander("Preview the three cleaned tables"):
    for i, t in enumerate(tables, 1):
        st.write(f"**File {i}**: {t.shape[0]} rows, {t.shape[1]} columns")
        st.dataframe(t.head(), use_container_width=True)

# ---------- 2. Merge ----------
keys = common_keys(tables)
if not keys:
    st.error("The three files share no column name. Rename a matching ID column so they can be joined.")
    st.stop()
with st.sidebar:
    st.header("2. Merge")
    key = st.selectbox("Join on", keys)
    how = st.selectbox("Join type", ["left", "inner", "outer"])

df = merge_all(tables, key, how)

# Derived farm metrics, only when the columns exist
cols = set(df.columns)
if {"yield_kg", "price_per_kg"} <= cols:
    df["revenue"] = df["yield_kg"] * df["price_per_kg"]
cost_cols = [c for c in df.columns if c.endswith("_cost")]
if cost_cols:
    df["total_cost"] = df[cost_cols].sum(axis=1)
if {"revenue", "total_cost"} <= set(df.columns):
    df["profit"] = df["revenue"] - df["total_cost"]

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df.select_dtypes(exclude="number").columns
            if df[c].nunique() <= 30 and "date" not in c]
date_cols = df.select_dtypes(include="datetime").columns.tolist()

# ---------- 3. Filter ----------
with st.sidebar:
    st.header("3. Filter")
    if cat_cols:
        fcol = st.selectbox("Filter column", ["(none)"] + cat_cols)
        if fcol != "(none)":
            picked = st.multiselect(f"{fcol}", sorted(df[fcol].dropna().unique()),
                                    default=sorted(df[fcol].dropna().unique()))
            df = df[df[fcol].isin(picked)]

# ---------- 4. KPIs ----------
st.subheader("Key numbers")
k = st.columns(4)
k[0].metric("Rows", f"{len(df):,}")
for slot, name in zip(k[1:], [c for c in ("revenue", "total_cost", "profit") if c in df.columns] or num_cols[:3]):
    slot.metric(name.replace("_", " ").title(), f"{df[name].sum():,.0f}")

# ---------- 5. Charts ----------
st.subheader("Explore")
c1, c2 = st.columns(2)
if cat_cols and num_cols:
    with c1:
        g = st.selectbox("Group by", cat_cols, key="g")
        v = st.selectbox("Measure", num_cols, index=num_cols.index("profit") if "profit" in num_cols else 0, key="v")
        agg = st.radio("Aggregation", ["sum", "mean"], horizontal=True)
        data = df.groupby(g, as_index=False)[v].agg(agg).sort_values(v, ascending=False)
        st.plotly_chart(px.bar(data, x=g, y=v, title=f"{agg.title()} of {v} by {g}"), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(df, x=v, nbins=20, title=f"Distribution of {v}"), use_container_width=True)
if len(num_cols) >= 2:
    a, b = st.columns(2)
    x = a.selectbox("X axis", num_cols, key="x")
    y = b.selectbox("Y axis", num_cols, index=1, key="y")
    st.plotly_chart(px.scatter(df, x=x, y=y, color=cat_cols[0] if cat_cols else None,
                               title=f"{y} vs {x}"), use_container_width=True)
if date_cols and num_cols:
    d = date_cols[0]
    month = df[d].dt.to_period("M").dt.to_timestamp()
    trend = df.groupby(month)[num_cols[0]].sum().reset_index()
    trend.columns = [d, num_cols[0]]
    st.plotly_chart(px.line(trend, x=d, y=num_cols[0], markers=True,
                            title=f"{num_cols[0]} over time"), use_container_width=True)

# ---------- 6. Table and export ----------
st.subheader("Merged data")
st.dataframe(df, use_container_width=True)
st.download_button("Download merged CSV", df.to_csv(index=False).encode(), "merged_data.csv", "text/csv")