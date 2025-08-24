# app.py — Streamlit GUI for fast name matching across TWO CSV files (generic, configurable)

import io
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ----------------------------- UI: Header -----------------------------
st.set_page_config(page_title="PairMatch", page_icon="🔗", layout="wide")
st.title("🔗 PairMatch")
st.caption("General-purpose, fast project/name matching using char n-gram TF-IDF + cosine similarity.")

with st.expander("How it works", expanded=False):
    st.markdown(
        """
        **PairMatch** computes character n-gram TF-IDF vectors for the name strings you choose from each side,
        then finds the nearest neighbor (by cosine similarity) on the Right for each row on the Left.
        
        - Bring **two CSV files** (Left & Right) — any domains, any schemas.
        - Choose which **column(s) to merge** into a single **matchable name** per side (e.g., `Project` + `API`).
        - Choose which **code** column to return for each side.
        - Optionally filter rows (e.g., `Linked? == 0`).
        - Tune **n-gram range** & **minimum cosine**.
        """
    )

# ----------------------------- Helpers -----------------------------
def normalize_text(s: str) -> str:
    """Lowercase, normalize greek micro -> mcg, strip accents, collapse spaces."""
    s = "" if s is None else str(s)
    s = s.lower()
    s = s.replace("μg", "mcg").replace("µg", "mcg")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip()
    return s

@st.cache_data(show_spinner=False)
def read_csv_file(file, encoding=None, sep=None) -> pd.DataFrame:
    """Read CSV into DataFrame as strings, preserving empty as empty-string where possible."""
    kwargs = dict(low_memory=False, dtype=str, keep_default_na=False, na_filter=False)
    if encoding:
        kwargs["encoding"] = encoding
    if sep:
        kwargs["sep"] = sep
    try:
        return pd.read_csv(file, **kwargs)
    except Exception:
        # Fallback: try utf-8-sig then latin-1 if no encoding was provided
        if not encoding:
            file.seek(0)
            try:
                return pd.read_csv(file, encoding="utf-8-sig", **kwargs)
            except Exception:
                file.seek(0)
                return pd.read_csv(file, encoding="latin-1", **kwargs)
        raise

def merge_columns(df: pd.DataFrame, cols: list[str], sep: str) -> pd.Series:
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    # Ensure string, then join
    parts = [df[c].astype(str) if c in df.columns else "" for c in cols]
    merged = parts[0]
    for p in parts[1:]:
        merged = merged + sep + p
    # normalize multiple separators/spaces
    merged = merged.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return merged

def ensure_has_columns(df: pd.DataFrame, needed: list[str]) -> list[str]:
    return [c for c in needed if c in df.columns]

def nearest_neighbor_top1(left_vecs, right_vecs):
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=1)
    nn.fit(right_vecs)
    dists, idxs = nn.kneighbors(left_vecs, n_neighbors=1, return_distance=True)
    best_j = idxs.ravel().astype(int)
    best_cos = (1.0 - dists.ravel()).astype(float)  # cosine
    return best_j, best_cos

# ----------------------------- Sidebar: Options -----------------------------
st.sidebar.header("Options")

ng_min = st.sidebar.number_input("n-gram min (char)", min_value=2, max_value=8, value=3, step=1)
ng_max = st.sidebar.number_input("n-gram max (char)", min_value=ng_min, max_value=10, value=5, step=1)
min_cosine = st.sidebar.slider("Minimum cosine to keep", 0.0, 1.0, 0.0, 0.01)

left_filter_enable = st.sidebar.checkbox("Filter Left by column == value", value=False)
right_filter_enable = st.sidebar.checkbox("Filter Right by column == value", value=False)

# CSV reading options (in case of weird encodings/separators)
with st.sidebar.expander("CSV parsing (optional)"):
    csv_encoding = st.text_input("Encoding (blank = auto)", value="")
    csv_sep = st.text_input("Separator (blank = auto)", value="")

# ----------------------------- Main: Uploads -----------------------------
st.subheader("1) Upload two CSV files")
colA, colB = st.columns(2)
with colA:
    left_file = st.file_uploader("Left CSV", type=["csv"], key="left_csv")
with colB:
    right_file = st.file_uploader("Right CSV", type=["csv"], key="right_csv")

if left_file and right_file:
    # Read
    left_df = read_csv_file(left_file, encoding=csv_encoding or None, sep=csv_sep or None)
    right_df = read_csv_file(right_file, encoding=csv_encoding or None, sep=csv_sep or None)

    st.success(f"Loaded Left: {left_df.shape[0]:,} rows, {left_df.shape[1]:,} cols • Right: {right_df.shape[0]:,} rows, {right_df.shape[1]:,} cols")

    # ----------------------------- Column selection -----------------------------
    st.subheader("2) Choose columns")

    lcol, rcol = st.columns(2)

    with lcol:
        st.markdown("**Left**")
        left_code_col = st.selectbox("Left code column (returned in output)", options=list(left_df.columns), index=0)
        left_merge_cols = st.multiselect(
            "Left name columns to merge (the compared name)",
            options=list(left_df.columns),
            help="Pick one or more columns to merge into the name string that will be matched."
        )
        left_sep = st.text_input("Left merge separator", value=" ")
        if left_filter_enable:
            left_filter_col = st.selectbox("Left filter column", options=list(left_df.columns))
            lf_unique = sorted(list(pd.unique(left_df[left_filter_col].astype(str))))[:1000]
            left_filter_val = st.selectbox("Keep rows where column == value", options=lf_unique)
        else:
            left_filter_col, left_filter_val = None, None

    with rcol:
        st.markdown("**Right**")
        right_code_col = st.selectbox("Right code column (returned in output)", options=list(right_df.columns), index=0)
        right_merge_cols = st.multiselect(
            "Right name columns to merge (the compared name)",
            options=list(right_df.columns),
            help="Pick one or more columns to merge into the name string that will be matched."
        )
        right_sep = st.text_input("Right merge separator", value=" ")
        if right_filter_enable:
            right_filter_col = st.selectbox("Right filter column", options=list(right_df.columns))
            rf_unique = sorted(list(pd.unique(right_df[right_filter_col].astype(str))))[:1000]
            right_filter_val = st.selectbox("Keep rows where column == value", options=rf_unique)
        else:
            right_filter_col, right_filter_val = None, None

    # ----------------------------- Run matching -----------------------------
    run = st.button("🚀 Run matching", type="primary", use_container_width=True)

    if run:
        # Defensive: apply filters
        if left_filter_col is not None:
            left_df = left_df[left_df[left_filter_col].astype(str) == str(left_filter_val)]
        if right_filter_col is not None:
            right_df = right_df[right_df[right_filter_col].astype(str) == str(right_filter_val)]

        # Validate selection
        if not left_merge_cols or not right_merge_cols:
            st.error("Please select at least one merge column for both Left and Right.")
            st.stop()

        # Build merged/normalized name columns
        left_name_raw = merge_columns(left_df, left_merge_cols, left_sep)
        right_name_raw = merge_columns(right_df, right_merge_cols, right_sep)

        left_name_norm = left_name_raw.map(normalize_text)
        right_name_norm = right_name_raw.map(normalize_text)

        # Vectorize together for consistent vocab
        vect = TfidfVectorizer(analyzer="char", ngram_range=(ng_min, ng_max), norm="l2", lowercase=False)
        X_all = vect.fit_transform(pd.concat([left_name_norm, right_name_norm], ignore_index=True))
        X_L = X_all[: len(left_name_norm)]
        X_R = X_all[len(left_name_norm) :]

        # Nearest neighbor Top-1
        best_j, best_cos = nearest_neighbor_top1(X_L, X_R)

        # Build results
        # Note: Right indices are positional; we must select by iloc
        right_codes = right_df.iloc[best_j][right_code_col].to_numpy()
        right_names = right_name_raw.iloc[best_j].to_numpy()

        out = pd.DataFrame(
            {
                "Left_Code": left_df[left_code_col].to_numpy(),
                "Left_Name": left_name_raw.to_numpy(),
                "Right_Code": right_codes,
                "Right_Name": right_names,
                "cosine": best_cos,
            }
        )

        # Apply minimum cosine filter (optional)
        if min_cosine > 0.0:
            out = out[out["cosine"] >= float(min_cosine)].reset_index(drop=True)

        # Report & preview
        if len(out) == 0:
            st.warning("No rows passed the minimum cosine threshold.")
        else:
            st.success(f"Done! {len(out):,} matches.")
            st.write(
                f"**Cosine stats** — min: {out['cosine'].min():.3f} • median: {out['cosine'].median():.3f} • max: {out['cosine'].max():.3f}"
            )

            st.dataframe(out.head(500), use_container_width=True)

            # Download
            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results (CSV)",
                data=csv_bytes,
                file_name="pairmatch_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    st.info("Upload both Left and Right CSVs to begin.")
