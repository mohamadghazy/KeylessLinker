import io
import pandas as pd
import streamlit as st

from Linker import run_matching

st.set_page_config(page_title="Char-Ngram Record Linker", page_icon="🔗", layout="wide")

st.title("🔗 Char-Ngram Record Linker")
st.caption(
    "Upload two CSVs, pick the identifier column on each side, "
    "choose one or more text columns to merge and compare, and get best matches by cosine similarity."
)

with st.sidebar:
    st.header("How it works")
    st.write(
        "- Uses char TF-IDF on a merged text column per side\n"
        "- Finds the nearest neighbor (Right) for each Left row by cosine similarity\n"
        "- Lets you filter by a minimum similarity threshold"
    )

# --------- Uploads ----------
col_u1, col_u2 = st.columns(2)
with col_u1:
    left_file = st.file_uploader("Upload LEFT CSV", type=["csv"], key="left_csv")
with col_u2:
    right_file = st.file_uploader("Upload RIGHT CSV", type=["csv"], key="right_csv")

if left_file and right_file:
    # Load data
    try:
        df_left = pd.read_csv(left_file, low_memory=False)
        df_right = pd.read_csv(right_file, low_memory=False)
    except Exception as e:
        st.error(f"Failed to read CSVs: {e}")
        st.stop()

    st.subheader("Pick Columns")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**LEFT**")
        st.dataframe(df_left.head(10), use_container_width=True, height=240)
        code_col_left = st.selectbox("Left: identifier (code) column", df_left.columns, key="code_left")
        text_cols_left = st.multiselect(
            "Left: text columns to merge (comparison field)", df_left.columns, key="text_left"
        )

    with c2:
        st.markdown("**RIGHT**")
        st.dataframe(df_right.head(10), use_container_width=True, height=240)
        code_col_right = st.selectbox("Right: identifier (code) column", df_right.columns, key="code_right")
        text_cols_right = st.multiselect(
            "Right: text columns to merge (comparison field)", df_right.columns, key="text_right"
        )

    st.subheader("Settings")
    s1, s2, s3 = st.columns([1, 1, 2])
    with s1:
        ngram_min = st.number_input("n-gram min", min_value=2, max_value=6, value=4, step=1,
        help="Minimum length of character n-grams used by TF-IDF (lower = more tolerant, but noisier).")

    with s2:
        ngram_max = st.number_input("n-gram max", min_value=2, max_value=12, value=8, step=1,
        help="Maximum length of character n-grams (higher = more specific, but more brittle to typos).")

    with s3:
        min_cosine = st.slider("Minimum cosine threshold", 0.0, 1.0, 0.0, 0.01,
        help="Keep only matches with cosine similarity ≥ this value (0–1). Higher = stricter.")


    if ngram_max < ngram_min:
        st.warning("n-gram max must be ≥ n-gram min.")
        st.stop()

    run_btn = st.button("🚀 Run Matching", type="primary", use_container_width=True)

    if run_btn:
        if not text_cols_left or not text_cols_right:
            st.error("Please select at least one text column on each side.")
            st.stop()

        with st.spinner("Matching in progress..."):
            result = run_matching(
                df_left=df_left,
                df_right=df_right,
                code_col_left=code_col_left,
                text_cols_left=text_cols_left,
                code_col_right=code_col_right,
                text_cols_right=text_cols_right,
                ngram_range=(int(ngram_min), int(ngram_max)),
                min_cosine=float(min_cosine),
            )

        st.success(f"Done! {len(result)} paired rows.")
        st.subheader("Preview")
        st.dataframe(result.head(50), use_container_width=True, height=400)

        # Download full results
        csv_bytes = result.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download matches CSV", csv_bytes, file_name="matches.csv", mime="text/csv")

        # Also return just the paired codes (as requested)
        st.subheader("Paired Codes Only")
        code_pairs = result[["Left_Code", "Right_Code"]]
        st.dataframe(code_pairs.head(50), use_container_width=True, height=240)
        pairs_csv = code_pairs.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download paired codes CSV", pairs_csv, file_name="paired_codes.csv", mime="text/csv")

else:
    st.info("Upload two CSVs to get started.")
