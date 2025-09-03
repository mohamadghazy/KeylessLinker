import re
import unicodedata
from typing import Iterable, Tuple, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ---------------- Normalization ----------------
def normalize_text(s: str) -> str:
    """
    Light, robust normalization suitable for pharmaceutical/product text.
    - lowercases
    - normalizes microgram glyphs to 'mcg'
    - strips accents
    - collapses whitespace
    """
    s = "" if s is None else str(s)
    s = s.lower()
    s = s.replace("μg", "mcg").replace("µg", "mcg")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------- Core matching ----------------
def _merge_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    """Merge selected columns into one string per row (with spaces), handling missing."""
    if not cols:
        # Fallback to blank string series of proper length
        return pd.Series([""] * len(df), index=df.index)
    return df[list(cols)].astype(str).agg(" ".join, axis=1)


def run_matching(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    code_col_left: str,
    text_cols_left: Iterable[str],
    code_col_right: str,
    text_cols_right: Iterable[str],
    ngram_range: Tuple[int, int] = (3, 5),
    min_cosine: float = 0.0,
) -> pd.DataFrame:
    """
    One-to-one (Left→best Right) nearest-neighbour matcher using char TF-IDF + cosine.
    Returns a DataFrame with codes, merged text, and similarity.

    Parameters
    ----------
    df_left, df_right : input dataframes
    code_col_left, code_col_right : columns to carry through as identifiers
    text_cols_left, text_cols_right : one or more columns to merge and compare
    ngram_range : (min_n, max_n) for char n-grams
    min_cosine : filter matches below this cosine similarity

    Returns
    -------
    DataFrame with columns:
        Left_Code, Left_Text, Right_Code, Right_Text, cosine
    """
    # Prepare comparable text
    left_raw = _merge_columns(df_left, text_cols_left)
    right_raw = _merge_columns(df_right, text_cols_right)

    left_norm = left_raw.map(normalize_text)
    right_norm = right_raw.map(normalize_text)

    # Vectorize over combined corpus to share vocabulary
    vect = TfidfVectorizer(analyzer="char", ngram_range=ngram_range, norm="l2", lowercase=False)
    X_all = vect.fit_transform(pd.concat([left_norm, right_norm], ignore_index=True))
    X_L = X_all[: len(left_norm)]
    X_R = X_all[len(left_norm) :]

    # Nearest neighbor (cosine distance)
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=1)
    nn.fit(X_R)
    dists, idxs = nn.kneighbors(X_L, n_neighbors=1, return_distance=True)

    best_j = idxs.ravel().astype(int)
    best_cos = (1.0 - dists.ravel()).astype(float)

    # Assemble results
    out = pd.DataFrame(
        {
            "Left_Code": df_left[code_col_left].to_numpy(),
            "Left_Text": left_raw.to_numpy(),
            "Right_Code": df_right.iloc[best_j][code_col_right].to_numpy(),
            "Right_Text": right_raw.iloc[best_j].to_numpy(),
            "cosine": best_cos,
        }
    )

    if min_cosine > 0:
        out = out[out["cosine"] >= min_cosine]

    return out.reset_index(drop=True)
