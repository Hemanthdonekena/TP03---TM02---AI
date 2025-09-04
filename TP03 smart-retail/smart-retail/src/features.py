from __future__ import annotations
import pandas as pd

def encode_categoricals(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, dict[str,int]]]:
    mappings: dict[str, dict[str,int]] = {}
    out = df.copy()
    for c in cols:
        uniques = sorted(out[c].dropna().unique().tolist())
        mapping = {v:i for i,v in enumerate(uniques)}
        out[c] = out[c].map(mapping).fillna(-1).astype(int)
        mappings[c] = mapping
    return out, mappings
