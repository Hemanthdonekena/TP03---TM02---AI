from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from .config import MODEL_DIR, TOP_K
from .data_prep import load_data
from .features import encode_categoricals

def load_artifacts(model_dir: str = MODEL_DIR):
    clf = joblib.load(os.path.join(model_dir, "classifier.joblib"))
    pre = joblib.load(os.path.join(model_dir, "preproc.joblib"))
    seg = pd.read_csv(os.path.join(model_dir, "customer_segments.csv"))
    return clf, pre, seg

def recommend_for_customer(customer_id: int, recent_category: str | None = None, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    try:
        clf, pre, seg = load_artifacts()
    except Exception:
        # Fallback to popularity if models are missing
        popularity = _fallback_popularity()
        return [{"category": c, "score": float(s)} for c, s in popularity[:top_k]]

    # Build single-row feature vector
    segment = int(seg.loc[seg["customer_id"] == customer_id, "segment"].iloc[0]) if (seg["customer_id"] == customer_id).any() else 0
    price = 20.0  # dummy expected price
    hour = 18     # evening default
    dow = 4       # friday default
    current_category = recent_category or "misc"

    # Map categoricals using training maps
    current_map = pre["maps"]["current_category"]
    next_map = pre["maps"]["next_category"]
    current_encoded = current_map.get(current_category, -1)

    feat_vals = np.array([[price, hour, dow, segment, current_encoded]])
    proba = clf.predict_proba(feat_vals)[0]

    # Inverse map categories
    inv_next = {v:k for k,v in next_map.items()}
    ranked = sorted([(inv_next.get(i, "unknown"), float(p)) for i,p in enumerate(proba)], key=lambda x: x[1], reverse=True)
    return [{"category": cat, "score": score} for cat, score in ranked[:top_k]]

def _fallback_popularity() -> list[tuple[str, float]]:
    # Popularity by category from transactions if available
    try:
        _, products, tx = load_data("data/customers.csv", "data/products.csv", "data/transactions.csv")  # incorrect signature use to trigger except
    except Exception:
        try:
            products = pd.read_csv("data/products.csv")
            tx = pd.read_csv("data/transactions.csv")
            merged = tx.merge(products[["product_id","category"]], on="product_id", how="left")
            pop = merged["category"].value_counts(normalize=True)
            return list(pop.items())
        except Exception:
            return [("general", 1.0)]
