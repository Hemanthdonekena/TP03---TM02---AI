from __future__ import annotations
import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from .config import MODEL_DIR, N_CLUSTERS, RANDOM_STATE
from .data_prep import build_customer_features, build_next_purchase_dataset, load_data
from .features import encode_categoricals

def train_all(data_dir: str = "data") -> None:
    customers_fp = os.path.join(data_dir, "customers.csv")
    products_fp = os.path.join(data_dir, "products.csv")
    tx_fp = os.path.join(data_dir, "transactions.csv")

    customers, products, tx = load_data(customers_fp, products_fp, tx_fp)

    # 1) Customer segmentation (KMeans)
    cust_feats = build_customer_features(customers, tx)
    seg_cols = ["n_tx","total_spent","avg_spent","n_unique_products","recency_days"]
    X_seg = cust_feats[seg_cols].values
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init="auto")
    cust_feats["segment"] = km.fit_predict(X_seg)

    # Save segmentation artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    cust_feats.to_csv(os.path.join(MODEL_DIR, "customer_segments.csv"), index=False)
    joblib.dump(km, os.path.join(MODEL_DIR, "kmeans.joblib"))

    # 2) Next-purchase classification (RandomForest by category)
    ds = build_next_purchase_dataset(tx, products)
    # Join segment to examples
    ds = ds.merge(cust_feats[["customer_id","segment"]], on="customer_id", how="left")

    # Encode categoricals
    encoded, maps = encode_categoricals(ds, ["current_category","next_category"])

    feat_cols = ["price","hour","dow","segment","current_category"]
    X = encoded[feat_cols].values
    y = encoded["next_category"].values

    clf = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X, y)

    # Save artifacts
    joblib.dump(clf, os.path.join(MODEL_DIR, "classifier.joblib"))
    joblib.dump({"feat_cols": feat_cols, "maps": maps}, os.path.join(MODEL_DIR, "preproc.joblib"))
    print("Training complete.")
