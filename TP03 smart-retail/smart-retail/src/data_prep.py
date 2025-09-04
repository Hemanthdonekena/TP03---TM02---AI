from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple

def load_data(customers_fp: str, products_fp: str, transactions_fp: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(customers_fp)
    products = pd.read_csv(products_fp)
    tx = pd.read_csv(transactions_fp, parse_dates=['timestamp'])
    return customers, products, tx

def build_customer_features(customers: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    agg = tx.groupby('customer_id').agg(
        n_tx=('transaction_id', 'count'),
        total_spent=('price', 'sum'),
        avg_spent=('price', 'mean'),
        n_unique_products=('product_id', 'nunique'),
        recency_days=('timestamp', lambda s: (tx['timestamp'].max() - s.max()).days),
    ).reset_index()
    out = customers.merge(agg, on='customer_id', how='left').fillna(0)
    return out

def build_next_purchase_dataset(tx: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    # Simplified next-product dataset: for each (customer, session order) predict next product category
    tx = tx.sort_values(['customer_id', 'timestamp'])
    tx['next_product_id'] = tx.groupby('customer_id')['product_id'].shift(-1)
    # join categories
    prod_cats = products[['product_id','category']].rename(columns={'category':'current_category'})
    tx = tx.merge(prod_cats, on='product_id', how='left')
    next_cats = products[['product_id','category']].rename(columns={'product_id':'next_product_id','category':'next_category'})
    tx = tx.merge(next_cats, on='next_product_id', how='left')
    # Drop rows without a next purchase
    tx = tx.dropna(subset=['next_category']).copy()
    # Simple numeric features
    feats = tx[['customer_id','product_id','price']].copy()
    feats['hour'] = tx['timestamp'].dt.hour
    feats['dow'] = tx['timestamp'].dt.dayofweek
    feats['current_category'] = tx['current_category']
    feats['next_category'] = tx['next_category']
    return feats
