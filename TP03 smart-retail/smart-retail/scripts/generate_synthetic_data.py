from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
np.random.seed(42)

CATS = ["grocery","household","electronics","fashion","beauty","sports","toys","books","pet","misc"]

def make_customers(n=1000):
    return pd.DataFrame({
        "customer_id": np.arange(1, n+1),
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.randint(30000, 150000, size=n),
        "region": np.random.choice(["north","south","east","west"], size=n, p=[0.25,0.25,0.25,0.25])
    })

def make_products(n=500):
    return pd.DataFrame({
        "product_id": np.arange(1, n+1),
        "category": np.random.choice(CATS, size=n),
        "base_price": np.round(np.random.uniform(5, 200, size=n), 2)
    })

def make_transactions(customers, products, n_tx=10000, days=120):
    rows = []
    start = datetime.now() - timedelta(days=days)
    for _ in range(n_tx):
        cid = np.random.choice(customers["customer_id"])
        prod = products.sample(1).iloc[0]
        ts = start + timedelta(minutes=int(np.random.uniform(0, days*24*60)))
        price = float(np.round(prod.base_price * np.random.uniform(0.8, 1.2), 2))
        rows.append((len(rows)+1, cid, int(prod.product_id), price, ts))
    tx = pd.DataFrame(rows, columns=["transaction_id","customer_id","product_id","price","timestamp"])
    return tx

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--n_customers", type=int, default=500)
    ap.add_argument("--n_products", type=int, default=250)
    ap.add_argument("--n_transactions", type=int, default=5000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    customers = make_customers(args.n_customers)
    products = make_products(args.n_products)
    tx = make_transactions(customers, products, n_tx=args.n_transactions)

    customers.to_csv(os.path.join(args.out_dir, "customers.csv"), index=False)
    products.to_csv(os.path.join(args.out_dir, "products.csv"), index=False)
    tx.to_csv(os.path.join(args.out_dir, "transactions.csv"), index=False)

    print(f"Wrote customers/products/transactions to {args.out_dir}")
