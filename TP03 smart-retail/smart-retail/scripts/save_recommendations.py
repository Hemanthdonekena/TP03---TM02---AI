import argparse
import pandas as pd
from src.inference import recommend_for_customer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer_id", type=int, required=True, help="Customer ID to recommend for")
    ap.add_argument("--recent_category", type=str, default=None, help="Recent category name (optional)")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--out_file", type=str, default="data/recommendations.csv")
    args = ap.parse_args()

    recs = recommend_for_customer(args.customer_id, args.recent_category, args.top_k)
    df = pd.DataFrame(recs)

    df.to_csv(args.out_file, index=False)
    print(f"✅ Saved {len(df)} recommendations for customer {args.customer_id} into {args.out_file}")
