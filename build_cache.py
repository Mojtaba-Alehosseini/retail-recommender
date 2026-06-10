"""One-off: read the downloaded Online Retail II Excel and cache small CSVs.

Usage:
    pip install python-calamine
    python build_cache.py path/to/online_retail_II.xlsx
"""
import sys
import pandas as pd
from src.recsys import clean_transactions

def main(xlsx="online_retail_II.xlsx"):
    frames = []
    for sh in ["Year 2009-2010", "Year 2010-2011"]:
        frames.append(pd.read_excel(
            xlsx, sheet_name=sh, engine="calamine",
            usecols=["Invoice", "StockCode", "Quantity", "Customer ID", "Price", "Description"]))
    raw = pd.concat(frames, ignore_index=True)
    inter = clean_transactions(raw)
    inter.to_csv("data/interactions.csv", index=False)
    desc = (raw.assign(StockCode=raw["StockCode"].astype(str).str.strip().str.upper())
               .dropna(subset=["Description"])
               .groupby("StockCode")["Description"].first().reset_index())
    desc.columns = ["item", "description"]
    desc.to_csv("data/item_desc.csv", index=False)
    print(f"cached {len(inter):,} interactions, {len(desc):,} item descriptions")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "online_retail_II.xlsx")
