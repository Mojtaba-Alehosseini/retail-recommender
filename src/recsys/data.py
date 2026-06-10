"""Clean raw transactions -> implicit (user, item, qty) interactions.

Dataset: UCI Online Retail II.
"""
import pandas as pd

NON_PRODUCTS = {"POST", "D", "M", "C2", "DOT", "CRUK", "PADS", "BANK CHARGES",
                "AMAZONFEE", "S", "DCGSSBOY", "DCGSSGIRL", "GIFT_0001", "ADJUST", "TEST"}


def clean_transactions(df: pd.DataFrame, min_user=5, min_item=5) -> pd.DataFrame:
    """Raw transactions -> cleaned (user, item, qty) implicit interactions."""
    df = df.rename(columns={"Customer ID": "user", "StockCode": "item", "Quantity": "qty"})
    df = df.dropna(subset=["user"]).copy()
    df["user"] = df["user"].astype(int)
    df["item"] = df["item"].astype(str).str.strip().str.upper()
    df = df[~df["Invoice"].astype(str).str.startswith("C")]   # cancellations
    df = df[df["qty"] > 0]                                     # returns
    df = df[~df["item"].isin(NON_PRODUCTS)]
    df = df[df["item"].str.len() >= 5]                         # real product codes
    inter = df.groupby(["user", "item"], as_index=False)["qty"].sum()
    for _ in range(3):
        ic = inter["item"].value_counts()
        inter = inter[inter["item"].isin(ic[ic >= min_item].index)]
        uc = inter["user"].value_counts()
        inter = inter[inter["user"].isin(uc[uc >= min_user].index)]
    return inter.reset_index(drop=True)


def load_interactions(path: str = "data/interactions.csv") -> pd.DataFrame:
    return pd.read_csv(path)
