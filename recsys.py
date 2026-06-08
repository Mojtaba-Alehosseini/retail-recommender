"""
Retail product recommender (implicit feedback) — self-contained module.

Cleaning + sparse user/item matrix + three models (popularity, item-item CF,
matrix factorization) + leave-one-out ranking evaluation.

Dataset: UCI Online Retail II.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD

NON_PRODUCTS = {"POST", "D", "M", "C2", "DOT", "CRUK", "PADS", "BANK CHARGES",
                "AMAZONFEE", "S", "DCGSSBOY", "DCGSSGIRL", "GIFT_0001", "ADJUST", "TEST"}

# --------------------------------------------------------------------------- data
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

# ------------------------------------------------------------------------- matrix
class Dataset:
    def __init__(self, inter, cap_items: int | None = 1500):
        if cap_items:
            top = inter["item"].value_counts().head(cap_items).index
            inter = inter[inter["item"].isin(top)]
        self.users = inter["user"].unique().tolist()
        self.items = inter["item"].unique().tolist()
        self.u_index = {u: i for i, u in enumerate(self.users)}
        self.i_index = {it: j for j, it in enumerate(self.items)}
        self.inter = inter.reset_index(drop=True)

    def matrix(self, rows=None) -> csr_matrix:
        df = self.inter if rows is None else self.inter.iloc[rows]
        r = df["user"].map(self.u_index).to_numpy()
        c = df["item"].map(self.i_index).to_numpy()
        return csr_matrix((np.ones(len(df), np.float32), (r, c)),
                          shape=(len(self.users), len(self.items)))

# ------------------------------------------------------------------------- models
class Popularity:
    def fit(self, ui):
        self.scores = np.asarray(ui.sum(axis=0)).ravel(); return self
    def recommend(self, ui, k=10):
        order = np.argsort(-self.scores)
        seen = {u: set(ui[u].indices) for u in range(ui.shape[0])}
        recs = []
        for u in range(ui.shape[0]):
            rec, s = [], seen[u]
            for it in order:
                if it not in s:
                    rec.append(it)
                    if len(rec) == k: break
            recs.append(rec)
        return recs

class ItemItemCF:
    def fit(self, ui):
        iu = normalize(ui.T.tocsr(), axis=1)
        self.sim = (iu @ iu.T).toarray().astype(np.float32)
        np.fill_diagonal(self.sim, 0.0); return self
    def recommend(self, ui, k=10):
        return _topk(np.asarray(ui @ self.sim), ui, k)

class MatrixFactorization:
    def __init__(self, n_components=50, seed=42):
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)
    def fit(self, ui):
        self.uf = self.svd.fit_transform(ui); self.iff = self.svd.components_.T; return self
    def recommend(self, ui, k=10):
        return _topk(self.uf @ self.iff.T, ui, k)

def _topk(scores, ui, k):
    scores = scores.copy(); scores[ui.nonzero()] = -np.inf
    return np.argpartition(-scores, k, axis=1)[:, :k]

# ----------------------------------------------------------------------- evaluate
def leave_one_out_split(ds, seed=42):
    rng = np.random.default_rng(seed)
    inter = ds.inter
    test_rows, held = [], {}
    for u, rows in inter.groupby("user").indices.items():
        if len(rows) >= 2:
            pick = int(rng.choice(rows)); test_rows.append(pick)
            held[ds.u_index[u]] = ds.i_index[inter.iloc[pick]["item"]]
    test_set = set(test_rows)
    train_rows = [i for i in range(len(inter)) if i not in test_set]
    return train_rows, held

def evaluate(model, ds, train_ui, held, k=10):
    recs = model.recommend(train_ui, k=k)
    hits = ap = 0; n = len(held)
    for u, true_item in held.items():
        rec = list(recs[u])
        if true_item in rec:
            hits += 1; ap += 1.0 / (rec.index(true_item) + 1)
    return {f"precision@{k}": round(hits / (n * k), 4),
            f"recall@{k}": round(hits / n, 4),
            f"MAP@{k}": round(ap / n, 4), "hits": hits, "users": n}
