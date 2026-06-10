"""Sparse user x item matrix + id mappings."""
import numpy as np
from scipy.sparse import csr_matrix


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
