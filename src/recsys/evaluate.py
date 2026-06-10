"""Leave-one-out ranking evaluation: precision@k / recall@k / MAP@k."""
import numpy as np


def leave_one_out_split(ds, seed=42):
    rng = np.random.default_rng(seed)
    inter = ds.inter
    test_rows, held = [], {}
    for u, rows in inter.groupby("user").indices.items():
        if len(rows) >= 2:
            pick = int(rng.choice(rows))
            test_rows.append(pick)
            held[ds.u_index[u]] = ds.i_index[inter.iloc[pick]["item"]]
    test_set = set(test_rows)
    train_rows = [i for i in range(len(inter)) if i not in test_set]
    return train_rows, held


def evaluate(model, ds, train_ui, held, k=10):
    recs = model.recommend(train_ui, k=k)
    hits = ap = 0
    n = len(held)
    for u, true_item in held.items():
        rec = list(recs[u])
        if true_item in rec:
            hits += 1
            ap += 1.0 / (rec.index(true_item) + 1)
    return {f"precision@{k}": round(hits / (n * k), 4),
            f"recall@{k}": round(hits / n, 4),
            f"MAP@{k}": round(ap / n, 4), "hits": hits, "users": n}
