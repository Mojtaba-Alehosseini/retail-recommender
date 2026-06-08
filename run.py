"""End-to-end recommender: popularity vs item-item CF vs matrix factorization,
leave-one-out ranking metrics + sample recommendations.
Run:  python run.py   (needs data/interactions.csv — see data/README.md)
"""
import os
import pandas as pd
import recsys as R

K = 10
os.makedirs("reports", exist_ok=True)

def main():
    inter = R.load_interactions("data/interactions.csv")
    print(f"Interactions: {len(inter):,} | users: {inter['user'].nunique():,} "
          f"| items: {inter['item'].nunique():,}")
    ds = R.Dataset(inter, cap_items=1500)
    train_rows, held = R.leave_one_out_split(ds, seed=42)
    train_ui = ds.matrix(rows=train_rows)
    print(f"Evaluating on {len(held):,} held-out users\n")

    models = {"Popularity": R.Popularity(), "ItemItemCF": R.ItemItemCF(),
              "MatrixFactorization": R.MatrixFactorization(50)}
    res = {}
    for name, m in models.items():
        m.fit(train_ui); res[name] = R.evaluate(m, ds, train_ui, held, k=K)
        print(name, res[name])
    pd.DataFrame(res).T.to_csv("reports/metrics.csv")

    recs = models["ItemItemCF"].recommend(train_ui, k=K)
    desc = pd.read_csv("data/item_desc.csv").set_index("item")["description"]
    u = next(iter(held))
    with open("reports/sample_recommendations.txt", "w") as f:
        f.write(f"User {ds.users[u]} previously bought (sample):\n")
        for i in train_ui[u].indices[:8]:
            it = ds.items[i]; f.write(f"  - {it}: {desc.get(it,'')}\n")
        f.write(f"\nTop-{K} recommendations:\n")
        for i in recs[u]:
            it = ds.items[i]; f.write(f"  - {it}: {desc.get(it,'')}\n")
    print("\nSaved reports/sample_recommendations.txt")

if __name__ == "__main__":
    main()
