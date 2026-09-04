# Retail Product Recommender (implicit feedback)

A **personalisation / recommendation** system on real e-commerce data: it learns what
products a customer is likely to want next from their purchase history, and benchmarks
three approaches with proper **ranking metrics**.

> Dataset: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
>  ~1.07M real online-retail transactions. Cleaned to **479k** implicit interactions over
> **5.5k customers** and **4.2k products**.

## Problem
Given each customer's past purchases (implicit feedback: no ratings), recommend the
top-N products they haven't bought yet, and measure how often the held-out next purchase
actually appears in the recommendations.

## Approach
- **Cleaning** (`src/recsys/data.py`): drop cancellations, returns, missing customers and
  non-product codes (postage, fees, adjustments); keep customers/products with ≥5 records.
- **Models** (`src/recsys/models.py`):
  - **Popularity**: most-bought items (the baseline every recommender must beat)
  - **Item-item CF**: cosine similarity over the user–item matrix
  - **Matrix factorization**: Truncated-SVD latent factors
- **Evaluation** (`src/recsys/evaluate.py`): **leave-one-out**: hold out one purchase per
  customer, train on the rest, score precision@10, recall@10 and MAP@10.

## Results (reproducible: `python run.py`)

Leave-one-out over 5,519 customers (higher is better):

| Model                |  recall@10 |  MAP@10 | precision@10 |
|----------------------|-----------:|--------:|-------------:|
| Popularity baseline  |     0.049  |  0.019  |   0.005      |
| Item-item CF         |     0.172  |  0.082  |   0.017      |
| **Matrix factorization** | **0.213** | **0.094** | **0.021** |

Matrix factorization lands the held-out next purchase in the top-10 for **~21%** of
customers, more than **4×** the popularity baseline.

**Example** (item-item CF), a customer who bought parasols, umbrellas and doormats:

```
Top-10 recommendations:
  DOORMAT HEARTS · DOOR MAT ENGLISH ROSE · DOORMAT KEEP CALM AND COME IN ·
  DOOR MAT TOPIARY · EDWARDIAN PARASOL NATURAL · RED/WHITE DOTS RUFFLED UMBRELLA ...
```

## Data
`run.py` expects `data/interactions.csv` (cleaned implicit feedback) and `data/item_desc.csv`.
Build them once from the raw Excel:
```bash
# download online_retail_II.xlsx from the UCI link above
pip install python-calamine
python build_cache.py online_retail_II.xlsx   # writes data/interactions.csv + data/item_desc.csv
```

## Run it
```bash
pip install -r requirements.txt
python run.py     # reads data/interactions.csv, writes reports/
```

## Structure
```
src/recsys/
  data.py       # clean transactions -> implicit (user, item, qty)
  matrix.py     # sparse user x item matrix + id mappings
  models.py     # popularity, item-item CF, matrix factorization
  evaluate.py   # leave-one-out: precision@k / recall@k / MAP@k
run.py          # end-to-end: train, evaluate, sample recommendations
build_cache.py  # rebuild interactions.csv from the raw Excel (calamine)
```

## Notes
Implicit feedback is treated as binary "purchased". Leave-one-out with ranking metrics is
used because, for a marketplace, *what you surface in the top-10* matters more than a
rating prediction. Item cap (1500 most-popular) keeps the item–item similarity dense-friendly;
raising it trades memory for a little more coverage.
