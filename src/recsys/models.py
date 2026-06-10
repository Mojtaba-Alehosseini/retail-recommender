"""Models: popularity baseline, item-item CF, matrix factorization."""
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD


class Popularity:
    def fit(self, ui):
        self.scores = np.asarray(ui.sum(axis=0)).ravel()
        return self

    def recommend(self, ui, k=10):
        order = np.argsort(-self.scores)
        seen = {u: set(ui[u].indices) for u in range(ui.shape[0])}
        recs = []
        for u in range(ui.shape[0]):
            rec, s = [], seen[u]
            for it in order:
                if it not in s:
                    rec.append(it)
                    if len(rec) == k:
                        break
            recs.append(rec)
        return recs


class ItemItemCF:
    def fit(self, ui):
        iu = normalize(ui.T.tocsr(), axis=1)
        self.sim = (iu @ iu.T).toarray().astype(np.float32)
        np.fill_diagonal(self.sim, 0.0)
        return self

    def recommend(self, ui, k=10):
        return _topk(np.asarray(ui @ self.sim), ui, k)


class MatrixFactorization:
    def __init__(self, n_components=50, seed=42):
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed)

    def fit(self, ui):
        self.uf = self.svd.fit_transform(ui)
        self.iff = self.svd.components_.T
        return self

    def recommend(self, ui, k=10):
        return _topk(self.uf @ self.iff.T, ui, k)


def _topk(scores, ui, k):
    scores = scores.copy()
    scores[ui.nonzero()] = -np.inf
    return np.argpartition(-scores, k, axis=1)[:, :k]
