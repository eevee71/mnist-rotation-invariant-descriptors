import numpy as np
from sklearn.decomposition import PCA


class BlockWhitener:

    def __init__(self, inv, chiral_weight=0.0, var_keep=0.999, seed=0):
        self.ach = list(inv.achiral_idx)
        self.chi = list(inv.chiral_idx)
        self.chiral_weight = chiral_weight
        self.var_keep = var_keep
        self.seed = seed
        self.pca_a = None
        self.pca_c = None


    def fit(self, raw):
        raw = self._np(raw)
        self.pca_a = PCA(n_components=self.var_keep, whiten=True,
                         random_state=self.seed).fit(raw[:, self.ach])
        if self.chiral_weight > 0 and len(self.chi) > 0:
            self.pca_c = PCA(n_components=self.var_keep, whiten=True,
                             random_state=self.seed).fit(raw[:, self.chi])
        return self


    def transform(self, raw):
        raw = self._np(raw)
        a = self.pca_a.transform(raw[:, self.ach])
        if self.pca_c is not None:
            c = self.pca_c.transform(raw[:, self.chi]) * self.chiral_weight
            return np.concatenate([a, c], axis=1)
        return a                                    # chiral_weight=0 -> skip Im block


    def fit_transform(self, raw):
        return self.fit(raw).transform(raw)


    @staticmethod
    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)