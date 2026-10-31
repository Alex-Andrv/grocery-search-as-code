# -*- coding: utf-8 -*-
"""Индексы для кандгенов демки: dense (flat / HNSW), многовекторный MaxSim (точный) и MUVERA (FDE → MIPS → MaxSim-рескор).

MUVERA (Dhulipala et al., 2024, arXiv 2405.19504): многовекторный документ сводится к одному вектору фиксированной
длины (Fixed Dimensional Encoding) так, что скалярное произведение FDE приближает Chamfer/MaxSim. Строим R повторов:
SimHash на k_sim гиперплоскостях делит пространство на 2^k_sim корзин; в корзине документ усредняет свои векторы,
запрос — суммирует; пустые корзины документа заполняются ближайшим по Хэммингу вектором; затем случайная проекция в d_proj.
"""
import time, numpy as np, torch


class Dense:
    def __init__(self, X, hnsw=False, M=32, ef=128):
        self.X = torch.tensor(X, dtype=torch.float16, device="cuda") if torch.cuda.is_available() else torch.tensor(X, dtype=torch.float32)
        self.hnsw = None
        if hnsw:
            import faiss
            self.hnsw = faiss.IndexHNSWFlat(X.shape[1], M, faiss.METRIC_INNER_PRODUCT)
            self.hnsw.hnsw.efConstruction = 200
            self.hnsw.add(X.astype(np.float32))
            self.hnsw.hnsw.efSearch = ef

    def search(self, q, k=50):
        if self.hnsw is not None:
            s, i = self.hnsw.search(q.reshape(1, -1).astype(np.float32), k)
            return [(int(a), float(b)) for a, b in zip(i[0], s[0]) if a >= 0]
        qt = torch.tensor(q, dtype=self.X.dtype, device=self.X.device)
        s = self.X @ qt
        v, i = torch.topk(s.float(), k)
        return list(zip(i.tolist(), v.tolist()))


class MaxSim:
    """Точный MaxSim по всем токен-векторам каталога (на GPU)."""
    def __init__(self, V, off):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.V = torch.tensor(V, dtype=torch.float16 if dev == "cuda" else torch.float32, device=dev)
        self.off = off
        self.doc = torch.tensor(np.repeat(np.arange(len(off) - 1), np.diff(off)), device=dev)
        self.N = len(off) - 1

    def scores(self, Q, docs=None):
        qt = torch.tensor(Q, dtype=self.V.dtype, device=self.V.device)
        if docs is None:
            S = (self.V @ qt.T).float()                                   # [tokens, q]
            best = torch.full((self.N, S.shape[1]), -1e4, device=S.device)
            best = best.scatter_reduce(0, self.doc[:, None].expand_as(S), S, reduce="amax")
            return best.sum(1)
        out = []
        for d in docs:
            a, b = self.off[d], self.off[d + 1]
            out.append(float((self.V[a:b] @ qt.T).float().max(0).values.sum()))
        return out

    def search(self, Q, k=50):
        s = self.scores(Q)
        v, i = torch.topk(s, k)
        return list(zip(i.tolist(), v.tolist()))


class Muvera:
    def __init__(self, V, off, reps=10, k_sim=4, d_proj=16, seed=0):
        rng = np.random.default_rng(seed)
        d = V.shape[1]
        self.G = rng.standard_normal((reps, d, k_sim)).astype(np.float32)
        self.S = (rng.choice([-1.0, 1.0], size=(reps, d, d_proj)) / np.sqrt(d_proj)).astype(np.float32)
        self.reps, self.k, self.dp = reps, k_sim, d_proj
        self.B = 2 ** k_sim
        self.pw = 2 ** np.arange(k_sim)
        t = time.time()
        F = np.stack([self.fde(V[off[i]:off[i + 1]].astype(np.float32), doc=True) for i in range(len(off) - 1)])
        self.build_sec = time.time() - t
        self.dim = F.shape[1]
        self.index = Dense(F)

    def fde(self, X, doc):
        parts = []
        for r in range(self.reps):
            bits = (X @ self.G[r] > 0)
            bucket = bits.astype(np.int64) @ self.pw
            out = np.zeros((self.B, X.shape[1]), np.float32)
            cnt = np.bincount(bucket, minlength=self.B)
            np.add.at(out, bucket, X)
            if doc:
                nz = cnt > 0
                out[nz] /= cnt[nz, None]
                if (~nz).any():
                    for b in np.where(~nz)[0]:
                        # ближайший по Хэммингу вектор документа к корзине b
                        dist = np.array([bin(int(x) ^ int(b)).count("1") for x in bucket])
                        out[b] = X[int(np.argmin(dist))]
            parts.append((out @ self.S[r]).ravel())
        return np.concatenate(parts)

    def search(self, Q, k=50):
        return self.index.search(self.fde(Q.astype(np.float32), doc=False), k)
