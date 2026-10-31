# -*- coding: utf-8 -*-
"""Нейросетевые кандгены и ранкер для демки (нужен GPU).

  ann:  "dense"   — одновекторный BGE-M3, HNSW (базовая линия: обычный ANN)
        "dense_ft"— одновекторный двухбашенный BGE-M3, дообученный на разметке учителя
        "maxsim"  — многовекторный BGE-M3 (ColBERT-векторы по токенам), точный MaxSim (в стиле Perplexity)
        "muvera"  — многовекторный через MUVERA: FDE → MIPS → рескор MaxSim
        "hybrid"  — RRF двухбашенного dense_ft и дообученного многовекторного MaxSim (BGE-M3 unified fine-tuning)
  image:  SigLIP2 текст → фото товара; rerank: кросс-энкодер (zero-shot или дообученный); personal: близость к истории.
"""
import os, sys, pathlib, numpy as np, torch
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "ret"))
import indexes


class Retrieval:
    def __init__(self, emb_dir, bge_m3=None, dense_ft=None, siglip=None, reranker=None, ann_kind="dense", mv_dir=None):
        E = pathlib.Path(emb_dir)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.ann_kind = ann_kind
        self.items = np.load(E / "items_dense.npy").astype(np.float32)      # для personal
        self.m3 = None
        if bge_m3 and ann_kind in ("dense", "maxsim", "muvera", "hybrid"):
            from FlagEmbedding import BGEM3FlagModel
            self.m3 = BGEM3FlagModel(bge_m3, use_fp16=True, devices=[self.dev])
        if ann_kind == "dense":
            self.dense = indexes.Dense(self.items, hnsw=True)
        if ann_kind == "hybrid":
            from sentence_transformers import SentenceTransformer
            self.enc = SentenceTransformer(dense_ft, device=self.dev)
            self.dense = indexes.Dense(np.load(E / "items_dense_ft.npy").astype(np.float32))
            M = pathlib.Path(mv_dir)
            self.ms = indexes.MaxSim(np.load(M / "items_colbert.npy"), np.load(M / "items_colbert_off.npy"))
        if ann_kind == "dense_ft" and dense_ft:
            from sentence_transformers import SentenceTransformer
            self.enc = SentenceTransformer(dense_ft, device=self.dev)
            self.dense = indexes.Dense(np.load(E / "items_dense_ft.npy").astype(np.float32))
        if ann_kind in ("maxsim", "muvera"):
            V, off = np.load(E / "items_colbert.npy"), np.load(E / "items_colbert_off.npy")
            self.ms = indexes.MaxSim(V, off)
            self.mu = indexes.Muvera(V, off, reps=20, k_sim=5, d_proj=16) if ann_kind == "muvera" else None
        self.sig = None
        if siglip and (E / "items_image.npy").exists():
            from transformers import AutoModel, AutoProcessor
            self.sig = AutoModel.from_pretrained(siglip, torch_dtype=torch.float16).to(self.dev).eval()
            self.proc = AutoProcessor.from_pretrained(siglip)
            self.img = indexes.Dense(np.load(E / "items_image.npy").astype(np.float32))
        self.ce = None
        if reranker:
            from sentence_transformers import CrossEncoder
            self.ce = CrossEncoder(reranker, max_length=320, device=self.dev)
        self.text = None

    def _m3(self, q):
        return self.m3.encode([q], max_length=64, return_dense=True, return_colbert_vecs=True)

    def ann(self, q, k):
        if self.ann_kind == "dense":
            return [i for i, _ in self.dense.search(self._m3(q)["dense_vecs"][0], k)]
        if self.ann_kind == "dense_ft":
            return [i for i, _ in self.dense.search(self.enc.encode([q], normalize_embeddings=True)[0], k)]
        Q = self._m3(q)["colbert_vecs"][0].astype(np.float32)
        if self.ann_kind == "hybrid":
            import collections
            sc = collections.defaultdict(float)
            v = self.enc.encode([q], normalize_embeddings=True)[0]
            for res in (self.dense.search(v, max(100, k)), self.ms.search(Q, max(100, k))):
                for r, (i, _) in enumerate(res):
                    sc[i] += 1 / (60 + r)
            return sorted(sc, key=lambda i: -sc[i])[:k]
        if self.ann_kind == "maxsim":
            return [i for i, _ in self.ms.search(Q, k)]
        c = [i for i, _ in self.mu.search(Q, max(200, k))]
        s = self.ms.scores(Q, c)
        return [i for i, _ in sorted(zip(c, s), key=lambda x: -x[1])[:k]]

    @torch.no_grad()
    def image(self, q, k):
        if not self.sig:
            return []
        x = self.proc(text=[q.lower()], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.dev)
        o = self.sig.get_text_features(**x)
        o = o if torch.is_tensor(o) else o.pooler_output
        v = torch.nn.functional.normalize(o.float(), dim=-1)[0].cpu().numpy()
        return [i for i, s in self.img.search(v, k) if s > 0]

    def rerank(self, q, items):
        if not self.ce or not items:
            return items
        s = self.ce.predict([(q, self.text(i)) for i in items], batch_size=64, show_progress_bar=False)
        return [items[j] for j in np.argsort(-s)]

    def personal(self, items, hist):
        if not hist:
            return items
        c = self.items[hist].mean(0)
        return sorted(items, key=lambda i: -float(self.items[i] @ c))
