# -*- coding: utf-8 -*-
"""Двухбашенный кандген: дообучение BGE-M3 (dense, общий энкодер для запроса и товара) на разметке учителя.

  python ret/train_dense.py <bge_m3_dir> <emb_dir> <out_model>
Обучение: запросы train-части (qi % 10 ∉ {0, 1}), позитивы — товары с оценкой 2, жёсткие негативы — оценка 0 из пула;
MultipleNegativesRankingLoss (in-batch + hard negative). После обучения пересчитывает items_dense_ft.npy и queries_dense_ft.npy.
"""
import json, pathlib, random, sys, collections
import numpy as np, torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
import core

R = pathlib.Path(__file__).parent.parent
D = pathlib.Path(__file__).parent
base, E, out = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
cat = core.Catalog(R / "data" / "off_ru.parquet")
Q = json.load(open(E / "queries.json"))
N_GEN = sum(1 for _ in open(R / "train" / "queries_gen.jsonl"))
LAB = collections.defaultdict(dict)
for l in open(D / "labels.jsonl"):
    x = json.loads(l)
    LAB[x["qi"]][x["item"]] = x["grade"]
random.seed(0)
ex = []
for qi in range(N_GEN):
    if qi % 10 in (0, 1):
        continue
    pos = [i for i, g in LAB[qi].items() if g == 2]
    neg = [i for i, g in LAB[qi].items() if g == 0]
    for p in pos[:4]:
        n = random.choice(neg) if neg else random.randrange(cat.N)
        ex.append(InputExample(texts=[Q[qi], core.item_text(cat.rows[p]), core.item_text(cat.rows[n])]))
print("обучающих троек:", len(ex), flush=True)
m = SentenceTransformer(base, device="cuda")
m.max_seq_length = 256
dl = DataLoader(ex, shuffle=True, batch_size=48)
m.fit(train_objectives=[(dl, losses.MultipleNegativesRankingLoss(m))], epochs=2, warmup_steps=100, use_amp=True, show_progress_bar=False)
m.save(out)
it = m.encode([core.item_text(r) for r in cat.rows], batch_size=128, normalize_embeddings=True, convert_to_numpy=True)
qv = m.encode(Q, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
np.save(E / "items_dense_ft.npy", it.astype(np.float16)); np.save(E / "queries_dense_ft.npy", qv.astype(np.float16))
print("сохранено", out, flush=True)
