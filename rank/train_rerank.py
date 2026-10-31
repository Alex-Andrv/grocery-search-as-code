# -*- coding: utf-8 -*-
"""Ранкер: кросс-энкодер bge-reranker-v2-m3, zero-shot против дообученного на разметке учителя (оценка/2 как мягкая метка).

  python rank/train_rerank.py <reranker_dir> <emb_dir> <out_model>
Оценка — переранжирование пула test-запросов (все размеченные кандидаты, перемешаны): nDCG@10 до и после обучения.
"""
import json, math, pathlib, random, sys, collections
import numpy as np
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
import core

R = pathlib.Path(__file__).parent.parent
base, E, out = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
cat = core.Catalog(R / "data" / "off_ru.parquet")
Q = json.load(open(E / "queries.json"))
N_GEN = sum(1 for _ in open(R / "train" / "queries_gen.jsonl"))
LAB = collections.defaultdict(dict)
for l in open(R / "ret" / "labels.jsonl"):
    x = json.loads(l)
    LAB[x["qi"]][x["item"]] = x["grade"]
TXT = lambda i: core.item_text(cat.rows[i])


def ndcg(ranked, lab, k=10):
    dcg = sum((2 ** lab.get(i, 0) - 1) / math.log2(r + 2) for r, i in enumerate(ranked[:k]))
    ideal = sorted(lab.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


test = [qi for qi in range(N_GEN) if qi % 10 == 0 and any(g == 2 for g in LAB[qi].values())]


def evaluate(model):
    random.seed(1)
    nd = []
    for qi in test:
        c = list(LAB[qi]); random.shuffle(c)
        s = model.predict([(Q[qi], TXT(i)) for i in c], batch_size=64, show_progress_bar=False)
        nd.append(ndcg([c[j] for j in np.argsort(-s)], LAB[qi]))
    return round(float(np.mean(nd)), 4)


m = CrossEncoder(base, num_labels=1, max_length=320, device="cuda")
zs = evaluate(m)
print("zero-shot nDCG@10:", zs, flush=True)
ex = [InputExample(texts=[Q[qi], TXT(i)], label=g / 2) for qi in range(N_GEN) if qi % 10 not in (0, 1) for i, g in LAB[qi].items()]
print("обучающих пар:", len(ex), flush=True)
m.fit(train_dataloader=DataLoader(ex, shuffle=True, batch_size=32), epochs=1, warmup_steps=200, use_amp=True, show_progress_bar=False)
ft = evaluate(m)
print("после обучения nDCG@10:", ft, flush=True)
m.save(out)
json.dump({"reranker_zero_shot_ndcg10": zs, "reranker_ft_ndcg10": ft, "n_test": len(test), "n_train_pairs": len(ex)},
          open(R / "rank" / "eval_rerank.json", "w"), indent=1)
