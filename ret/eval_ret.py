# -*- coding: utf-8 -*-
"""Сравнение кандгенов и индексов на размеченных запросах (тестовая часть синтетики, запросы с qi % 10 == 0).

  python ret/eval_ret.py <emb_dir> [методы через запятую]   → ret/eval_ret.json, ret/unjudged.json
Метрики: nDCG@10 по оценкам 0/1/2, Recall@50 по товарам с оценкой 2 (из объединения всего размеченного),
доля неразмеченных в top-10 (дыры пула: их доразмечаем `label.py extra unjudged.json` и пересчитываем), latency на запрос.
"""
import json, math, pathlib, sys, time, collections
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import core, indexes

R = pathlib.Path(__file__).parent.parent
D = pathlib.Path(__file__).parent
E = pathlib.Path(sys.argv[1])
cat = core.Catalog(R / "data" / "off_ru.parquet")
QUERIES = json.load(open(E / "queries.json"))
N_GEN = sum(1 for _ in open(R / "train" / "queries_gen.jsonl"))
TEST = [qi for qi in range(N_GEN) if qi % 10 == 0]
LAB = collections.defaultdict(dict)
for l in open(D / "labels.jsonl"):
    x = json.loads(l)
    LAB[x["qi"]][x["item"]] = x["grade"]
TEST = [qi for qi in TEST if any(g == 2 for g in LAB[qi].values())]


def load_mv(name):
    return np.load(E / f"{name}_colbert.npy"), np.load(E / f"{name}_colbert_off.npy")


def methods(names):
    M = {}
    qd = np.load(E / "queries_dense.npy").astype(np.float32)
    if "bm25" in names:
        M["bm25"] = lambda qi, k: cat.bm25(QUERIES[qi], k)
    if {"dense", "dense_hnsw", "dense_ft"} & set(names):
        it = np.load(E / "items_dense.npy").astype(np.float32)
        if "dense" in names:
            ix = indexes.Dense(it); M["dense"] = lambda qi, k: ix.search(qd[qi], k)
        if "dense_hnsw" in names:
            ih = indexes.Dense(it, hnsw=True); M["dense_hnsw"] = lambda qi, k: ih.search(qd[qi], k)
    if (E / "items_dense_ft.npy").exists() and "dense_ft" in names:
        itf = indexes.Dense(np.load(E / "items_dense_ft.npy").astype(np.float32)); qf = np.load(E / "queries_dense_ft.npy").astype(np.float32)
        M["dense_ft"] = lambda qi, k: itf.search(qf[qi], k)
    if {"maxsim", "muvera", "muvera_rr"} & set(names):
        V, off = load_mv("items"); QV, qoff = load_mv("queries")
        Q = lambda qi: QV[qoff[qi]:qoff[qi + 1]].astype(np.float32)
        ms = indexes.MaxSim(V, off)
        if "maxsim" in names:
            M["maxsim"] = lambda qi, k: ms.search(Q(qi), k)
        if {"muvera", "muvera_rr"} & set(names):
            mu = indexes.Muvera(V, off)
            print(f"MUVERA: FDE dim {mu.dim}, сборка {mu.build_sec:.0f} c", flush=True)
            M["muvera"] = lambda qi, k: mu.search(Q(qi), k)

            def rr(qi, k):
                c = [i for i, _ in mu.search(Q(qi), 200)]
                s = ms.scores(Q(qi), c)
                return sorted(zip(c, s), key=lambda x: -x[1])[:k]
            M["muvera_rr"] = rr
    if "hybrid" in names and "dense_ft" in M and "maxsim" in M:
        def hyb(qi, k):
            sc = collections.defaultdict(float)
            for f in (M["dense_ft"], M["maxsim"]):
                for r, (i, _) in enumerate(f(qi, 100)):
                    sc[i] += 1 / (60 + r)
            return sorted(sc.items(), key=lambda x: -x[1])[:k]
        M["hybrid"] = hyb
    if (E / "items_image.npy").exists() and "image" in names:
        ii = indexes.Dense(np.load(E / "items_image.npy").astype(np.float32)); qi_ = np.load(E / "queries_image.npy").astype(np.float32)
        M["image"] = lambda qi, k: ii.search(qi_[qi], k)
    return M


def ndcg(ranked, lab, k=10):
    dcg = sum((2 ** lab.get(i, 0) - 1) / math.log2(r + 2) for r, i in enumerate(ranked[:k]))
    ideal = sorted(lab.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


names = (sys.argv[2] if len(sys.argv) > 2 else "bm25,dense,dense_hnsw,maxsim,muvera,muvera_rr,dense_ft,image").split(",")
M = methods(names)
res, unj = {}, collections.defaultdict(set)
for name, fn in M.items():
    nd, rc, hole, lat = [], [], [], []
    for qi in TEST:
        t = time.time()
        r = [i for i, _ in fn(qi, 50)]
        lat.append(time.time() - t)
        lab = LAB[qi]
        pos = {i for i, g in lab.items() if g == 2}
        nd.append(ndcg(r, lab)); rc.append(len(pos & set(r)) / len(pos))
        hole.append(sum(i not in lab for i in r[:10]) / 10)
        unj[qi] |= {i for i in r[:10] if i not in lab}
    res[name] = {"ndcg@10": round(float(np.mean(nd)), 4), "recall@50": round(float(np.mean(rc)), 4),
                 "unjudged@10": round(float(np.mean(hole)), 3), "ms_per_query": round(1000 * float(np.median(lat)), 2)}
    print(name, res[name], flush=True)
json.dump(res, open(D / "eval_ret.json", "w"), indent=1)
json.dump({str(q): sorted(int(i) for i in s) for q, s in unj.items() if s}, open(D / "unjudged.json", "w"))
print("тестовых запросов:", len(TEST))
