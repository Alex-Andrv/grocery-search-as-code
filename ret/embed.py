# -*- coding: utf-8 -*-
"""Эмбеддинги BGE-M3 для каталога и запросов: dense (1 вектор) и многовекторные ColBERT-векторы по токенам.

  python ret/embed.py <model_dir> <out_dir>
Выход: items_dense.npy [N,1024] fp16, items_colbert.npy (все токен-векторы подряд, fp16) + items_colbert_off.npy,
       queries_*.npy аналогично для train/queries_gen.jsonl и eval/queries.json (порядок: gen, затем eval).
"""
import json, pathlib, sys, time, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
import core
from FlagEmbedding import BGEM3FlagModel

R = pathlib.Path(__file__).parent.parent
model_dir, out = sys.argv[1], pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
cat = core.Catalog(R / "data" / "off_ru.parquet")
m = BGEM3FlagModel(model_dir, use_fp16=True, devices=["cuda:0"])


def run(texts, name, maxlen):
    t = time.time()
    o = m.encode(texts, batch_size=128, max_length=maxlen, return_dense=True, return_sparse=False, return_colbert_vecs=True)
    np.save(out / f"{name}_dense.npy", o["dense_vecs"].astype(np.float16))
    cv = o["colbert_vecs"]
    off = np.cumsum([0] + [len(x) for x in cv])
    np.save(out / f"{name}_colbert.npy", np.concatenate(cv).astype(np.float16))
    np.save(out / f"{name}_colbert_off.npy", off)
    print(f"{name}: {len(texts)} текстов, {off[-1]} токен-векторов, {time.time()-t:.0f} c", flush=True)


run([core.item_text(r) for r in cat.rows], "items", 256)
gen = [json.loads(l)["query"] for l in open(R / "train" / "queries_gen.jsonl")]
ev = [q["query"] for q in json.load(open(R / "eval" / "queries.json"))]
json.dump(gen + ev, open(out / "queries.json", "w"), ensure_ascii=False)
run(gen + ev, "queries", 64)
