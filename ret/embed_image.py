# -*- coding: utf-8 -*-
"""Кандген по картинке: SigLIP2 (multilingual) — эмбеддинги фото товаров и текстов запросов в одном пространстве.

  python ret/embed_image.py <siglip2_dir> <emb_dir> <img_dir>
Товары без фото получают нулевой вектор (в выдачу не попадают). Выход: items_image.npy, queries_image.npy.
"""
import json, pathlib, sys
import numpy as np, torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
import core


def feats(o):
    return o if torch.is_tensor(o) else o.pooler_output

R = pathlib.Path(__file__).parent.parent
md, E, IMG = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
cat = core.Catalog(R / "data" / "off_ru.parquet")
model = AutoModel.from_pretrained(md, torch_dtype=torch.float16).cuda().eval()
proc = AutoProcessor.from_pretrained(md)
dim = model.config.text_config.hidden_size
items = np.zeros((cat.N, dim), np.float16)
have = [i for i, r in enumerate(cat.rows) if (IMG / f"{r['code']}.jpg").exists()]
print("товаров с фото:", len(have), "из", cat.N, flush=True)
with torch.no_grad():
    for k in range(0, len(have), 128):
        b = have[k:k + 128]
        ims = [Image.open(IMG / f"{cat.rows[i]['code']}.jpg").convert("RGB") for i in b]
        x = proc(images=ims, return_tensors="pt").to("cuda")
        v = feats(model.get_image_features(pixel_values=x["pixel_values"].half()))
        items[b] = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy().astype(np.float16)
    Q = json.load(open(E / "queries.json"))
    qs = []
    for k in range(0, len(Q), 256):
        x = proc(text=[q.lower() for q in Q[k:k + 256]], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
        v = feats(model.get_text_features(**x))
        qs.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy())
np.save(E / "items_image.npy", items); np.save(E / "queries_image.npy", np.concatenate(qs).astype(np.float16))
print("готово", flush=True)
