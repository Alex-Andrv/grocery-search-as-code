# Русские товары из Open Food Facts (ODbL): читаем food.parquet по сети, только нужные столбцы.
import time, threading, numpy as np, pandas as pd, fsspec, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor, as_completed
URL = "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet"
COLS = ["code", "countries_tags", "product_name", "generic_name", "brands", "categories_tags", "quantity", "product_quantity",
        "product_quantity_unit", "allergens_tags", "labels_tags", "ingredients_text", "nutriments", "images", "unique_scans_n", "nova_group", "nutriscore_grade"]
NUTR = {"energy-kcal": "kcal", "proteins": "protein", "fat": "fat", "carbohydrates": "carbs", "sugars": "sugar", "fiber": "fiber", "salt": "salt"}
fs = fsspec.filesystem("https", client_kwargs={"trust_env": True})
def open_pf():
    for a in range(10):
        try: return pq.ParquetFile(fs.open(URL, block_size=2 << 20))
        except Exception: time.sleep(3 + 5 * a)
nrg = open_pf().metadata.num_row_groups
local = threading.local(); out = []; lock = threading.Lock()
def txt(v, prefer="ru"):
    if not v: return None
    if isinstance(v, str): return v
    d = {x["lang"]: x["text"] for x in v if x.get("text")}
    return d.get(prefer) or d.get("main") or next(iter(d.values()), None)
def img(code, images):
    if not images: return None
    keys = {x["key"]: x for x in images}
    for k in ("front_ru", "front", "front_en"):
        if k in keys and keys[k].get("rev"):
            c = code.zfill(13) if code.isdigit() and len(code) < 13 else code
            path = f"{c[0:3]}/{c[3:6]}/{c[6:9]}/{c[9:]}" if len(c) > 8 else c
            return f"https://images.openfoodfacts.org/images/products/{path}/{k}.{keys[k]['rev']}.400.jpg"
    return None
def work(rgs):
    if not hasattr(local, "f"): local.f = open_pf()
    res = []
    for rg in rgs:
        for a in range(6):
            try: t = local.f.read_row_group(rg, columns=COLS); break
            except Exception: time.sleep(2 + 5 * a); local.f = open_pf()
        ct = t["countries_tags"].to_pylist()
        idx = [i for i, c in enumerate(ct) if c and "en:russia" in c]
        if not idx: continue
        for r in t.take(idx).to_pylist():
            nu = {NUTR[x["name"]]: x.get("100g") for x in (r["nutriments"] or []) if x.get("name") in NUTR}
            res.append(dict(code=r["code"], name=txt(r["product_name"]), generic=txt(r["generic_name"]), brands=r["brands"],
                            categories=r["categories_tags"], quantity=r["quantity"], qty=r["product_quantity"], qty_unit=r["product_quantity_unit"],
                            allergens=r["allergens_tags"], labels=r["labels_tags"], ingredients=txt(r["ingredients_text"]),
                            image=img(r["code"], r["images"]), scans=r["unique_scans_n"], nova=r["nova_group"], nutriscore=r["nutriscore_grade"], **nu))
    with lock: out.extend(res)
    return len(rgs)
chunks = [list(range(i, min(i + 16, nrg))) for i in range(0, nrg, 16)]; done, t0 = 0, time.time()
with ThreadPoolExecutor(4) as ex:
    for f in as_completed([ex.submit(work, c) for c in chunks]):
        done += f.result()
        if done % 480 < 16: print(f"{done}/{nrg} rg, товаров {len(out):,}, {(time.time()-t0)/60:.1f} мин", flush=True)
D = pd.DataFrame(out); D.to_parquet("data/off_ru.parquet")
print("готово:", len(D), "с фото:", D.image.notna().sum(), "с названием:", D.name.notna().sum(), "с БЖУ:", D.protein.notna().sum(), flush=True)
