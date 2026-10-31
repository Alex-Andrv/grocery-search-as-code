# Фото товаров Open Food Facts (CC BY-SA) → data/img/<code>.jpg (256 px). С подов CDN недоступен — качаем с ноутбука через прокси.
import pathlib, io, sys, time, urllib.request, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
OUT = pathlib.Path("data/img")
D = pd.read_parquet("data/off_ru.parquet")
D = D[D.image.notna() & D.name.notna()].drop_duplicates("code")
todo = [(c, u) for c, u in zip(D.code, D.image) if not (OUT / f"{c}.jpg").exists()]
print("к скачиванию", len(todo), flush=True)
def one(cu):
    c, u = cu
    for a in range(3):
        try:
            b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "sirius-research/0.1"}), timeout=30).read()
            im = Image.open(io.BytesIO(b)).convert("RGB"); im.thumbnail((256, 256))
            im.save(OUT / f"{c}.jpg", quality=85); return 1
        except Exception:
            time.sleep(1 + a)
    return 0
ok, t = 0, time.time()
with ThreadPoolExecutor(24) as ex:
    for n, r in enumerate(ex.map(one, todo), 1):
        ok += r
        if n % 1000 == 0: print(n, "ok", ok, f"{n/(time.time()-t):.1f}/c", flush=True)
print("готово", ok, "из", len(todo), flush=True)
