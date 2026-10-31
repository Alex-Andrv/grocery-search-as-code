# многовекторный индекс каталога дообученным BGE-M3 (как ret/embed.py, только товары)
import sys, time, numpy as np
sys.path.insert(0, "/opt/grocery/app")
import core
from FlagEmbedding import BGEM3FlagModel
t = time.time()
cat = core.Catalog("/opt/grocery/data/off_ru.parquet")
m = BGEM3FlagModel("/opt/grocery/ship/m3-ft", use_fp16=True, devices=["cuda:0"])
o = m.encode([core.item_text(r) for r in cat.rows], batch_size=256, max_length=256, return_dense=False, return_sparse=False, return_colbert_vecs=True)
cv = o["colbert_vecs"]
np.save("/opt/grocery/ship/emb_m3ft/items_colbert.npy", np.concatenate(cv).astype(np.float16))
np.save("/opt/grocery/ship/emb_m3ft/items_colbert_off.npy", np.cumsum([0] + [len(x) for x in cv]))
print("colbert:", len(cv), "товаров,", round(time.time() - t), "с")
