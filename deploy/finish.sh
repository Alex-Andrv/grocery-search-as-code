#!/usr/bin/env bash
# Довести VM до рабочей демки после setup.sh: забрать пакет с приватного HF, пересчитать многовекторный индекс, поднять сервисы.
# Запуск: sudo HF_TOKEN=... bash /opt/grocery/deploy/finish.sh   (токен не сохраняется на диск)
set -euo pipefail
G=/opt/grocery; PY=$G/venv-app/bin/python
$PY - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("alex-andrv/grocery-demo-artifacts", local_dir="/opt/grocery/ship", max_workers=8)
EOF
if [ ! -f $G/ship/emb_m3ft/items_colbert.npy ]; then
  mkdir -p $G/ship/emb_m3ft
  cd $G && $PY - <<'EOF'
# многовекторный индекс каталога дообученным BGE-M3 (как ret/embed.py, только товары)
import sys, numpy as np
sys.path.insert(0, "/opt/grocery/app")
import core
from FlagEmbedding import BGEM3FlagModel
cat = core.Catalog("/opt/grocery/data/off_ru.parquet")
m = BGEM3FlagModel("/opt/grocery/ship/m3-ft", use_fp16=True, devices=["cuda:0"])
o = m.encode([core.item_text(r) for r in cat.rows], batch_size=256, max_length=256, return_dense=False, return_sparse=False, return_colbert_vecs=True)
cv = o["colbert_vecs"]
np.save("/opt/grocery/ship/emb_m3ft/items_colbert.npy", np.concatenate(cv).astype(np.float16))
np.save("/opt/grocery/ship/emb_m3ft/items_colbert_off.npy", np.cumsum([0] + [len(x) for x in cv]))
print("colbert:", len(cv), "товаров")
EOF
fi
bash $G/deploy/setup.sh
