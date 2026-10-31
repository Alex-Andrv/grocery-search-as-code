#!/usr/bin/env bash
# Скачать репо HF через aria2 (параллельно): hfget.sh <repo> <dir> [токен из $HF_TOKEN]
set -euo pipefail
R=$1; D=$2; mkdir -p $D; H=(); [ -n "${HF_TOKEN:-}" ] && H=(--header="Authorization: Bearer $HF_TOKEN")
A=(); [ -n "${HF_TOKEN:-}" ] && A=(-H "Authorization: Bearer $HF_TOKEN")
curl -sf "${A[@]}" "https://huggingface.co/api/models/$R/tree/main?recursive=1" | python3 -c "
import json,sys
for f in json.load(sys.stdin):
    if f['type']=='file' and not f['path'].endswith(('.onnx','.gguf')) and not f['path'].startswith(('onnx/','openvino/')):
        print(f'https://huggingface.co/$R/resolve/main/'+f['path']); print('  out='+f['path'])" > /tmp/hf_$$.txt
aria2c -q -c -j 6 -x 8 -s 8 -k 4M --auto-file-renaming=false "${H[@]}" -d $D -i /tmp/hf_$$.txt
echo "$R → $D готово"
