#!/usr/bin/env bash
# Разворачивание демки на VM immers.cloud (Ubuntu 22.04 + NVIDIA). Запуск: sudo bash /opt/grocery/setup.sh
# Ожидает в /opt/grocery: ship/ (deploy/fetch_exec.sh: дельты оркестратора и постфильтра, fp16-энкодеры, эмбеддинги),
# каталоги app/ ret/ data/ (с data/img) eval/ deploy/.
set -euo pipefail
G=/opt/grocery; WS=$G/ws; ORCH=${ORCH:-orch-4b-v2}
GPU_MAIN=${GPU_MAIN:-0}; GPU_AUX=${GPU_AUX:-0}            # при 2 картах: GPU_AUX=1
UTIL_MAIN=${UTIL_MAIN:-0.52}                              # доля памяти карты под 30B (на 80 ГБ — 0.50, на отдельной 48 ГБ — 0.90)
apt-get update -qq && apt-get install -y -qq python3.10-venv python3.10-dev build-essential curl git >/dev/null
mkdir -p $WS/models && cd $WS
for d in emb emb_m3ft dense-ft rerank-ft m3-ft; do ln -sfn $G/ship/$d $WS/$d; done
ln -sfn $G/data/img $WS/img
apt-get install -y -qq aria2 >/dev/null
[ -x $G/venv-vllm/bin/vllm ] || bash $G/deploy/wheels.sh $G/venv-vllm vllm==0.11.0 "transformers<5"
[ -x $G/venv-app/bin/uvicorn ] || bash $G/deploy/wheels.sh $G/venv-app torch==2.8.0 transformers sentence-transformers FlagEmbedding faiss-cpu \
  fastapi uvicorn httpx pandas pyarrow pillow huggingface_hub
dl() { [ -d $WS/models/$2 ] || $G/venv-app/bin/huggingface-cli download $1 --local-dir $WS/models/$2 --exclude "*.onnx" "*.gguf"; }
dl Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 qwen3-30b-fp8
dl Qwen/Qwen3-4B-Instruct-2507 qwen3-4b
dl Qwen/Qwen3-1.7B qwen3-1.7b
dl BAAI/bge-m3 bge-m3
dl BAAI/bge-reranker-v2-m3 bge-reranker
dl google/siglip2-base-patch16-256 siglip2
# дообученные LLM: база + дельта (SVD ранга 32 от LoRA), конфиг и токенизатор — от дообученной модели
rebuild() {  # имя, база, дельта
  [ -f $WS/$1/model.safetensors ] && return
  $G/venv-app/bin/python $G/deploy/shrink.py apply $WS/models/$2 $G/ship/$3 $WS/$1
  cp $G/ship/meta/$1/* $WS/$1/
  cp $WS/models/$2/config.json $WS/models/$2/generation_config.json $WS/$1/   # конфиг в формате transformers 4: v5 прячет rope_theta, vLLM берёт 10000
}
rebuild orch-4b-v2 qwen3-4b orch.delta.safetensors
rebuild verify-1.7b qwen3-1.7b verify.delta.safetensors

unit() {  # имя, команда, GPU
cat > /etc/systemd/system/$1.service <<EOF
[Unit]
Description=$1
After=network-online.target
[Service]
Environment=CUDA_VISIBLE_DEVICES=$3
WorkingDirectory=$G
ExecStart=$2
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
}
V=$G/venv-vllm/bin/python
unit grocery-llm-main "$V -m vllm.entrypoints.openai.api_server --model $WS/models/qwen3-30b-fp8 --served-model-name qwen3-30b-a3b --port 8100 --max-model-len 32768 --gpu-memory-utilization $UTIL_MAIN --enable-auto-tool-choice --tool-call-parser hermes" $GPU_MAIN
unit grocery-llm-orch "$V -m vllm.entrypoints.openai.api_server --model $WS/$ORCH --served-model-name orch-4b --port 8300 --max-model-len 8192 --gpu-memory-utilization 0.13" $GPU_AUX
unit grocery-llm-verify "$V -m vllm.entrypoints.openai.api_server --model $WS/verify-1.7b --served-model-name verify-1.7b --port 8200 --max-model-len 4096 --gpu-memory-utilization 0.07" $GPU_AUX
cat > $G/app.env <<EOF
WS=$WS
IMG_DIR=$WS/img
LLM_URL=http://localhost:8100/v1/chat/completions
LLM_MODEL=qwen3-30b-a3b
ORCH_URL=http://localhost:8300/v1/chat/completions
ORCH_MODEL=orch-4b
VERIFY_URL=http://localhost:8200/v1/chat/completions
VERIFY_MODEL=verify-1.7b
PORT=7861
EOF
unit grocery-app "/bin/bash -c 'set -a; . $G/app.env; exec $G/venv-app/bin/python $G/app/app.py'" $GPU_AUX
systemctl daemon-reload
for s in grocery-llm-main grocery-llm-orch grocery-llm-verify grocery-app; do systemctl enable --now $s; done
echo "Готово. Через 3–5 минут: http://$(hostname -I | awk '{print $1}'):7861"
