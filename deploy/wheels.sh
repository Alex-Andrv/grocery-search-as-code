#!/usr/bin/env bash
# Быстрая установка venv: pip резолвит (только метаданные), колёса качает aria2 в 16 потоков (одно соединение до PyPI ~0.3 МБ/с), ставим офлайн.
# wheels.sh <venv> <пакеты...>
set -euo pipefail
V=$1; shift; W=/opt/grocery/wheelhouse; mkdir -p $W
[ -x $V/bin/python ] || python3.10 -m venv $V
$V/bin/pip install -q -U pip -i https://pypi.org/simple 2>/dev/null || true
$V/bin/pip install -i https://pypi.org/simple --dry-run --ignore-installed --report /tmp/rep.json -q "$@"
python3 - <<'PY' > /tmp/urls.txt
import json
for x in json.load(open("/tmp/rep.json"))["install"]:
    print(x["download_info"]["url"])
PY
echo "колёс: $(wc -l < /tmp/urls.txt)"
cd $W && aria2c -q -c -j 8 -x 8 -s 8 -k 1M --auto-file-renaming=false -i /tmp/urls.txt
$V/bin/pip install -q --no-index --find-links $W "$@"
echo "$V готов"
