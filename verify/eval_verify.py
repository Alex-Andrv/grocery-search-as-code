# -*- coding: utf-8 -*-
"""Оценка постфильтра на dev: согласие с учителем (Qwen3-30B) и с правилом-эталоном по тексту состава.

  python verify/eval_verify.py <model_dir> [<base_model_dir>]   (vLLM offline, fp16)
Для «нарушено» важнее всего полнота: пропущенное нарушение (сказали met, а по правилу violated) — худшая ошибка.
"""
import json, pathlib, re, sys, collections
from vllm import LLM, SamplingParams

V = pathlib.Path(__file__).parent
dev = [json.loads(l) for l in open(V / "sft_dev.jsonl")]


def run(path):
    llm = LLM(path, dtype="half", max_model_len=2048, gpu_memory_utilization=0.85)
    outs = llm.chat([d["messages"][:-1] for d in dev], SamplingParams(max_tokens=120, temperature=0))
    pred = []
    for o in outs:
        try:
            t = o.outputs[0].text; pred.append(json.JSONDecoder().raw_decode(t[t.index("{"):])[0].get("verdict"))
        except Exception:
            pred.append("parse_error")
    teacher = [json.loads(d["messages"][-1]["content"])["verdict"] for d in dev]
    rule = [d["rule"] for d in dev]
    n = len(dev)
    miss = sum(p == "met" and r == "violated" for p, r in zip(pred, rule))
    viol = sum(r == "violated" for r in rule)
    res = {"n": n, "agree_teacher": round(sum(p == t for p, t in zip(pred, teacher)) / n, 3),
           "agree_rule": round(sum(p == r for p, r in zip(pred, rule)) / n, 3),
           "missed_violations": f"{miss}/{viol}", "parse_errors": pred.count("parse_error"),
           "teacher_agree_rule": round(sum(t == r for t, r in zip(teacher, rule)) / n, 3),
           "teacher_missed_violations": f"{sum(t == 'met' and r == 'violated' for t, r in zip(teacher, rule))}/{viol}"}
    del llm
    return res


out = {"sft": run(sys.argv[1])}
print(out, flush=True)
json.dump(out, open(V / "eval_verify.json", "w"), indent=1)
