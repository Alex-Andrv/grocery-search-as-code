# -*- coding: utf-8 -*-
"""Данные для вкладок «Метрики» и «Примеры» демки → app/static/showcase.json.

  python eval/build_showcase.py
Берёт: eval/summary.json (одиночные запросы, v1), eval/summary2.json + eval/results2.jsonl (сценарные диалоги, все конфиги),
results/retrieval.json (кандгены на 452 тестовых запросах), results/eval_rerank.json, results/eval_verify.json.
Примеры — диалоги из results2.jsonl (повтор 0), где search as code выигрывает у базовой линии у попарного судьи и по токенам;
свой список можно задать в eval/showcase_pick.json: ["basket_carbonara", ...].
"""
import collections, json, pathlib, sys
R = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(R / "app"))
import core, agent

E, RES = R / "eval", R / "results"
CAT = core.Catalog(R / "data" / "off_ru.parquet")
load = lambda p: json.load(open(p)) if p.exists() else None
CFG_NAMES = {"base": "Базовая линия", "ours": "Search as code (наш)", "ours_30b": "Search as code, оркестратор 30B",
             "code_basecg": "Search as code + базовые кандгены", "tools_ourcg": "Tool calling + наши кандгены"}
KINDS = {"exact": "Точный поиск", "cond": "Условия по составу", "memory": "Память о покупателе", "basket": "Корзина под блюдо",
         "broad": "Широкий запрос", "clarify": "Уточняющий вопрос", "remember": "Запомнить и применить", "forget": "Забыть факт",
         "schedule": "Задача по расписанию", "watch": "Слежение за ценой", "followup": "Уточнение выдачи"}


def products(choice, day):
    out = []
    for code in choice:
        i = CAT.by_code.get(code)
        if i is None:
            continue
        c = core.full_card(CAT.rows[i], 300)
        c["price"] = agent.price(CAT.rows[i], day)
        c["image"] = f"/img/{code}.jpg" if (R / "data" / "img" / f"{code}.jpg").exists() else CAT.rows[i]["image"]
        out.append(c)
    return out


def turn(x, day):
    return {"msg": x.get("msg", ""), "error": x.get("error"), "answer": x.get("answer", ""), "side": x.get("side", []),
            "trace": x.get("trace", []), "usage": {k: x.get(k, 0) for k in ("calls", "in", "out", "sec")},
            "products": products(x.get("choice", []), day), "judge": x.get("judge"), "win_vs_base": x.get("win_vs_base")}


def examples_single():
    """Запасной вариант до сценарного прогона: одиночные запросы eval/results.jsonl (v1)."""
    by = collections.defaultdict(dict)
    for l in open(E / "results.jsonl"):
        x = json.loads(l)
        if x.get("rep", 0) == 0 and not x.get("error"):
            by[x["query"]][x["pipeline"]] = x
    def score(q):
        b, c = by[q]["baseline"], by[q]["code"]
        return (c["judge"] - b["judge"]) + 0.8 * min((b["in"] + b["out"]) / max(c["in"] + c["out"], 1), 6) + (c["coverage"] or 0) - (b["coverage"] or 0)
    qs = sorted((q for q in by if len(by[q]) == 2), key=lambda q: -score(q))[:8]
    out = []
    for q in qs:
        b, c = by[q]["baseline"], by[q]["code"]
        mk = lambda x: [dict(turn({**x, "msg": q}, 0), win_vs_base=None)]
        out.append({"id": q, "title": q, "scenario": b["type"].capitalize() + " запрос", "user": "новый покупатель", "base": mk(b), "ours": mk(c)})
    return out


def examples():
    p = E / "results2.jsonl"
    if not p.exists():
        return examples_single()
    X = [json.loads(l) for l in open(p)]
    scen = {s["id"]: s for s in json.load(open(E / "scenarios.json"))}
    by = collections.defaultdict(dict)
    for x in X:
        if x.get("rep", 0) == 0 and x["cfg"] in ("base", "ours"):
            by[x["id"]].setdefault(x["cfg"], []).append(x)
    score = {}
    for sid, d in by.items():
        if len(d) < 2 or any(t.get("error") for t in d["ours"]):
            continue
        tb = sum(t.get("in", 0) + t.get("out", 0) for t in d["base"])
        to = sum(t.get("in", 0) + t.get("out", 0) for t in d["ours"]) or 1
        win = [t["win_vs_base"] for t in d["ours"] if t.get("win_vs_base") is not None]
        score[sid] = (sum(win) / len(win) if win else 0.5) + 0.1 * min(tb / to, 5)
    pick = load(E / "showcase_pick.json") or sorted(score, key=lambda s: -score[s])[:8]
    out = []
    for sid in pick:
        if sid not in by:
            continue
        sc, d = scen[sid], by[sid]
        days = [t.get("day", 0) for t in sc["turns"]]
        side = {cfg: [turn(t, days[t["turn"]]) for t in sorted(d[cfg], key=lambda t: t["turn"])] for cfg in ("base", "ours")}
        out.append({"id": sid, "title": sc["turns"][0].get("msg", sid), "scenario": KINDS.get(sid.split("_")[0], ""), "user": agent.USERS[sc["user"]]["about"],
                    "base": side["base"], "ours": side["ours"]})
    return out


S = {"single": load(E / "summary.json"), "dialogs": load(E / "summary2.json"), "cfg_names": CFG_NAMES,
     "retrieval": load(RES / "retrieval.json"), "rerank": load(RES / "eval_rerank.json"), "verify": load(RES / "eval_verify.json"),
     "examples": examples()}
json.dump(S, open(R / "app" / "static" / "showcase.json", "w"), ensure_ascii=False)
print(f"примеров: {len(S['examples'])}; диалоговая сводка: {'есть' if S['dialogs'] else 'нет'}")
