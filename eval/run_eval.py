# -*- coding: utf-8 -*-
"""Сравнение пайплайнов на наборе запросов: токены, вызовы LLM, latency и проверяемые кодом условия.

  python eval/run_eval.py [n_repeats]   → eval/results.jsonl, eval/summary.json

Проверки (по выбранным товарам, строка ВЫБОР в ответе):
  coverage   — доля позиций (slots), для которых выбран товар с подходящим названием;
  violations — товары, которые по данным каталога точно нарушают условие запроса;
  unverified — товары, по которым данных для проверки условия нет (например, нет состава);
  judge      — оценка 1–5 той же LLM вслепую (какой пайплайн — не говорим): насколько выдача отвечает запросу.
"""
import asyncio, json, math, pathlib, sys, time, statistics as st
import httpx
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
import core

E = pathlib.Path(__file__).parent
Q = json.load(open(E / "queries.json"))
REP = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CAT = core.Catalog(E.parent / "data" / "off_ru.parquet")
COND = ("max_kcal", "min_protein", "max_sugar", "max_fat", "exclude", "allergen_free")


def check(q, choice):
    ch = q["checks"]
    rows = [CAT.rows[CAT.by_code[c]] for c in choice if c in CAT.by_code]
    cond = {k: ch[k] for k in COND if k in ch}
    viol = unver = 0
    for r in rows:
        if cond:
            ok_loose, _ = core.item_ok(r, strict=False, **cond)
            ok_strict, _ = core.item_ok(r, strict=True, **cond)
            viol += not ok_loose
            unver += ok_loose and not ok_strict
    slots = ch.get("slots") or []
    cov = None
    if slots:
        names = [(r["name"] or "").lower().replace("ё", "е") for r in rows]
        cov = sum(any(any(a in n for a in alt) for n in names) for alt in slots) / len(slots)
    brand = None
    if ch.get("brand"):
        brand = any(ch["brand"] in (r["brands"] or "").lower() for r in rows)
    return {"n_chosen": len(choice), "n_valid": len(rows), "hallucinated": len(choice) - len(rows), "violations": viol,
            "unverified": unver, "coverage": cov, "brand": brand}


JUDGE = ("Оцени, насколько подборка товаров отвечает запросу покупателя продуктового магазина. Учитывай соответствие запросу и "
         "условиям, полноту набора. Ответь одним числом от 1 до 5.")


async def judge(client, query, choice):
    rows = [core.compact(CAT.rows[CAT.by_code[c]]) for c in choice if c in CAT.by_code]
    if not rows:
        return 1
    u = core.Usage()
    m = await core.chat(client, [{"role": "system", "content": JUDGE},
                                 {"role": "user", "content": f"Запрос: {query}\nПодборка: {json.dumps(rows, ensure_ascii=False)}\nОценка:"}],
                        u, "judge", max_tokens=5, temperature=0)
    try:
        return int((m.get("content") or "1").strip()[0])
    except Exception:
        return 1


async def one(client, q, rep, sem):
    async with sem:
        out = []
        for name, fn in (("baseline", core.run_baseline), ("code", core.run_code)):
            t = time.time()
            try:
                res = await fn(CAT, q["query"], client)
                err = None
            except Exception as e:
                res, err = {"usage": {"calls": 0, "in": 0, "out": 0}, "choice": [], "answer": ""}, f"{type(e).__name__}: {e}"
            wall = time.time() - t
            rec = {"query": q["query"], "type": q["type"], "rep": rep, "pipeline": name, "error": err, "wall": round(wall, 2),
                   **res["usage"], "choice": res["choice"], "answer": res["answer"], "trace": res.get("trace"), **check(q, res["choice"])}
            rec["judge"] = await judge(client, q["query"], res["choice"])
            out.append(rec)
        print(f"{q['query'][:40]:40s} " + " | ".join(f"{r['pipeline']}: {r['in']}+{r['out']} tok, {r['calls']} выз, judge {r['judge']}, "
                                                    f"viol {r['violations']}" for r in out), flush=True)
        return out


async def main():
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient(trust_env=False) as client:
        res = await asyncio.gather(*[one(client, q, r, sem) for r in range(REP) for q in Q])
    R = [x for pair in res for x in pair]
    with open(E / "results.jsonl", "w") as f:
        for r in R:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {}
    for p in ("baseline", "code"):
        X = [r for r in R if r["pipeline"] == p]
        def m(k, xs=X):
            v = [x[k] for x in xs if x[k] is not None]
            return round(st.mean(v), 3) if v else None
        summ[p] = {"n": len(X), "errors": sum(bool(x["error"]) for x in X), "tokens_in": m("in"), "tokens_out": m("out"),
                   "tokens_total": round(st.mean(x["in"] + x["out"] for x in X), 1), "llm_calls": m("calls"), "wall_sec": m("wall"),
                   "judge": m("judge"), "coverage": m("coverage"), "violations_per_query": m("violations"),
                   "unverified_per_query": m("unverified"), "hallucinated_per_query": m("hallucinated"), "brand_hit": m("brand"),
                   "by_type": {t: {"tokens_total": round(st.mean(x["in"] + x["out"] for x in X if x["type"] == t), 1),
                                   "judge": round(st.mean(x["judge"] for x in X if x["type"] == t), 2)}
                               for t in sorted(set(x["type"] for x in X))}}
    json.dump(summ, open(E / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


asyncio.run(main())
