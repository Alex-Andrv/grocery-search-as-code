# -*- coding: utf-8 -*-
"""Данные для дообучения маленького оркестратора писать DSL-программы (search as code).

  python train/gen_data.py queries N   → train/queries_gen.jsonl  (учитель придумывает запросы покупателей)
  python train/gen_data.py programs    → train/programs.jsonl     (учитель пишет программы, runtime их проверяет)
  python train/gen_data.py sft         → train/sft_{train,dev}.jsonl (только прошедшие проверку пары)

Запросы — по случайным товарам и категориям каталога, четыре типа как в пропоузале. Запросы из eval/queries.json
и их близкие перефразы в обучение не попадают (фильтр по пересечению слов).
"""
import asyncio, json, pathlib, random, re, sys, collections
import httpx
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
import core

T = pathlib.Path(__file__).parent
CAT = core.Catalog(T.parent / "data" / "off_ru.parquet")
EVAL = [q["query"] for q in json.load(open(T.parent / "eval" / "queries.json"))]
TYPES = {
    "точный": "конкретный товар, иногда с брендом, вкусом, жирностью или объёмом",
    "с условиями": "товар с проверяемыми условиями: калорийность, белок, сахар, жирность на 100 г, без глютена/молока/орехов, без свинины, веганское",
    "сценарный": "набор продуктов под задачу: блюдо по рецепту, завтрак, пикник, ужин на компанию, иногда с ограничением по составу",
    "широкий": "размытое желание без конкретного товара: к кино, к чаю, перекус, что-нибудь полезное",
}
GEN = ("Ты придумываешь реалистичные запросы покупателей в приложении доставки продуктов (как Самокат). Пиши по-русски, "
       "разговорно, как люди пишут в поиске или ассистенту, разной длины, иногда с опечатками. Тип запросов: {t} — {d}. "
       "Опирайся на эти товары каталога (не копируй названия дословно): {seeds}. Выведи JSON-массив из {n} строк, без пояснений.")


def seeds(k=6):
    rs = random.sample(CAT.rows, k)
    return "; ".join(f"{r['name']} ({r['brands'] or ''})" for r in rs)


def too_close(q):
    a = set(core.toks(q))
    return any(len(a & set(core.toks(e))) / max(1, len(a | set(core.toks(e)))) > 0.5 for e in EVAL)


async def gen_queries(n_batches):
    sem = asyncio.Semaphore(32)
    out = []
    async with httpx.AsyncClient(trust_env=False) as cl:
        async def one(t):
            async with sem:
                u = core.Usage()
                m = await core.chat(cl, [{"role": "user", "content": GEN.format(t=t, d=TYPES[t], seeds=seeds(), n=8)}], u, "gen",
                                    max_tokens=700, temperature=1.0)
                try:
                    qs = json.loads(re.search(r"\[.*\]", m["content"], re.S).group(0))
                except Exception:
                    return []
                return [{"type": t, "query": q.strip()} for q in qs if isinstance(q, str) and 3 <= len(q) <= 160]
        res = await asyncio.gather(*[one(t) for t in TYPES for _ in range(n_batches)])
    seen = set()
    for r in res:
        for x in r:
            k = x["query"].lower()
            if k not in seen and not too_close(x["query"]):
                seen.add(k)
                out.append(x)
    with open(T / "queries_gen.jsonl", "w") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("запросов:", len(out), collections.Counter(x["type"] for x in out))


async def gen_programs():
    Q = [json.loads(l) for l in open(T / "queries_gen.jsonl")]
    sem = asyncio.Semaphore(48)
    async with httpx.AsyncClient(trust_env=False) as cl:
        async def one(x):
            async with sem:
                msgs = [{"role": "system", "content": core.DSL_DOC}]
                for q, p in core.FEWSHOT:
                    msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": f"```python\n{p}\n```"}]
                msgs.append({"role": "user", "content": x["query"]})
                u = core.Usage()
                try:
                    m = await core.chat(cl, msgs, u, "program", max_tokens=600, temperature=0.3)
                except Exception as e:
                    return {**x, "error": str(e)}
                txt = m.get("content") or ""
                mm = core.PROG_RE.search(txt)
                src = (mm.group(1) if mm else txt).strip()
                try:
                    slots, log = core.execute(CAT, src)
                    err = None
                except Exception as e:
                    slots, log, err = {}, [], f"{type(e).__name__}: {e}"
                empty = [k for k, v in slots.items() if not v]
                summ = core.summarize(CAT, slots) if not err else {}
                score = None
                if not err and slots and not empty:
                    jm = await core.chat(cl, [{"role": "system", "content": "Оцени, насколько найденные товары отвечают запросу покупателя "
                                                                            "продуктового магазина (соответствие и полнота набора). Одно число 1–5."},
                                              {"role": "user", "content": f"Запрос: {x['query']}\nНайдено: {json.dumps(summ, ensure_ascii=False)}\nОценка:"}],
                                         u, "judge", max_tokens=4, temperature=0)
                    try:
                        score = int((jm.get("content") or "0").strip()[0])
                    except Exception:
                        score = 0
                return {**x, "program": src, "error": err, "empty": empty, "n_slots": len(slots), "judge": score, "log": log}
        res = await asyncio.gather(*[one(x) for x in Q])
    with open(T / "programs.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in res if not r.get("error") and not r.get("empty") and (r.get("judge") or 0) >= 4]
    print("программ:", len(res), "ошибок:", sum(bool(r.get("error")) for r in res), "пустых:", sum(bool(r.get("empty")) for r in res),
          "прошли (judge>=4):", len(ok), collections.Counter(r["type"] for r in ok))


# системный промпт дообученной модели: без документации DSL и few-shot — она их выучила
SFT_SYS = "Напиши программу поиска товаров на DSL для запроса покупателя."


def build_sft():
    R = [json.loads(l) for l in open(T / "programs.jsonl")]
    ok = [r for r in R if not r.get("error") and not r.get("empty") and (r.get("judge") or 0) >= 4]
    random.seed(0)
    random.shuffle(ok)
    n_dev = max(50, len(ok) // 20)
    for name, part in (("dev", ok[:n_dev]), ("train", ok[n_dev:])):
        with open(T / f"sft_{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps({"messages": [{"role": "system", "content": SFT_SYS}, {"role": "user", "content": r["query"]},
                                                 {"role": "assistant", "content": f"```python\n{r['program']}\n```"}]}, ensure_ascii=False) + "\n")
        print(name, len(part))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "queries":
        asyncio.run(gen_queries(int(sys.argv[2]) if len(sys.argv) > 2 else 150))
    elif cmd == "programs":
        asyncio.run(gen_programs())
    elif cmd == "sft":
        build_sft()
