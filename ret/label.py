# -*- coding: utf-8 -*-
"""Разметка релевантности «запрос — товар» учителем (Qwen3-30B) по пулу кандидатов.

  python ret/label.py pool <emb_dir>          → ret/pool.json: для каждого запроса BM25 top-20 ∪ dense top-20
  python ret/label.py judge                   → ret/labels.jsonl: оценки 0/1/2 (кэш, повторный запуск размечает только новое)
  python ret/label.py extra <cands.json>      → доразметить кандидатов новых методов ({qi: [item_idx,...]}), тот же кэш

Шкала: 2 — подходит под запрос (тот самый товар или годная позиция для набора), 1 — частично (похожий, но не то
условие/вкус/формат), 0 — не подходит. Оценка листвая: 10 товаров в одном промпте.
"""
import asyncio, json, pathlib, re, sys
import numpy as np, httpx
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
import core

R = pathlib.Path(__file__).parent.parent
D = pathlib.Path(__file__).parent
cat = core.Catalog(R / "data" / "off_ru.parquet")
QUERIES = [json.loads(l)["query"] for l in open(R / "train" / "queries_gen.jsonl")] + [q["query"] for q in json.load(open(R / "eval" / "queries.json"))]
LAB = D / "labels.jsonl"
PROMPT = ("Покупатель продуктового магазина написал запрос. Оцени каждый товар: 2 — подходит под запрос (нужный товар или годная "
          "позиция для набора под задачу, условия запроса выполнены), 1 — частично (похожий товар, но не тот вкус/формат или "
          "условие под вопросом), 0 — не подходит. Ответь JSON-объектом {{\"номер\": оценка}} для всех товаров.\n\n"
          "Запрос: {q}\n\nТовары:\n{items}")


def pool(emb):
    E = pathlib.Path(emb)
    it = np.load(E / "items_dense.npy").astype(np.float32)
    qv = np.load(E / "queries_dense.npy").astype(np.float32)
    P = {}
    for qi, q in enumerate(QUERIES):
        bm = [i for i, _ in cat.bm25(q, 20)]
        dn = list(np.argsort(-(it @ qv[qi]))[:20])
        P[qi] = list(dict.fromkeys([int(x) for x in bm + dn]))
    json.dump(P, open(D / "pool.json", "w"))
    print("пул:", sum(map(len, P.values())), "пар", flush=True)


def load_cache():
    C = {}
    if LAB.exists():
        for l in open(LAB):
            x = json.loads(l)
            C[(x["qi"], x["item"])] = x["grade"]
    return C


async def judge_pairs(todo):
    C = load_cache()
    by_q = {}
    for qi, items in todo.items():
        rest = [i for i in items if (int(qi), int(i)) not in C]
        if rest:
            by_q[int(qi)] = rest
    jobs = [(qi, its[k:k + 10]) for qi, its in by_q.items() for k in range(0, len(its), 10)]
    print("промптов к разметке:", len(jobs), flush=True)
    sem = asyncio.Semaphore(96)
    f = open(LAB, "a")
    done = [0]
    async with httpx.AsyncClient(trust_env=False) as cl:
        async def one(qi, its):
            async with sem:
                txt = "\n".join(f"{n+1}. {core.item_text(cat.rows[i], 150)}" for n, i in enumerate(its))
                u = core.Usage()
                for _ in range(3):
                    try:
                        m = await core.chat(cl, [{"role": "user", "content": PROMPT.format(q=QUERIES[qi], items=txt)}], u, "label",
                                            max_tokens=120, temperature=0)
                        g = json.loads(re.search(r"\{.*\}", m["content"], re.S).group(0))
                        break
                    except Exception:
                        g = None
                if not g:
                    return
                for n, i in enumerate(its):
                    v = g.get(str(n + 1))
                    if v in (0, 1, 2, "0", "1", "2"):
                        f.write(json.dumps({"qi": qi, "item": int(i), "grade": int(v)}) + "\n")
                done[0] += 1
                if done[0] % 2000 == 0:
                    f.flush(); print("размечено промптов", done[0], flush=True)
        await asyncio.gather(*[one(qi, its) for qi, its in jobs])
    f.close()
    print("готово, всего пар в кэше:", len(load_cache()), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "pool":
        pool(sys.argv[2])
    elif sys.argv[1] == "judge":
        asyncio.run(judge_pairs(json.load(open(D / "pool.json"))))
    elif sys.argv[1] == "extra":
        asyncio.run(judge_pairs(json.load(open(sys.argv[2]))))
