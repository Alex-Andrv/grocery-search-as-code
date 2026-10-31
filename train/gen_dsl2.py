# -*- coding: utf-8 -*-
"""Данные SFT оркестратора на DSL v2 (все сценарии): учитель Qwen3-30B с полным промптом DSL2 + few-shot проигрывает
диалоги, в обучение идут только удачные программы. У ученика в промпте только короткая инструкция (ORCH_SYS) —
документацию DSL и примеры он выучивает.

  python train/gen_dsl2.py N   → train/dsl2_traces.jsonl, train/dsl2_sft_{train,dev}.jsonl
Диалог = случайный демо-пользователь + 1–3 реплики: запрос из синтетики (train-часть, qi % 10 ∉ {0,1}), затем с
вероятностью — уточнение/вопрос по выдаче/«запомни»/расписание/цена. Сценарии eval/scenarios.json в обучение не берём.
Фильтр: программа без ошибки, нет пустых позиций, оценка судьи результата ≥ 4 (для reply/clarify/side — судья по ответу).
"""
import asyncio, json, os, pathlib, random, sys
import httpx
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
import core, agent
from retrieval import Retrieval

T = pathlib.Path(__file__).parent
W = os.environ.get("WS", "/workspace")
CAT = core.Catalog(T.parent / "data" / "off_ru.parquet")
GEN = [json.loads(l) for l in open(T / "queries_gen.jsonl")]
TRAIN_Q = [g for k, g in enumerate(GEN) if k % 10 not in (0, 1)]
EVAL_MSGS = {t.get("msg", "").lower() for s in json.load(open(T.parent / "eval" / "scenarios.json")) for t in s["turns"]}
FOLLOW = ["а что из этого дешевле?", "а есть то же самое, но без сахара?", "сколько всё это стоит?", "замени первое на что-нибудь подешевле",
          "а без лактозы есть?", "добавь ещё что-нибудь к чаю", "а из этого что самое полезное?"]
SIDE = ["запомни: я не ем свинину", "запомни, что я люблю острое", "у меня аллергия на арахис, учти", "забудь, что я не ем свинину",
        "каждую пятницу в 19:00 собирай мне что-нибудь к фильму", "по будням к 9 утра собирай завтрак до 400 ккал",
        "сообщи, когда кофе станет дешевле 300 рублей", "напомни купить молоко каждое воскресенье"]
VAGUE = ["собери мне корзину", "хочу чего-нибудь", "купи продукты", "что посоветуешь?"]
ORCH_SYS = "Ты оркестратор поиска продуктового магазина. Напиши программу на DSL для хода пользователя."
JUDGE = ("Оцени действие ассистента продуктового магазина на реплику покупателя (подбор товаров, уточнение, ответ по прошлой выдаче, "
         "запоминание факта, создание задачи): уместность и качество. Одно число 1–5.")


async def judge(cl, ctx, r):
    items = [core.compact(CAT.rows[CAT.by_code[c]]) for c in r["choice"] if c in CAT.by_code]
    u = core.Usage()
    m = await core.chat(cl, [{"role": "system", "content": JUDGE},
                             {"role": "user", "content": f"{ctx}\nДействия: {r['side']}\nОтвет: {r['answer'][:800]}\nТовары: {json.dumps(items, ensure_ascii=False)}\nОценка:"}],
                        u, "judge", max_tokens=4, temperature=0)
    try:
        return int((m.get("content") or "0").strip()[0])
    except Exception:
        return 0


async def main(n):
    random.seed(0)
    ret = Retrieval(f"{W}/emb", bge_m3=f"{W}/models/bge-m3", siglip=f"{W}/models/siglip2",
                    reranker=f"{W}/rerank-ft" if pathlib.Path(f"{W}/rerank-ft").exists() else f"{W}/models/bge-reranker", ann_kind="maxsim")
    tools = agent.Tools(CAT, ret, os.environ.get("VERIFY17_URL", "http://localhost:8200/v1/chat/completions"), "verify-1.7b")
    plans = []
    for _ in range(n):
        uid = random.choice(list(agent.USERS))
        first = random.choice(VAGUE) if random.random() < 0.05 else random.choice(TRAIN_Q)["query"]
        turns = [first]
        x = random.random()
        if x < 0.25:
            turns.append(random.choice(FOLLOW))
        elif x < 0.45:
            turns.insert(0, random.choice(SIDE))
        elif x < 0.55:
            turns.append(random.choice(SIDE))
        plans.append((uid, [t for t in turns if t.lower() not in EVAL_MSGS]))
    sem = asyncio.Semaphore(12)
    out = []
    async with httpx.AsyncClient(trust_env=False, timeout=300) as cl:
        async def one(uid, turns):
            async with sem:
                sess, rec = agent.Session(CAT, uid), []
                for msg in turns:
                    try:
                        r = await agent.run_code2(tools, sess, msg, cl)
                    except Exception as e:
                        rec.append({"msg": msg, "error": str(e)}); break
                    ctx = f"Покупатель: {agent.USERS[uid]['about']}\nРеплика: {msg}"
                    g = await judge(cl, ctx, r)
                    rec.append({"uid": uid, "msg": msg, "kind": r["kind"], "judge": g, "trace": r["trace"], "choice": r["choice"]})
                return rec
        res = await asyncio.gather(*[one(u, t) for u, t in plans])
    with open(T / "dsl2_traces.jsonl", "w") as f:
        for d in res:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    ex = []
    for d in res:
        for r in d:
            if r.get("error") or r.get("judge", 0) < 4:
                continue
            tr = r["trace"]
            # пары (промпт фазы → последняя удачная программа фазы): фаза до контекста и после контекста
            phases = {}
            for t in tr:
                if not t["error"]:
                    phases[t["prompt_user"]] = t["program"]
            for pu, prog in phases.items():
                ex.append({"messages": [{"role": "system", "content": ORCH_SYS}, {"role": "user", "content": pu},
                                        {"role": "assistant", "content": f"```python\n{prog}\n```"}], "kind": r["kind"]})
    random.shuffle(ex)
    nd = max(60, len(ex) // 20)
    for name, part in (("dev", ex[:nd]), ("train", ex[nd:])):
        with open(T / f"dsl2_sft_{name}.jsonl", "w") as f:
            for e in part:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    import collections
    print("реплик:", sum(len(d) for d in res), "пар SFT:", len(ex), collections.Counter(e["kind"] for e in ex), flush=True)


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500))
