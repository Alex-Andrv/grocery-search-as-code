# -*- coding: utf-8 -*-
"""Сценарный бенчмарк по пропоузалу: базовая линия против search as code + многовекторных кандгенов, с абляциями.

  python eval/run_eval2.py [конфиги через запятую] [повторы]   → eval/results2.jsonl, eval/summary2.json

Конфиги:
  base            tool calling; BM25 + одновекторный ANN (BGE-M3 dense, HNSW); проверку карточек делает сама LLM (30B)
  ours            search as code с дообученным оркестратором Qwen3-4B (короткий промпт); кандгены: BM25 + гибрид
                  двухбашенного и многовекторного MaxSim (оба дообучены) + фото (SigLIP2) + дообученный ранкер
                  + QA-постфильтр Qwen3-1.7B (дистиллирован с 30B)
  ours_30b        то же, но оркестрирует Qwen3-30B с полным промптом DSL (вклад SFT оркестратора)
  code_basecg     абляция: search as code на кандгенах базовой линии (вклад одного search as code)
  tools_ourcg     абляция: tool calling на кандгенах «ours» (вклад одних кандгенов)
Проверки кодом на каждой реплике: тип действия, покрытие позиций, нарушения условий запроса и памяти, «без нового поиска»
для вопросов по прошлой выдаче, побочные действия (память, задачи), повтор меню между запусками задачи.
Судья (Qwen3-30B): оценка 1–5 и попарное сравнение вслепую с базовой линией (порядок случайный).
"""
import asyncio, json, os, pathlib, random, sys, time, statistics as st, collections
import httpx
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
import core, agent
from retrieval import Retrieval

E = pathlib.Path(__file__).parent
W = os.environ.get("WS", "/workspace")
SC = json.load(open(E / "scenarios.json"))
if os.environ.get("SC_IDS"):
    SC = [s for s in SC if s["id"] in os.environ["SC_IDS"].split(",")]
CAT = core.Catalog(E.parent / "data" / "off_ru.parquet")
CONFIGS = (sys.argv[1] if len(sys.argv) > 1 else "base,ours,ours_30b,code_basecg,tools_ourcg").split(",")
REP = int(sys.argv[2]) if len(sys.argv) > 2 else 1
V30 = (core.LLM_URL, core.LLM_MODEL)
V17 = (os.environ.get("VERIFY17_URL", "http://localhost:8200/v1/chat/completions"), "verify-1.7b")

_ret = {}


def retr(kind):
    if kind not in _ret:
        if kind == "base":
            _ret[kind] = Retrieval(f"{W}/emb", bge_m3=f"{W}/models/bge-m3", ann_kind="dense")
        else:
            _ret[kind] = Retrieval(f"{W}/emb", bge_m3=f"{W}/m3-ft", dense_ft=f"{W}/dense-ft", siglip=f"{W}/models/siglip2",
                                   reranker=f"{W}/rerank-ft", ann_kind="hybrid", mv_dir=f"{W}/emb_m3ft")
    return _ret[kind]


ORCH = dict(orch_url=os.environ.get("ORCH_URL", "http://localhost:8300/v1/chat/completions"), orch_model="orch-4b",
            sys_prompt="Ты оркестратор поиска продуктового магазина. Напиши программу на DSL для хода пользователя.", fewshot=False)


def setup(cfg):
    """→ (функция пайплайна, инструменты, доп. аргументы)."""
    if cfg == "base":
        return agent.run_tools2, agent.Tools(CAT, retr("base"), *V30), {}
    if cfg == "ours":
        return agent.run_code2, agent.Tools(CAT, retr("ours"), *V17), ORCH
    if cfg == "ours_30b":
        return agent.run_code2, agent.Tools(CAT, retr("ours"), *V17), {}
    if cfg == "code_basecg":
        return agent.run_code2, agent.Tools(CAT, retr("base"), *V30), {}
    if cfg == "tools_ourcg":
        return agent.run_tools2, agent.Tools(CAT, retr("ours"), *V17), {}


SEARCH_TOOLS = {"search", "semantic_search", "image_search", "purchase_history"}


def n_search(r):
    if r["pipeline"] == "tools":
        return sum(t["tool"] in SEARCH_TOOLS for t in r["trace"])
    return sum(l.split("(")[0].split(" ")[0] in ("search", "ann", "image", "history") for t in r["trace"] for l in t["log"])


def kind_of(r):
    if r["choice"]:
        return "answer"
    tail = (r["answer"] or "").strip()[-300:]
    return "clarify" if "?" in tail else "reply"


def check(turn, r, prev_choice):
    ex = turn["expect"]
    rows = [CAT.rows[CAT.by_code[c]] for c in r["choice"] if c in CAT.by_code]
    k = kind_of(r)
    out = {"kind": k, "n_search": n_search(r), "n_chosen": len(r["choice"]), "hallucinated": len(r["choice"]) - len(rows)}
    if "kind" in ex:
        out["kind_ok"] = (k == ex["kind"]) or (ex["kind"] == "reply" and out["n_search"] == 0 and bool(r["answer"]))
    if ex.get("no_search"):
        out["no_search_ok"] = out["n_search"] == 0
    if ex.get("side"):
        out["side_ok"] = any(ex["side"] in s for s in r["side"])
    cond = ex.get("cond") or {}
    if cond:
        out["violations"] = sum(not core.item_ok(x, strict=False, **cond)[0] for x in rows)
        out["unverified"] = sum(core.item_ok(x, strict=False, **cond)[0] and not core.item_ok(x, strict=True, **cond)[0] for x in rows)
    if ex.get("slots"):
        names = [(x["name"] or "").lower().replace("ё", "е") for x in rows]
        out["coverage"] = sum(any(any(a in n for a in alt) for n in names) for alt in ex["slots"]) / len(ex["slots"])
    if ex.get("brand"):
        out["brand_ok"] = any(ex["brand"] in (x["brands"] or "").lower() for x in rows)
    if ex.get("no_repeat") and prev_choice:
        out["repeat_share"] = len(set(r["choice"]) & set(prev_choice)) / max(1, len(r["choice"]))
    return out


JUDGE1 = ("Оцени ответ ассистента продуктового магазина на реплику покупателя: соответствие запросу и известным ограничениям "
          "покупателя, полнота набора, уместность действия (уточнение, ответ по прошлой выдаче, создание задачи). Одно число 1–5.")
JUDGE2 = ("Сравни два ответа ассистента продуктового магазина на одну и ту же реплику покупателя. Учитывай соответствие запросу и "
          "ограничениям покупателя, полноту и правильность товаров, уместность действия. Ответь одной буквой: A, B или =.")


def render(r):
    items = [core.compact(CAT.rows[CAT.by_code[c]]) for c in r["choice"] if c in CAT.by_code]
    return f"Текст: {(r['answer'] or '')[:1200]}\nТовары: {json.dumps(items, ensure_ascii=False)}"


async def judge1(cl, ctx, r):
    u = core.Usage()
    m = await core.chat(cl, [{"role": "system", "content": JUDGE1}, {"role": "user", "content": f"{ctx}\nОтвет:\n{render(r)}\nОценка:"}],
                        u, "judge", max_tokens=4, temperature=0)
    try:
        return int((m.get("content") or "1").strip()[0])
    except Exception:
        return 1


async def judge2(cl, ctx, a, b):
    flip = random.random() < 0.5
    x, y = (b, a) if flip else (a, b)
    u = core.Usage()
    m = await core.chat(cl, [{"role": "system", "content": JUDGE2},
                             {"role": "user", "content": f"{ctx}\n\nОтвет A:\n{render(x)}\n\nОтвет B:\n{render(y)}\n\nЛучше:"}],
                        u, "judge", max_tokens=3, temperature=0)
    v = (m.get("content") or "=").strip()[:1].upper()
    if v not in ("A", "B"):
        return 0.5
    return 1.0 if (v == "A") != flip else 0.0          # 1 — лучше первый аргумент (a)


async def dialog(cl, sc, cfg, rep):
    fn, tools, kw = setup(cfg)
    sess = agent.Session(CAT, sc["user"])
    recs, prev_task_choice = [], None
    for ti, turn in enumerate(sc["turns"]):
        day = turn.get("day", 0)
        if turn.get("run_task"):
            task = next((t for t in sess.tasks if t["kind"] == "schedule"), None)
            if not task:
                recs.append({"id": sc["id"], "turn": ti, "cfg": cfg, "rep": rep, "error": "нет задачи", "kind_ok": False}); continue
            msg = agent.task_message(sess, task, day)
        else:
            msg = turn["msg"]
        t = time.time()
        try:
            r = await fn(tools, sess, msg, cl, day=day, **kw)
            err = None
        except Exception as e:
            r, err = {"answer": "", "choice": [], "side": [], "trace": [], "pipeline": "?", "usage": {"calls": 0, "in": 0, "out": 0, "sec": 0}, "calls": []}, f"{type(e).__name__}: {e}"
        c = check(turn, r, prev_task_choice) if not err else {"kind_ok": False}
        if turn.get("run_task"):
            sess.runs.append([core.compact(CAT.rows[CAT.by_code[x]]) for x in r["choice"] if x in CAT.by_code])
            prev_task_choice = r["choice"]
        ctx = f"Покупатель: {agent.USERS[sc['user']]['about']}; память: {'; '.join(sess.memory) or '—'}\nРеплика: {msg}"
        recs.append({"id": sc["id"], "turn": ti, "cfg": cfg, "rep": rep, "msg": msg, "ctx": ctx, "error": err, "wall": round(time.time() - t, 2),
                     **r["usage"], "choice": r["choice"], "answer": r["answer"], "side": r["side"], "trace": r["trace"], **c,
                     "judge": await judge1(cl, ctx, r) if not err else 1})
    return recs


async def main():
    random.seed(0)
    sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient(trust_env=False, timeout=300) as cl:
        async def one(sc, cfg, rep):
            async with sem:
                return await dialog(cl, sc, cfg, rep)
        res = await asyncio.gather(*[one(sc, cfg, rep) for rep in range(REP) for sc in SC for cfg in CONFIGS])
        R = [x for d in res for x in d]
        # попарно с base
        by = {(x["id"], x["turn"], x["rep"], x["cfg"]): x for x in R}
        for x in R:
            if x["cfg"] != "base" and ("base" in CONFIGS):
                b = by.get((x["id"], x["turn"], x["rep"], "base"))
                if b and not x.get("error") and not b.get("error"):
                    x["win_vs_base"] = await judge2(cl, x["ctx"], x, b)
    with open(E / "results2.jsonl", "w") as f:
        for x in R:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    S = {}
    for cfg in CONFIGS:
        X = [x for x in R if x["cfg"] == cfg]

        def m(k, xs=X):
            v = [float(x[k]) for x in xs if x.get(k) is not None]
            return round(st.mean(v), 3) if v else None
        dialogs = collections.defaultdict(lambda: {"in": 0, "out": 0, "calls": 0, "sec": 0})
        for x in X:
            d = dialogs[(x["id"], x["rep"])]
            for k in ("in", "out", "calls"):
                d[k] += x.get(k, 0)
            d["sec"] += x.get("sec", 0)
        S[cfg] = {"turns": len(X), "errors": sum(bool(x.get("error")) for x in X),
                  "tokens_per_turn": round(st.mean(x.get("in", 0) + x.get("out", 0) for x in X), 1),
                  "tokens_in_per_turn": m("in"), "llm_calls_per_turn": m("calls"), "llm_sec_per_turn": m("sec"), "wall_per_turn": m("wall"),
                  "judge": m("judge"), "kind_ok": m("kind_ok"), "coverage": m("coverage"), "violations_per_turn": m("violations"),
                  "unverified_per_turn": m("unverified"), "no_search_ok": m("no_search_ok"), "side_ok": m("side_ok"), "brand_ok": m("brand_ok"),
                  "repeat_share": m("repeat_share"), "hallucinated": m("hallucinated"), "win_vs_base": m("win_vs_base")}
    json.dump(S, open(E / "summary2.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(S, ensure_ascii=False, indent=1))


asyncio.run(main())
