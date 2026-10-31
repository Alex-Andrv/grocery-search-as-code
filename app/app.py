# -*- coding: utf-8 -*-
"""Демка агентного поиска товаров: одна реплика → два агента параллельно, у каждого своя сессия.

  слева  — базовая линия: tool calling, BM25 + одновекторный ANN, проверку карточек делает сама LLM;
  справа — search as code: программа на DSL, многовекторный MaxSim, фото, ранкер, QA-постфильтр, дообученный оркестратор.

Запуск:  python app.py
env: LLM_URL/LLM_MODEL — большая LLM (ответы, контекст, базовая линия); ORCH_URL/ORCH_MODEL — дообученный оркестратор
     (пусто — оркестрирует большая LLM с полным промптом DSL); VERIFY_URL/VERIFY_MODEL — QA-постфильтр;
     WS — каталог с emb/ и моделями; CATALOG; PORT (7861).
"""
import asyncio, json, os, pathlib, uuid
import httpx, uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import core, agent

ROOT = pathlib.Path(__file__).parent
W = os.environ.get("WS", "/workspace")
CAT = core.Catalog(os.environ.get("CATALOG", ROOT.parent / "data" / "off_ru.parquet"))
IMG = pathlib.Path(os.environ.get("IMG_DIR", f"{W}/img"))
ORCH_URL, ORCH_MODEL = os.environ.get("ORCH_URL"), os.environ.get("ORCH_MODEL", "orch-4b")
V_URL, V_MODEL = os.environ.get("VERIFY_URL"), os.environ.get("VERIFY_MODEL", "verify-1.7b")
ORCH_SYS = "Ты оркестратор поиска продуктового магазина. Напиши программу на DSL для хода пользователя."

ret_base = ret_ours = None
if pathlib.Path(f"{W}/emb").exists():
    from retrieval import Retrieval
    ret_base = Retrieval(f"{W}/emb", bge_m3=f"{W}/models/bge-m3", ann_kind="dense")
    ret_ours = Retrieval(f"{W}/emb", bge_m3=f"{W}/m3-ft", dense_ft=f"{W}/dense-ft", siglip=f"{W}/models/siglip2",
                         reranker=f"{W}/rerank-ft", ann_kind="hybrid", mv_dir=f"{W}/emb_m3ft")
TOOLS = {"base": agent.Tools(CAT, ret_base, core.LLM_URL, core.LLM_MODEL), "ours": agent.Tools(CAT, ret_ours, V_URL, V_MODEL)}
print(f"каталог: {CAT.N} товаров; нейро-кандгены: {'да' if ret_ours else 'нет'}; оркестратор: {ORCH_URL or 'большая LLM'}", flush=True)

app = FastAPI()
client = httpx.AsyncClient(trust_env=False, timeout=300)
SESS = {}                    # sid -> {"base": Session, "ours": Session, "day": int}


class Turn(BaseModel):
    sid: str
    msg: str = ""
    run_task: bool = False
    day: int = 0
    side: str = ""          # "base" | "ours" — только один агент (фронтенд шлёт два запроса и показывает ответы по готовности)


class New(BaseModel):
    user: str = "new"


def products(choice, day):
    out = []
    for code in choice:
        i = CAT.by_code.get(code)
        if i is None:
            out.append({"code": code, "missing": True}); continue
        c = core.full_card(CAT.rows[i], 300)
        c["price"] = agent.price(CAT.rows[i], day)
        c["image"] = f"/img/{code}.jpg" if (IMG / f"{code}.jpg").exists() else CAT.rows[i]["image"]
        out.append(c)
    return out


async def one(side, s, msg, day):
    try:
        if side == "base":
            r = await agent.run_tools2(TOOLS["base"], s, msg, client, day=day)
        elif ORCH_URL:
            r = await agent.run_code2(TOOLS["ours"], s, msg, client, day=day, orch_url=ORCH_URL, orch_model=ORCH_MODEL, sys_prompt=ORCH_SYS, fewshot=False)
        else:
            r = await agent.run_code2(TOOLS["ours"], s, msg, client, day=day)
        r["products"] = products(r["choice"], day)
        r["alerts"] = agent.check_prices(TOOLS[side], s, day)
        r["tasks"] = [{k: v for k, v in t.items() if k != "item"} for t in s.tasks]
        return r
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.post("/api/new")
def new(req: New):
    sid = uuid.uuid4().hex[:12]
    SESS[sid] = {"base": agent.Session(CAT, req.user), "ours": agent.Session(CAT, req.user)}
    u = agent.USERS[SESS[sid]["base"].uid]
    hist = [core.compact(CAT.rows[h["item"]])["name"] for h in SESS[sid]["base"].history[:12]]
    return {"sid": sid, "user": u["name"], "about": u["about"], "memory": u["facts"], "history": hist}


@app.post("/api/turn")
async def turn(req: Turn):
    if req.sid not in SESS:
        raise HTTPException(404, "сессия не найдена")
    S = SESS[req.sid]
    sides = [req.side] if req.side in ("base", "ours") else ["base", "ours"]
    msgs = {}
    for side in sides:
        if req.run_task:
            task = next((t for t in S[side].tasks if t["kind"] == "schedule"), None)
            msgs[side] = agent.task_message(S[side], task, req.day) if task else None
        else:
            msgs[side] = req.msg
    async def run(side):
        if not msgs[side]:
            return {"answer": "У этого агента нет задачи по расписанию.", "choice": [], "products": [], "usage": {"calls": 0, "in": 0, "out": 0, "sec": 0},
                    "calls": [], "trace": [], "side": [], "tasks": [], "alerts": agent.check_prices(TOOLS[side], S[side], req.day), "memory": S[side].memory}
        r = await one(side, S[side], msgs[side], req.day)
        if req.run_task and not r.get("error"):
            S[side].runs.append([core.compact(CAT.rows[CAT.by_code[c]]) for c in r["choice"] if c in CAT.by_code])
        return r
    res = await asyncio.gather(*[run(s) for s in sides])
    return dict(zip(sides, res))


@app.get("/api/users")
def users():
    return [{"id": k, "name": v["name"], "about": v["about"]} for k, v in agent.USERS.items()]


@app.get("/api/presets")
def presets():
    return json.load(open(ROOT / "presets.json"))


@app.get("/img/{name}")
def img(name: str):
    p = IMG / pathlib.Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/showcase.json")
def showcase():
    p = ROOT / "static" / "showcase.json"
    return FileResponse(p) if p.exists() else {}


@app.get("/tour.js")
def tour_js():
    return FileResponse(ROOT / "static" / "tour.js", media_type="application/javascript")


@app.get("/metrics")
def metrics_page():
    return HTMLResponse((ROOT / "static" / "metrics.html").read_text())


@app.get("/")
def index():
    return HTMLResponse((ROOT / "static" / "index.html").read_text())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7861")))
