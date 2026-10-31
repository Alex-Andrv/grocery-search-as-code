# -*- coding: utf-8 -*-
"""Постфильтр (аналог OCC-RAG из пропоузала): по карточке товара ответить, выполнено ли требование покупателя.

  python verify/gen_verify.py          → verify/pairs.jsonl (учитель), verify/sft_{train,dev}.jsonl
Вердикт: met | violated | unknown, плюс дословная цитата из карточки. Требования — типовые ограничения по составу;
карточки — товары из BM25 по «опасным» для требования категориям (чтобы были и нарушения, и выполнения).
Для части требований есть правило-эталон по тексту состава: им проверяем и учителя, и ученика.
"""
import asyncio, json, pathlib, random, re, sys
import httpx
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "app"))
import core

R = pathlib.Path(__file__).parent.parent
V = pathlib.Path(__file__).parent
cat = core.Catalog(R / "data" / "off_ru.parquet")
# требование → (запросы для подбора карточек, подстроки-нарушения для правила-эталона)
REQ = {
    "без свинины": (["колбаса", "сосиски", "пельмени", "ветчина", "фарш", "паштет", "бекон", "котлеты"], ["свин", "бекон", "шпик", "сало "]),
    "без глютена": (["печенье", "хлеб", "макароны", "крупа", "хлопья", "колбаса"], ["пшенич", "рожь", "ржан", "ячмен", "овсян", "глютен"]),
    "без молока и лактозы": (["шоколад", "печенье", "каша", "мороженое", "соус", "хлеб"], ["молок", "сливк", "лактоз", "сыворот", "масло сливоч"]),
    "без сахара": (["йогурт", "сок", "печенье", "мюсли", "соус", "чай"], ["сахар", "сироп", "глюкоз", "фруктоз", "декстроз"]),
    "без орехов": (["шоколад", "печенье", "мюсли", "батончик", "паста"], ["орех", "фундук", "миндал", "кешью", "арахис"]),
    "без пальмового масла": (["печенье", "шоколад", "мороженое", "вафли", "сыр"], ["пальмов"]),
    "вегетарианское": (["пельмени", "суп", "соус", "консервы", "чипсы", "колбаса"], ["мясо", "говяд", "свин", "куриц", "рыб", "желатин"]),
    "без консервантов": (["колбаса", "соус", "майонез", "сок", "хлеб"], ["консервант", "сорбат", "бензоат", "нитрит"]),
}
SYS = ("Ты проверяешь карточку товара. Ответь JSON: {\"verdict\": \"met\" | \"violated\" | \"unknown\", \"quote\": \"...\"}. "
       "met — по карточке видно, что требование выполнено; violated — видно, что нарушено; unknown — в карточке нет сведений "
       "(например, нет состава). quote — дословный фрагмент карточки, на который опираешься (пусто при unknown).")


def card(r):
    c = core.full_card(r, 1200)
    return json.dumps({k: c[k] for k in ("name", "brand", "categories", "allergens", "labels", "ingredients")}, ensure_ascii=False)


def rule(req, r):
    ing = r["ingr_lc"]
    if not ing:
        return "unknown"
    return "violated" if any(w in ing for w in REQ[req][1]) else "met"


async def main():
    random.seed(0)
    pairs = []
    for req, (qs, _) in REQ.items():
        pool = sorted({i for q in qs for i, _ in cat.bm25(q, 300)})
        with_ing = [i for i in pool if cat.rows[i]["ingr_lc"]]
        no_ing = [i for i in pool if not cat.rows[i]["ingr_lc"]]
        # 85% карточек с составом: иначе почти все ответы «нет данных» и проверка бессодержательна
        pairs += [(req, i) for i in random.sample(with_ing, min(400, len(with_ing)))]
        pairs += [(req, i) for i in random.sample(no_ing, min(70, len(no_ing)))]
    sem = asyncio.Semaphore(96)
    async with httpx.AsyncClient(trust_env=False) as cl:
        async def one(req, i):
            async with sem:
                u = core.Usage()
                msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": f"Требование: {req}\nКарточка: {card(cat.rows[i])}"}]
                try:
                    m = await core.chat(cl, msgs, u, "verify", max_tokens=150, temperature=0)
                    j = json.loads(re.search(r"\{.*\}", m["content"], re.S).group(0))
                except Exception:
                    return None
                if j.get("verdict") not in ("met", "violated", "unknown"):
                    return None
                return {"req": req, "item": i, "teacher": j, "rule": rule(req, cat.rows[i]), "messages": msgs}
        res = [r for r in await asyncio.gather(*[one(q, i) for q, i in pairs]) if r]
    with open(V / "pairs.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps({k: r[k] for k in ("req", "item", "teacher", "rule")}, ensure_ascii=False) + "\n")
    agree = sum(r["teacher"]["verdict"] == r["rule"] for r in res) / len(res)
    print("пар:", len(res), "согласие учителя с правилом:", round(agree, 3), flush=True)
    random.shuffle(res)
    nd = len(res) // 10
    for name, part in (("dev", res[:nd]), ("train", res[nd:])):
        with open(V / f"sft_{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps({"messages": r["messages"] + [{"role": "assistant", "content": json.dumps(r["teacher"], ensure_ascii=False)}],
                                    "req": r["req"], "item": r["item"], "rule": r["rule"]}, ensure_ascii=False) + "\n")
        print(name, len(part), flush=True)


asyncio.run(main())
