# -*- coding: utf-8 -*-
"""Ядро демки агентного поиска товаров: каталог, BM25, инструменты и два пайплайна поверх одной LLM.

  baseline  — tool-calling loop: модель вызывает search / get_product, каждый ответ инструмента
              (карточки товаров) возвращается ей в контекст, и так до финального ответа;
  code      — search as code: модель пишет одну короткую программу на DSL, runtime проверяет и исполняет её
              (поиск, фильтры по полям и составу, ранжирование), модели возвращается только сжатая сводка.

Оба пайплайна используют одни и те же функции поиска и фильтрации, разница только в том, кто ими управляет.
"""
import ast, json, math, os, re, time, collections
import numpy as np, pandas as pd, httpx

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8100/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3-30b-a3b")
MAX_TOOL_STEPS = 8          # лимит шагов tool-calling loop
MAX_PROGRAMS = 2            # не больше двух итераций поиска (как в пропоузале)
CARD_INGR = 300             # сколько символов состава отдаём в карточке baseline

# ---------------------------------------------------------------- каталог и BM25
WORD = re.compile(r"[a-zа-я0-9]+")


def stem(w):
    return w[:6] if len(w) > 6 else w


def toks(s):
    return [stem(w) for w in WORD.findall((s or "").lower().replace("ё", "е"))]


def tag_text(tags):
    return " ".join(t.split(":", 1)[-1].replace("-", " ") for t in (tags or []))


class Catalog:
    def __init__(self, path):
        D = pd.read_parquet(path)
        D = D[D.name.notna() & (D.name.str.len() > 1)].drop_duplicates("code").reset_index(drop=True)
        for c in ("kcal", "protein", "fat", "carbs", "sugar", "fiber", "salt"):
            D[c] = pd.to_numeric(D[c], errors="coerce")
        for c in ("categories", "allergens", "labels"):
            D[c] = D[c].map(lambda v: list(v) if v is not None and len(v) else [])
        D["ingr_lc"] = D.ingredients.fillna("").str.lower().str.replace("ё", "е")
        self.D = D
        self.rows = D.to_dict("records")
        self.by_code = {r["code"]: i for i, r in enumerate(self.rows)}
        docs = [toks(" ".join(str(x) for x in (r["name"], r["generic"], r["brands"], tag_text(r["categories"])) if x)) for r in self.rows]
        self.df = collections.Counter(t for d in docs for t in set(d))
        self.post = collections.defaultdict(list)
        for i, d in enumerate(docs):
            for t, c in collections.Counter(d).items():
                self.post[t].append((i, c))
        self.dl = np.array([len(d) for d in docs], float)
        self.avgdl = self.dl.mean()
        self.N = len(docs)
        # популярность как слабый приор при равных BM25
        self.pop = np.log1p(D.scans.fillna(0).values.astype(float))

    def bm25(self, q, k=50, k1=1.2, b=0.75):
        sc = np.zeros(self.N)
        for t in set(toks(q)):
            if t not in self.post:
                continue
            idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, tf in self.post[t]:
                sc[i] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.dl[i] / self.avgdl))
        nz = np.nonzero(sc)[0]
        if not len(nz):
            return []
        sc2 = sc[nz] + 0.01 * self.pop[nz]
        o = nz[np.argsort(-sc2)][:k]
        return [(int(i), float(sc[i])) for i in o]


# ---------------------------------------------------------------- общие функции поиска и фильтрации
ALLERGEN_RU = {"gluten": "глютен", "milk": "молоко", "eggs": "яйца", "nuts": "орехи", "peanuts": "арахис", "soybeans": "соя",
               "fish": "рыба", "crustaceans": "ракообразные", "sesame-seeds": "кунжут", "celery": "сельдерей", "mustard": "горчица"}


# аллерген по-русски/по-английски → (тег OFF, корни слов в названии и составе)
ALLERGEN_RU = {k: v for keys, v in [
    (("молок", "milk", "dairy", "лакто", "молоч"), ("milk", ("молок", "молоч", "сливк", "сыворот", "лактоз", "казеин"))),
    (("яйцо", "яйца", "яиц", "eggs", "egg"), ("eggs", ("яйц", "яичн", "меланж"))),
    (("орех", "nuts", "nut"), ("nuts", ("орех", "фундук", "миндал", "кешью", "фисташ"))),
    (("арахи", "peanu"), ("peanuts", ("арахис",))),
    (("глюте", "glute", "пшени"), ("gluten", ("пшени", "глютен", "ячмен", "ржан"))),
    (("соя", "soy", "soybe"), ("soybeans", ("соев", "соя"))),
    (("рыба", "fish"), ("fish", ("рыб",))),
    (("мясо", "meat", "свини", "pork"), ("meat", ("свин", "говя", "мяс", "бекон", "желатин"))),
] for k in keys}


def item_ok(r, max_kcal=None, min_protein=None, max_sugar=None, max_fat=None, exclude=None, allergen_free=None, labels=None, strict=True):
    """Проверка условий по полям и составу. Возвращает (ok, причина). Неизвестное значение при strict — не подходит."""
    def num(field, val, cmp):
        v = r.get(field)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return not strict, f"нет данных: {field}"
        return cmp(v, val), f"{field}={v:g}"
    for field, val, cmp in (("kcal", max_kcal, lambda v, x: v <= x), ("protein", min_protein, lambda v, x: v >= x),
                            ("sugar", max_sugar, lambda v, x: v <= x), ("fat", max_fat, lambda v, x: v <= x)):
        if val is not None:
            ok, why = num(field, val, cmp)
            if not ok:
                return False, why
    if exclude:
        ing = r.get("ingr_lc") or ""
        if not ing and strict:
            return False, "нет состава"
        for w in exclude:
            w = w.lower().replace("ё", "е")
            if w and (w in ing or w in (r["name"] or "").lower()):
                return False, f"в составе «{w}»"
    if allergen_free:
        al = set(a.split(":")[-1] for a in (r.get("allergens") or []))
        text = (r.get("ingr_lc") or "") + " " + (r["name"] or "").lower()
        for a in allergen_free:
            tag, roots = ALLERGEN_RU.get(str(a).lower().replace("ё", "е")[:5], (str(a).lower(), ()))
            if tag in al or any(w in text for w in roots):
                return False, f"аллерген {a}"
    if labels:
        lb = " ".join(r.get("labels") or [])
        if not all(l in lb for l in labels):
            return False, "нет метки"
    return True, ""


def item_text(r, ingr_chars=200):
    """Текст товара для эмбеддеров и ранкера: название, бренд, вес, категории, начало состава."""
    cats = ", ".join(t.split(":", 1)[-1].replace("-", " ") for t in (r["categories"] or [])[:5])
    ing = (r["ingredients"] or "")[:ingr_chars]
    return " | ".join(x for x in (r["name"], r["brands"], r["quantity"], cats, ing and "Состав: " + ing) if x)


def compact(r):
    """Короткая запись товара для сводки search-as-code."""
    def f(x):
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 1)
    d = {"code": r["code"], "name": r["name"], "brand": r["brands"], "qty": r["quantity"], "kcal": f(r["kcal"]), "protein": f(r["protein"]),
         "fat": f(r["fat"]), "sugar": f(r["sugar"])}
    return {k: v for k, v in d.items() if v is not None}


def full_card(r, ingr_chars=CARD_INGR):
    """Карточка товара, как её отдаёт обычный инструмент поиска: всё, что нужно модели для проверки условий."""
    def f(x):
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 1)
    ing = r["ingredients"] or ""
    return {"code": r["code"], "name": r["name"], "brand": r["brands"], "quantity": r["quantity"],
            "categories": [t.split(":", 1)[-1] for t in (r["categories"] or [])][:6],
            "per_100g": {k: f(r[k]) for k in ("kcal", "protein", "fat", "carbs", "sugar", "salt")},
            "allergens": [a.split(":")[-1] for a in (r["allergens"] or [])], "labels": [l.split(":")[-1] for l in (r["labels"] or [])][:6],
            "ingredients": ing[:ingr_chars] + ("…" if len(ing) > ingr_chars else "")}


# ---------------------------------------------------------------- LLM
class Usage:
    def __init__(self):
        self.calls = []

    def add(self, kind, u, dt):
        self.calls.append({"kind": kind, "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0), "sec": round(dt, 2)})

    def total(self):
        return {"calls": len(self.calls), "in": sum(c["in"] for c in self.calls), "out": sum(c["out"] for c in self.calls),
                "sec": round(sum(c["sec"] for c in self.calls), 2)}


async def chat(client, messages, usage, kind, tools=None, max_tokens=1500, temperature=0.2):
    body = {"model": LLM_MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    t = time.time()
    r = await client.post(LLM_URL, json=body, timeout=180)
    r.raise_for_status()
    j = r.json()
    usage.add(kind, j.get("usage", {}), time.time() - t)
    return j["choices"][0]["message"]


CHOICE_RE = re.compile(r"ВЫБОР\s*:\s*(\[[^\]]*\])", re.S)


def parse_choice(text):
    m = CHOICE_RE.search(text or "")
    if not m:
        return []
    try:
        return [str(x) for x in json.loads(m.group(1))]
    except Exception:
        return re.findall(r"\d{8,14}", m.group(1))


ANSWER_RULES = ("Ответь пользователю по-русски коротко: что подобрал и почему, какие условия не удалось проверить или выполнить. "
                "Используй только товары из результатов поиска. В последней строке выведи коды выбранных товаров в формате "
                "ВЫБОР: [\"код1\", \"код2\", ...]")

# ---------------------------------------------------------------- пайплайн 1: tool-calling loop
TOOLS = [
    {"type": "function", "function": {
        "name": "search", "description": "Поиск товаров в каталоге продуктового магазина. Возвращает карточки товаров: название, бренд, "
                                         "вес, КБЖУ на 100 г, аллергены, метки и начало состава.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "поисковый запрос, например «спагетти» или «греческий йогурт»"},
            "k": {"type": "integer", "description": "сколько карточек вернуть (по умолчанию 10, максимум 20)"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_product", "description": "Полная карточка товара по коду, включая весь состав.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]

SYS_BASELINE = ("Ты — ассистент продуктового магазина с доставкой. Помогаешь найти товары в каталоге. "
                "Для поиска вызывай инструменты; проверяй условия пользователя (калорийность, белок, состав, аллергены) по карточкам. "
                "Для набора продуктов найди каждую нужную позицию отдельным поиском. Если продукт по природе нарушает условие "
                "(бекон при «без свинины»), ищи замену. Поисковые запросы — короткие слова, как на полке магазина. "
                "Когда товаров достаточно, дай финальный ответ. " + ANSWER_RULES)


async def run_baseline(cat, query, client):
    usage, trace = Usage(), []
    msgs = [{"role": "system", "content": SYS_BASELINE}, {"role": "user", "content": query}]
    seen = {}
    for step in range(MAX_TOOL_STEPS):
        m = await chat(client, msgs, usage, "tool-loop", tools=TOOLS)
        calls = m.get("tool_calls") or []
        if not calls:
            return {"answer": m.get("content") or "", "choice": parse_choice(m.get("content")), "usage": usage.total(),
                    "calls": usage.calls, "trace": trace, "seen": seen}
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        for tc in calls:
            fn = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except Exception:
                args = {}
            if fn == "search":
                k = max(1, min(int(args.get("k") or 10), 20))
                res = [full_card(cat.rows[i]) for i, _ in cat.bm25(args.get("query", ""), k)]
                for c in res:
                    seen[c["code"]] = c
                out = json.dumps(res, ensure_ascii=False)
                trace.append({"step": step, "tool": "search", "args": args, "n": len(res), "chars": len(out)})
            elif fn == "get_product":
                i = cat.by_code.get(str(args.get("code")))
                res = full_card(cat.rows[i], 5000) if i is not None else {"error": "нет такого товара"}
                out = json.dumps(res, ensure_ascii=False)
                trace.append({"step": step, "tool": "get_product", "args": args, "chars": len(out)})
            else:
                out = json.dumps({"error": "нет такого инструмента"})
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
    # лимит шагов исчерпан: просим ответить по найденному
    msgs.append({"role": "user", "content": "Лимит поиска исчерпан. Ответь по уже найденным товарам. " + ANSWER_RULES})
    m = await chat(client, msgs, usage, "tool-loop")
    return {"answer": m.get("content") or "", "choice": parse_choice(m.get("content")), "usage": usage.total(),
            "calls": usage.calls, "trace": trace, "seen": seen}


# ---------------------------------------------------------------- пайплайн 2: search as code
DSL_DOC = """Ты — планировщик поиска продуктового магазина. Вместо пошаговых вызовов инструментов ты пишешь ОДНУ короткую
программу на DSL (подмножество Python), runtime её исполняет и возвращает сводку. Разрешены только присваивания и вызовы:

  search(query, k=50)            -> пул кандидатов из каталога (BM25 по названию, бренду, категориям)
  union(pool1, pool2, ...)       -> объединение пулов (Reciprocal Rank Fusion)
  filt(pool, max_kcal=None, min_protein=None, max_sugar=None, max_fat=None,
       exclude=[...], allergen_free=[...])
                                 -> оставить товары, которые проходят условия; КБЖУ на 100 г;
                                    exclude — подстроки, которых не должно быть в составе (например "свинин", "бекон");
                                    allergen_free — коды аллергенов: gluten, milk, eggs, nuts, peanuts, soybeans, fish, sesame-seeds;
                                    товар без нужных данных отбрасывается
  rank(pool, by="relevance")     -> сортировка: relevance | protein | -kcal | -sugar | protein_per_kcal
  top(pool, n)                   -> первые n
  answer(slot1=pool, slot2=pool) -> результат: именованные позиции корзины (для одного товара — одна позиция)

Правила:
- каждая нужная позиция (продукт корзины) — отдельный слот answer; разные продукты в один пул не объединяй;
  union — только для синонимов одного продукта («спагетти» и «паста»);
- фильтры ставь только из явных условий пользователя (состав, аллергены, КБЖУ); не добавляй своих;
- если продукт по природе нарушает условие (бекон при «без свинины»), ищи замену («индейка», «курица»);
- поисковые запросы — короткие слова, как на полке магазина («спагетти», «пармезан», «сливки»);
- последняя строка — вызов answer(...); других функций и конструкций нет.
Выведи только программу в блоке ```python```."""

FEWSHOT = [
    ("молоко простоквашино 3,2", 'p = search("молоко простоквашино 3,2", k=20)\nanswer(milk=top(p, 3))'),
    ("собери завтрак: высокобелковый йогурт и мюсли без сахара",
     'y = rank(filt(search("йогурт"), min_protein=6), by="protein")\n'
     'm = filt(union(search("мюсли"), search("гранола")), max_sugar=5)\n'
     'answer(yogurt=top(y, 3), muesli=top(m, 3))'),
    ("продукты на борщ, без говядины",
     'ex = ["говяд"]\n'
     'answer(beet=top(search("свекла"), 2), cabbage=top(search("капуста"), 2), potato=top(search("картофель"), 2),\n'
     '       meat=top(filt(union(search("курица"), search("индейка")), exclude=ex), 2), tomato=top(search("томатная паста"), 2),\n'
     '       sour_cream=top(search("сметана"), 2))'),
]

ALLOWED = {"search", "union", "filt", "rank", "top", "answer"}
MAX_STMTS, MAX_SEARCH = 25, 12


class Pool(list):
    pass


def validate(src):
    tree = ast.parse(src)
    if len(tree.body) > MAX_STMTS:
        raise ValueError("слишком длинная программа")
    n_search = 0
    for node in ast.walk(tree):
        ok = isinstance(node, (ast.Module, ast.Assign, ast.Expr, ast.Call, ast.Name, ast.Constant, ast.List, ast.Tuple,
                               ast.keyword, ast.Load, ast.Store, ast.UnaryOp, ast.USub))
        if not ok:
            raise ValueError(f"недопустимая конструкция: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED:
                raise ValueError("недопустимая функция")
            n_search += node.func.id == "search"
        if isinstance(node, ast.Assign) and not all(isinstance(t, ast.Name) for t in node.targets):
            raise ValueError("присваивать можно только переменным")
    if n_search > MAX_SEARCH:
        raise ValueError("слишком много вызовов search")
    return tree


def execute(cat, src):
    """Исполняет проверенную программу. Возвращает (слоты: имя -> список индексов, лог шагов)."""
    tree = validate(src)
    log, result = [], {}

    def search(query, k=50):
        p = Pool(i for i, _ in cat.bm25(str(query), max(1, min(int(k), 200))))
        log.append(f"search({query!r}) -> {len(p)}")
        return p

    def union(*pools):
        sc = collections.defaultdict(float)
        for p in pools:
            for r, i in enumerate(p):
                sc[i] += 1 / (60 + r)
        p = Pool(sorted(sc, key=lambda i: -sc[i]))
        log.append(f"union -> {len(p)}")
        return p

    def filt(pool, **kw):
        kw = {k: v for k, v in kw.items() if k in ("max_kcal", "min_protein", "max_sugar", "max_fat", "exclude", "allergen_free")}
        p = Pool(i for i in pool if item_ok(cat.rows[i], **kw)[0])
        log.append(f"filt({', '.join(f'{k}={v}' for k, v in kw.items())}) {len(pool)} -> {len(p)}")
        return p

    def rank(pool, by="relevance"):
        key = {"protein": lambda i: -(cat.rows[i]["protein"] or 0), "-kcal": lambda i: cat.rows[i]["kcal"] if cat.rows[i]["kcal"] == cat.rows[i]["kcal"] else 1e9,
               "-sugar": lambda i: cat.rows[i]["sugar"] if cat.rows[i]["sugar"] == cat.rows[i]["sugar"] else 1e9,
               "protein_per_kcal": lambda i: -((cat.rows[i]["protein"] or 0) / max(cat.rows[i]["kcal"] or 1, 1))}.get(by)
        return Pool(sorted(pool, key=key)) if key else Pool(pool)

    def top(pool, n=3):
        return Pool(pool[:max(1, min(int(n), 10))])

    def answer(**slots):
        for k, v in slots.items():
            if not isinstance(v, Pool):
                raise ValueError(f"answer: слот {k} должен быть пулом товаров")
        result.update({k: list(v) for k, v in slots.items()})

    env = {"search": search, "union": union, "filt": filt, "rank": rank, "top": top, "answer": answer, "__builtins__": {}}
    exec(compile(tree, "<dsl>", "exec"), env)
    return result, log


def summarize(cat, slots):
    return {k: [compact(cat.rows[i]) for i in v] or "НИЧЕГО НЕ НАЙДЕНО" for k, v in slots.items()}


PROG_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S)


async def run_code(cat, query, client):
    usage, trace = Usage(), []
    msgs = [{"role": "system", "content": DSL_DOC}]
    for q, p in FEWSHOT:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": f"```python\n{p}\n```"}]
    msgs.append({"role": "user", "content": query})
    slots, summary = {}, {}
    for it in range(MAX_PROGRAMS):
        m = await chat(client, msgs, usage, "program", max_tokens=600)
        txt = m.get("content") or ""
        mm = PROG_RE.search(txt)
        src = (mm.group(1) if mm else txt).strip()
        try:
            slots, log = execute(cat, src)
            err = None
        except Exception as e:
            log, err = [], f"{type(e).__name__}: {e}"
        summary = summarize(cat, slots) if not err else {}
        trace.append({"iter": it + 1, "program": src, "log": log, "error": err})
        empty = [k for k, v in summary.items() if isinstance(v, str)]
        if not err and not empty and slots:
            break
        fb = f"Ошибка программы: {err}" if err else f"Пустые позиции: {', '.join(empty) or 'нет результата'}. Сводка: {json.dumps(summary, ensure_ascii=False)}"
        msgs += [{"role": "assistant", "content": txt}, {"role": "user", "content": fb + "\nИсправь программу (переформулируй запросы или ослабь только неявные условия)."}]
    ans_msgs = [{"role": "system", "content": "Ты — ассистент продуктового магазина. " + ANSWER_RULES},
                {"role": "user", "content": f"Запрос пользователя: {query}\n\nРезультаты поиска (условия программы уже проверены runtime по полям и составу):\n"
                                            f"{json.dumps(summary, ensure_ascii=False)}"}]
    m = await chat(client, ans_msgs, usage, "answer", max_tokens=700)
    seen = {c["code"]: c for v in summary.values() if isinstance(v, list) for c in v}
    return {"answer": m.get("content") or "", "choice": parse_choice(m.get("content")), "usage": usage.total(),
            "calls": usage.calls, "trace": trace, "seen": seen}
