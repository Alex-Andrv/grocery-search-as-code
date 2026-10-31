# -*- coding: utf-8 -*-
"""Многоходовой агент демки: сценарии из пропоузала поверх двух пайплайнов с одними и теми же инструментами.

Сценарии: точный запрос, запрос с условиями, сценарная корзина (рецепт через инструмент контекста, замены),
широкий запрос с персонализацией (история покупок, факты из памяти), уточняющий вопрос, вопрос по прошлой выдаче
без нового поиска, задача по расписанию (обед к 13:00, КБЖУ на весь набор, без повтора меню), мониторинг цены.

  tool calling   — модель вызывает инструменты по одному, все ответы инструментов копятся в контексте;
  search as code — модель пишет программу на DSL, runtime исполняет; в контекст модели идёт только сводка.

Кандгены: BM25 (search), двухбашенный dense / многовекторный MaxSim (ann), SigLIP2 по фото (image), история
покупок (history); ранкер — кросс-энкодер (rerank); постфильтр — QA-проверка карточки с цитатой (verify).
Цены и история покупок синтетические (в Open Food Facts их нет), это помечено в интерфейсе.
"""
import ast, hashlib, json, math, os, re, time, collections
import numpy as np
import core
from core import Pool, Usage, chat, compact, item_ok, parse_choice, PROG_RE

VERIFY_URL = os.environ.get("VERIFY_URL", core.LLM_URL)
VERIFY_MODEL = os.environ.get("VERIFY_MODEL", core.LLM_MODEL)
MAX_PROGRAMS = 3

# ---------------------------------------------------------------- синтетические цены
PRICE_PER_KG = [("сыр", 900), ("пармез", 2200), ("колбас", 800), ("сосис", 600), ("мяс", 700), ("куриц", 380), ("индейк", 550),
                ("рыб", 900), ("икр", 6000), ("шокол", 1100), ("конфет", 900), ("печен", 450), ("чай", 1600), ("кофе", 2000),
                ("масло", 600), ("молок", 95), ("кефир", 110), ("йогурт", 280), ("творог", 450), ("сметан", 300), ("сливк", 350),
                ("хлеб", 180), ("макарон", 200), ("спагет", 220), ("круп", 150), ("греч", 160), ("рис", 170), ("мук", 70),
                ("сахар", 80), ("сок", 120), ("вод", 40), ("овощ", 180), ("фрукт", 250), ("орех", 1400), ("мёд", 900), ("мед ", 900)]


def grams(r):
    try:
        q = float(str(r.get("qty") or "").replace(",", "."))
    except ValueError:
        q = None
    if q and r.get("qty_unit") in ("g", "ml") and 5 <= q <= 5000:
        return q
    m = re.search(r"(\d+[.,]?\d*)\s*(г|гр|g|мл|ml|кг|kg|л|l)\b", (r.get("quantity") or "").lower())
    if m:
        v = float(m.group(1).replace(",", "."))
        return v * 1000 if m.group(2) in ("кг", "kg", "л", "l") else v
    return None


def price(r, day=0):
    """Синтетическая цена, ₽: по категории за кг, вес упаковки, детерминированный шум; day сдвигает цену (для мониторинга)."""
    name = (r["name"] or "").lower() + " " + core.tag_text(r["categories"])
    per_kg = next((p for k, p in PRICE_PER_KG if k in name), 400)
    g = grams(r) or 300
    h = int(hashlib.md5(f"{r['code']}:{day}".encode()).hexdigest(), 16)
    noise = 0.75 + (h % 1000) / 1000 * 0.6
    if day and h % 7 == 0:
        noise *= 0.8                                                        # иногда скидка
    return round(max(29, per_kg * g / 1000 * noise) / 10) * 10 - 1


# ---------------------------------------------------------------- демо-пользователи (синтетическая история покупок)
USERS = {
    "anna": {"name": "Анна", "about": "спорт, следит за белком, не ест свинину",
             "facts": ["не ест свинину", "старается есть больше белка"],
             "history_queries": ["творог", "куриная грудка", "греческий йогурт", "протеиновый батончик", "яйца", "гречка", "зелёный чай"]},
    "semya": {"name": "Семья Петровых", "about": "двое детей, покупают на всю семью",
              "facts": ["в семье двое детей", "у младшего аллергия на орехи"],
              "history_queries": ["молоко", "каша детская", "печенье", "сок", "сыр", "макароны", "сосиски", "яблоки"]},
    "vegan": {"name": "Илья", "about": "веган, сладкоежка",
              "facts": ["веган: без мяса, молока и яиц", "любит тёмный шоколад"],
              "history_queries": ["тофу", "растительное молоко", "тёмный шоколад", "хумус", "орехи", "овсяное печенье", "чечевица"]},
    "new": {"name": "Новый пользователь", "about": "истории нет", "facts": [], "history_queries": []},
}


def build_history(cat, uid):
    rows = []
    for k, q in enumerate(USERS[uid]["history_queries"]):
        for j, (i, _) in enumerate(cat.bm25(q, 3)):
            rows.append({"item": i, "times": 6 - 2 * j - (k % 2)})
    return rows


# ---------------------------------------------------------------- сессия
class Session:
    def __init__(self, cat, uid="new"):
        self.uid = uid if uid in USERS else "new"
        self.memory = list(USERS[self.uid]["facts"])
        self.history = build_history(cat, self.uid)
        self.dialog = []            # [{"role","content"}] — только тексты реплик
        self.last = {}              # сводка последней выдачи: слот -> список compact
        self.tasks = []             # задачи scheduler
        self.runs = []              # прошлые запуски задач (для «не повторять меню»)


# ---------------------------------------------------------------- инструменты (общие для обоих пайплайнов)
class Tools:
    def __init__(self, cat, ret=None, verify_url=None, verify_model=None):
        self.cat, self.ret = cat, ret     # ret — Retrieval (ann/image/rerank), может быть None
        self.vurl, self.vmodel = verify_url or VERIFY_URL, verify_model or VERIFY_MODEL
        if ret is not None:
            ret.text = lambda i: core.item_text(cat.rows[i])

    # кандгены
    def search(self, q, k):
        return [i for i, _ in self.cat.bm25(q, k)]

    def ann(self, q, k):
        return self.ret.ann(q, k) if self.ret else self.search(q, k)

    def image(self, q, k):
        return self.ret.image(q, k) if self.ret else []

    def history(self, sess, k):
        return [h["item"] for h in sorted(sess.history, key=lambda h: -h["times"])][:k]

    def rerank(self, q, items):
        return self.ret.rerank(q, items) if self.ret else items

    def personal_rank(self, sess, items):
        if not self.ret or not sess.history:
            return items
        return self.ret.personal(items, [h["item"] for h in sess.history])

    async def context(self, client, usage, question):
        m = await chat(client, [{"role": "system", "content": "Ты кулинарный и продуктовый справочник. Отвечай кратко, списком, по-русски."},
                                {"role": "user", "content": question}], usage, "context", max_tokens=250, temperature=0)
        return (m.get("content") or "").strip()

    async def verify(self, client, usage, items, requirement, n=12):
        """QA-постфильтр по карточке: met / violated / unknown + цитата."""
        import httpx
        out = {}
        from asyncio import gather

        async def one(i):
            r = self.cat.rows[i]
            card = core.full_card(r, 1200)
            msgs = [{"role": "system", "content": VERIFY_SYS},
                    {"role": "user", "content": f"Требование: {requirement}\nКарточка: " + json.dumps(
                        {k: card[k] for k in ("name", "brand", "categories", "allergens", "labels", "ingredients")}, ensure_ascii=False)}]
            body = {"model": self.vmodel, "messages": msgs, "max_tokens": 220, "temperature": 0}
            t = time.time()
            try:
                j = (await client.post(self.vurl, json=body, timeout=60)).json()
                usage.add("verify", j.get("usage", {}), time.time() - t)
                t = j["choices"][0]["message"]["content"]; v = json.JSONDecoder().raw_decode(t[t.index("{"):])[0]
            except Exception:
                v = {"verdict": "unknown", "quote": ""}
            out[i] = v
        await gather(*[one(i) for i in items[:n]])
        return out

    def price(self, i, day=0):
        return price(self.cat.rows[i], day)


VERIFY_SYS = ("Ты проверяешь карточку товара. Ответь JSON: {\"verdict\": \"met\" | \"violated\" | \"unknown\", \"quote\": \"...\"}. "
              "met — по карточке видно, что требование выполнено; violated — видно, что нарушено; unknown — в карточке нет сведений "
              "(например, нет состава). quote — короткий (до 12 слов) дословный фрагмент карточки, на который опираешься (пусто при unknown).")


def card_line(cat, i, day=0, extra=None):
    r = cat.rows[i]
    d = compact(r)
    d["price"] = price(r, day)
    g = grams(r)
    if g and r["kcal"] == r["kcal"] and r["kcal"] is not None:
        d["kcal_pack"] = round(r["kcal"] * g / 100)
    if g and r["protein"] == r["protein"] and r["protein"] is not None:
        d["protein_pack"] = round(r["protein"] * g / 100, 1)
    if extra:
        d.update(extra)
    return d


# ---------------------------------------------------------------- пайплайн search as code
DSL2 = """Ты — оркестратор поиска продуктового магазина. На каждый ход пользователя пишешь ОДНУ короткую программу на DSL
(подмножество Python: только присваивания и вызовы функций ниже). Runtime исполняет её и возвращает сводку.

Кандгены (возвращают пул товаров):
  search(q, k=50)       BM25 по названию, бренду, категориям — точные названия и бренды
  ann(q, k=50)          семантический поиск (эмбеддинги) — описательные и широкие запросы
  image(q, k=50)        поиск по фото товара — «что-то яркое», вид упаковки, «как на картинке»
  history(k=30)         товары из истории покупок пользователя
Обработка пула:
  union(p1, p2, ...)    объединение (RRF), только для одной позиции
  filt(p, max_kcal=, min_protein=, max_sugar=, max_fat=, max_price=, exclude=[...], allergen_free=[...])
  rank(p, by="relevance" | "protein" | "-kcal" | "-sugar" | "price" | "personal")
  rerank(p, q)          точный ранкер (кросс-энкодер) по запросу q — ставь перед top для важных позиций
  verify(p, "требование")  проверка карточек QA-моделью (состав, «без свинины», «веганское»); оставляет подтверждённые
  top(p, n)
Контекст, память, задачи:
  context("вопрос")     справка: состав рецепта, на что смотреть при выборе. Результат вернётся тебе — затем напиши новую программу
  remember("факт"), forget("факт")   долговременные факты о пользователе (ограничения, вкусы)
  schedule(when="пн-пт 13:00", request="...")   регулярная задача; watch_price(p, below=цена)   уведомить о снижении цены
Завершение хода (ровно одно, последней строкой):
  answer(slot1=p, slot2=p, ...)   выдача; одна позиция корзины — один слот
  clarify("вопрос")               уточнить, если без ответа нельзя подобрать (не спрашивай, если можно разумно предположить)
  reply("текст")                  ответить по прошлой выдаче или подтвердить действие без нового поиска

Правила:
- КАЖДЫЙ слот answer — свой поиск под свой продукт: answer(pasta=top(search("спагетти"),2), eggs=top(search("яйца"),2)).
  Нельзя собирать разные продукты в один пул и раздавать его слотам;
- явные ограничения пользователя и факты из памяти обязательны (exclude/allergen_free/verify); своих фильтров не добавляй;
- продукт, нарушающий условие по природе (бекон при «не ест свинину»), заменяй другим продуктом (индейка, курица);
- вопрос о прошлой выдаче («что дешевле», «а без сахара есть?» по уже показанному) — reply(...) без нового поиска;
- для сценарного запроса без известного состава сначала context(...);
- широкие запросы при наличии истории: union(ann(...), history()) и rank(by="personal");
- запрос про ВНЕШНИЙ ВИД (упаковка, цвет, «яркое», «красивое», «в подарок», «как на фото») — ОБЯЗАТЕЛЬНО image(...) в union с ann(...):
  текстовые карточки внешний вид не описывают, его видно только на фото;
- schedule(request=...) — текст запроса покупателя по-русски, не код; после schedule можно сразу показать первый вариант через answer;
- обед/ужин/меню — 3–4 слота (основное блюдо, гарнир или салат, напиток/десерт); калории и белок считаются на весь набор
  (runtime даёт итог по упаковкам), поэтому не фильтруй каждый товар по калориям всего обеда.
Выведи только программу в блоке ```python```."""

FEWSHOT2 = [
    ("[память: —]\nмолоко простоквашино 3,2", 'answer(milk=top(search("молоко простоквашино 3,2", k=20), 3))'),
    ("[память: аллергия на орехи]\nчто-нибудь к чаю детям",
     'p = filt(union(ann("сладкое к чаю для детей"), search("печенье")), allergen_free=["nuts", "peanuts"])\n'
     'answer(sweets=top(rerank(p, "сладкое к чаю для детей"), 4))'),
    ("[память: веган]\nчто-нибудь вкусное в яркой упаковке",
     'p = filt(union(image("яркая красочная упаковка сладостей"), ann("сладости в красивой упаковке")), exclude=["молоко", "яйц", "желатин"])\n'
     'p = verify(p, "веганский продукт: без молока, яиц и мяса")\n'
     'answer(gift=top(rerank(p, "сладости в яркой упаковке"), 4))'),
    ("[память: —]\nпродукты на борщ, без говядины", 'context("ингредиенты для борща")'),
    ("[контекст: свекла, капуста, картофель, морковь, лук, говядина, томатная паста, сметана]\nпродукты на борщ, без говядины",
     'answer(beet=top(search("свекла"), 2), cabbage=top(search("капуста"), 2), potato=top(search("картофель"), 2),\n'
     '       carrot=top(search("морковь"), 2), meat=top(union(search("курица"), search("индейка")), 2),\n'
     '       tomato=top(search("томатная паста"), 2), sour_cream=top(search("сметана"), 2))'),
    ("[прошлая выдача: yogurt, muesli]\nа что из этого дешевле?", 'reply("Дешевле всего из подобранного — см. цены в прошлой выдаче")'),
]

ALLOWED2 = {"search", "ann", "image", "history", "union", "filt", "rank", "rerank", "verify", "top", "context", "remember", "forget",
            "schedule", "watch_price", "answer", "clarify", "reply"}


def validate2(src):
    tree = ast.parse(src)
    if len(tree.body) > 30:
        raise ValueError("слишком длинная программа")
    n = collections.Counter()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.Assign, ast.Expr, ast.Call, ast.Name, ast.Constant, ast.List, ast.Tuple,
                                 ast.Dict, ast.keyword, ast.Load, ast.Store, ast.UnaryOp, ast.USub)):
            raise ValueError(f"недопустимая конструкция: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED2:
                raise ValueError(f"недопустимая функция {getattr(node.func, 'id', '?')}")
            n[node.func.id] += 1
        if isinstance(node, ast.Assign) and not all(isinstance(t, ast.Name) for t in node.targets):
            raise ValueError("присваивать можно только переменным")
    if n["search"] + n["ann"] + n["image"] > 14 or n["verify"] > 4:
        raise ValueError("превышен бюджет вызовов")
    # слоты корзины не должны делить один пул: top(p, 1), top(p, 2)... из одной переменной — признак «всё в одну кучу»
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "answer" and len(node.keywords) > 1:
            src_vars = [ast.dump(kw.value.args[0]) if isinstance(kw.value, ast.Call) and getattr(kw.value.func, "id", "") == "top" and kw.value.args
                        else ast.dump(kw.value) for kw in node.keywords]
            if len(set(src_vars)) < len(src_vars):
                raise ValueError("несколько слотов answer берут товары из одного пула — для каждого продукта нужен свой поиск")
    return tree


class Runtime:
    """Исполняет DSL-программу. Синхронные примитивы собирают план, verify/context исполняются асинхронно после."""

    def __init__(self, tools, sess, client, usage, day=0):
        self.t, self.s, self.cl, self.u, self.day = tools, sess, client, usage, day
        self.log, self.end, self.side, self.pending_verify, self.quotes = [], None, [], [], {}

    async def run(self, src):
        tree = validate2(src)
        env = self.env()
        # verify и context требуют await: исполняем программу построчно, подменяя их результаты
        for stmt in tree.body:
            await self.exec_stmt(stmt, env)
            if self.end and self.end[0] == "context":
                break
        return self.end

    def env(self):
        T, S, L = self.t, self.s, self.log

        def P(x):
            return Pool(x)

        def V(p):   # пул после verify(...) — заглушка до раскрытия; раскрываем в любой операции над пулом
            return self._verified(p) if isinstance(p, tuple) else p

        def search(q, k=50):
            p = P(T.search(str(q), max(1, min(int(k), 200)))); L.append(f"search({q!r}) → {len(p)}"); return p

        def ann(q, k=50):
            p = P(T.ann(str(q), max(1, min(int(k), 200)))); L.append(f"ann({q!r}) → {len(p)}"); return p

        def image(q, k=50):
            p = P(T.image(str(q), max(1, min(int(k), 200)))); L.append(f"image({q!r}) → {len(p)}"); return p

        def history(k=30):
            p = P(T.history(S, int(k))); L.append(f"history → {len(p)}"); return p

        def union(*ps):
            sc = collections.defaultdict(float)
            for p in ps:
                for r, i in enumerate(V(p)):
                    sc[i] += 1 / (60 + r)
            p = P(sorted(sc, key=lambda i: -sc[i])); L.append(f"union → {len(p)}"); return p

        def filt(p, max_price=None, **kw):
            p = V(p)
            kw = {k: v for k, v in kw.items() if k in ("max_kcal", "min_protein", "max_sugar", "max_fat", "exclude", "allergen_free")}
            ok = lambda i: (lambda r: r[0] or r[1] == "нет состава")(item_ok(T.cat.rows[i], **kw))   # без состава пропускаем — проверит verify
            q = P(i for i in p if ok(i) and (max_price is None or T.price(i, self.day) <= max_price))
            L.append(f"filt({', '.join(f'{k}={v}' for k, v in {**kw, 'max_price': max_price}.items() if v is not None)}) {len(p)} → {len(q)}")
            return q

        def rank(p, by="relevance"):
            p = V(p)
            R = T.cat.rows
            nan = lambda v: v is None or v != v
            keys = {"protein": lambda i: -(0 if nan(R[i]["protein"]) else R[i]["protein"]),
                    "-kcal": lambda i: 1e9 if nan(R[i]["kcal"]) else R[i]["kcal"],
                    "-sugar": lambda i: 1e9 if nan(R[i]["sugar"]) else R[i]["sugar"],
                    "price": lambda i: T.price(i, self.day)}
            if by == "personal":
                return P(T.personal_rank(S, list(p)))
            return P(sorted(p, key=keys[by])) if by in keys else P(p)

        def rerank(p, q):
            if isinstance(p, tuple):   # после verify: ранжируем подтверждённые и «нет данных» по отдельности, подтверждённые — первыми
                ok, unk = self._vgroups(p)
                r = P(T.rerank(str(q), ok) + T.rerank(str(q), unk)); L.append(f"rerank({q!r}) {len(r)}"); return r
            p = V(p)
            r = P(T.rerank(str(q), list(p)[:50]) + list(p)[50:]); L.append(f"rerank({q!r}) {len(p)}"); return r

        def verify(p, req):
            ph = ("__verify__", list(p), str(req)); return ph

        def top(p, n=3):
            if isinstance(p, tuple):
                p = self._verified(p)
            # итог проверки состава действует до конца программы: нарушители выкинуты, подтверждённые — первыми
            met = {i for res in self.quotes.values() for i, q in res.items() if q.get("verdict") == "met"}
            bad = {i for res in self.quotes.values() for i, q in res.items() if q.get("verdict") == "violated"}
            p = sorted((i for i in p if i not in bad), key=lambda i: i not in met)
            return P(p[:max(1, min(int(n), 10))])

        def context(q):
            self.end = ("context", str(q)); return ""

        def remember(f):
            if f not in S.memory: S.memory.append(str(f))
            self.side.append(f"запомнил: {f}"); return None

        def forget(f):
            S.memory[:] = [m for m in S.memory if str(f).lower() not in m.lower()]
            self.side.append(f"забыл: {f}"); return None

        def schedule(when, request):
            S.tasks.append({"kind": "schedule", "when": str(when), "what": str(request)})
            self.side.append(f"задача: {when} — {request}"); return None

        def watch_price(p, below):
            for i in list(p)[:3]:
                S.tasks.append({"kind": "price", "item": i, "below": float(below), "what": f"цена {T.cat.rows[i]['name']} ниже {below} ₽"})
            self.side.append(f"слежу за ценой ниже {below} ₽"); return None

        def answer(**slots):
            for k, v in slots.items():
                if not isinstance(v, (Pool, tuple)):
                    raise ValueError(f"answer: слот {k} должен быть пулом товаров")
            self.end = ("answer", slots); return None

        def clarify(q):
            self.end = ("clarify", str(q)); return None

        def reply(t):
            self.end = ("reply", str(t)); return None

        return {k: v for k, v in locals().items() if k in ALLOWED2} | {"__builtins__": {}}

    def _vorder(self, items):
        """Порядок проверки: сначала товары с указанным составом (без состава QA-модели проверять нечего)."""
        return sorted(items, key=lambda i: not self.t.cat.rows[i].get("ingr_lc"))

    def _vgroups(self, ph):
        _, items, req = ph
        items = self._vorder(list(items))
        res = self.quotes.get((req, tuple(items[:12])), {})
        return ([i for i in items[:12] if res.get(i, {}).get("verdict") == "met"],
                [i for i in items[:12] if res.get(i, {}).get("verdict") not in ("met", "violated")])

    def _verified(self, ph):
        _, items, req = ph
        items = self._vorder(list(items))
        res = self.quotes.get((req, tuple(items[:12])), {})
        ok = [i for i in items[:12] if res.get(i, {}).get("verdict") == "met"]
        unk = [i for i in items[:12] if res.get(i, {}).get("verdict") not in ("met", "violated")]
        bad = len(items[:12]) - len(ok) - len(unk)
        self.log.append(f"verify({req!r}) {len(items[:12])} → подтверждено {len(ok)}, нет данных {len(unk)}, отсеяно {bad}")
        return Pool(ok + unk)      # только проверенные: подтверждённые вперёд, «нет данных» после, нарушители отсеяны

    async def exec_stmt(self, stmt, env):
        # заранее выполняем verify(...) внутри выражения: считаем аргументы, вызываем QA-модель, кладём в кэш
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "verify":
                pool = eval(compile(ast.Expression(node.args[0]), "<dsl>", "eval"), env)
                req = eval(compile(ast.Expression(node.args[1]), "<dsl>", "eval"), env)
                pool = self._vorder(list(pool) if not isinstance(pool, tuple) else list(self._verified(pool)))
                key = (str(req), tuple(pool[:12]))
                if key not in self.quotes:
                    self.quotes[key] = await self.t.verify(self.cl, self.u, pool, str(req))
        exec(compile(ast.Module([stmt], []), "<dsl>", "exec"), env)


def summarize2(tools, rt, slots, day=0):
    out = {}
    for k, v in slots.items():
        p = rt._verified(v) if isinstance(v, tuple) else v
        quotes = {i: q for (req, _), res in rt.quotes.items() for i, q in res.items() if q.get("verdict") == "met"}
        checked = {i for (req, _), res in rt.quotes.items() for i in res}
        def extra(i):
            if i in quotes:
                return {"состав": "подтверждено", **({"quote": quotes[i]["quote"][:120]} if quotes[i].get("quote") else {})}
            return {"состав": "не проверен (нет данных)"} if i in checked else None
        out[k] = [card_line(tools.cat, i, day, extra(i)) for i in p] or "НИЧЕГО НЕ НАЙДЕНО"
    tops = [v[0] for v in out.values() if isinstance(v, list) and v]
    total = {"price": sum(c["price"] for c in tops), "kcal_pack": sum(c.get("kcal_pack", 0) for c in tops),
             "protein_pack": round(sum(c.get("protein_pack", 0) for c in tops), 1)}
    return out, total


def user_block(sess, extra=""):
    mem = "; ".join(sess.memory) or "—"
    last = ", ".join(sess.last.keys()) if sess.last else "—"
    dlg = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in sess.dialog[-4:])
    return f"[память: {mem}]\n[история покупок: {'есть' if sess.history else 'нет'}]\n[прошлая выдача: {last}]\n" + \
           (f"[диалог:\n{dlg}]\n" if dlg else "") + extra


ANSWER2 = ("Ты — ассистент продуктового магазина. Коротко ответь по-русски: что подобрал и почему, замены, что не удалось "
           "проверить. Для корзины назови итог по цене (и КБЖУ, если просили). Используй только товары из результатов. "
           "Если у товара есть поле «состав», выбирай сначала «подтверждено»; «не проверен» — только когда подтверждённых нет, и прямо скажи, "
           "что состав не указан. Не утверждай того, чего нет в карточке. Цены синтетические. В последней строке: ВЫБОР: [\"код1\", ...]")


async def run_code2(tools, sess, msg, client, day=0, orch_url=None, orch_model=None, sys_prompt=None, fewshot=True):
    usage, trace = Usage(), []
    ctx = ""
    base = [{"role": "system", "content": sys_prompt or DSL2}]
    if fewshot and not sys_prompt:
        for q, p in FEWSHOT2:
            base += [{"role": "user", "content": q}, {"role": "assistant", "content": f"```python\n{p}\n```"}]
    msgs = base + [{"role": "user", "content": user_block(sess) + msg}]
    end, rt = None, None
    for it in range(MAX_PROGRAMS):
        m = await chat_to(client, msgs, usage, "program", orch_url, orch_model, max_tokens=700)
        txt = m.get("content") or ""
        mm = PROG_RE.search(txt)
        src = (mm.group(1) if mm else txt).strip()
        rt = Runtime(tools, sess, client, usage, day)
        try:
            end = await rt.run(src)
            if not end and rt.side:
                end = ("reply", "; ".join(rt.side))           # программа только из remember/schedule/watch_price
            err = None if end else "программа не завершилась answer/clarify/reply/context"
        except Exception as e:
            end, err = None, f"{type(e).__name__}: {e}"
        trace.append({"iter": it + 1, "program": src, "log": rt.log, "side": rt.side, "error": err, "prompt_user": msgs[len(base)]["content"]})
        if err:
            msgs += [{"role": "assistant", "content": txt}, {"role": "user", "content": f"Ошибка: {err}. Исправь программу."}]
            continue
        if end[0] == "context":
            ctx = await tools.context(client, usage, end[1])
            trace[-1]["context"] = ctx
            msgs = base + [{"role": "user", "content": user_block(sess, f"[контекст: {ctx}]\n") + msg}]
            continue
        if end[0] == "answer":
            summ, total = summarize2(tools, rt, end[1], day)
            empty = [k for k, v in summ.items() if isinstance(v, str)]
            if empty and it < MAX_PROGRAMS - 1:
                msgs += [{"role": "assistant", "content": txt},
                         {"role": "user", "content": f"Пустые позиции: {', '.join(empty)}. Переформулируй запросы или найди замену (явные условия не ослабляй)."}]
                continue
        break
    return await finish(tools, sess, msg, client, usage, trace, end, rt, day, "code")


async def chat_to(client, msgs, usage, kind, url=None, model=None, max_tokens=700):
    if not url:
        return await chat(client, msgs, usage, kind, max_tokens=max_tokens)
    t = time.time()
    r = await client.post(url, json={"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0.2}, timeout=120)
    j = r.json()
    usage.add(kind, j.get("usage", {}), time.time() - t)
    return j["choices"][0]["message"]


async def finish(tools, sess, msg, client, usage, trace, end, rt, day, pipe):
    side = [s for t in trace for s in t.get("side", [])]
    if not end:
        text, choice, summ = "Не получилось составить поиск. Переформулируйте запрос.", [], {}
    elif end[0] == "clarify":
        text, choice, summ = end[1], [], {}
    elif end[0] == "reply":
        m = await chat(client, [{"role": "system", "content": ANSWER2},
                                {"role": "user", "content": f"Запрос: {msg}\nПрошлая выдача: {json.dumps(sess.last, ensure_ascii=False)}\n"
                                                            f"Действия: {side}\nЧерновик ответа: {end[1]}"}], usage, "answer", max_tokens=500)
        text, summ = m.get("content") or "", sess.last
        choice = parse_choice(text)
    else:
        summ, total = summarize2(tools, rt, end[1], day)
        m = await chat(client, [{"role": "system", "content": ANSWER2},
                                {"role": "user", "content": user_block(sess) + f"Запрос: {msg}\nДействия: {side}\nРезультаты (условия проверены runtime, "
                                                            f"quote — цитата из карточки): {json.dumps(summ, ensure_ascii=False)}\nИтог по набору (первые товары позиций): {json.dumps(total)}"}],
                       usage, "answer", max_tokens=700)
        text = m.get("content") or ""
        choice = parse_choice(text)
        sess.last = {k: v for k, v in summ.items() if isinstance(v, list)}
    sess.dialog += [{"role": "user", "content": msg}, {"role": "assistant", "content": re.sub(r"ВЫБОР\s*:.*$", "", text, flags=re.S).strip()}]
    return {"answer": text, "choice": choice, "kind": end[0] if end else "error", "usage": usage.total(), "calls": usage.calls,
            "trace": trace, "side": side, "memory": list(sess.memory), "tasks": list(sess.tasks), "pipeline": pipe}


# ---------------------------------------------------------------- пайплайн tool calling (те же возможности)
def fn(name, desc, props, req=()):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": list(req)}}}


S_, I_ = {"type": "string"}, {"type": "integer"}
TOOLS2 = [
    fn("search", "BM25-поиск по каталогу (точные названия, бренды). Возвращает карточки: название, бренд, вес, цена, КБЖУ, аллергены, состав.",
       {"query": S_, "k": I_}, ["query"]),
    fn("semantic_search", "Семантический поиск (эмбеддинги) для описательных и широких запросов. Карточки как у search.", {"query": S_, "k": I_}, ["query"]),
    fn("image_search", "Поиск по фото товара по текстовому описанию внешнего вида.", {"query": S_, "k": I_}, ["query"]),
    fn("purchase_history", "Товары из истории покупок пользователя (карточки).", {"k": I_}),
    fn("get_product", "Полная карточка товара по коду.", {"code": S_}, ["code"]),
    fn("check_product", "Проверить по карточке, выполнено ли требование (состав, «без свинины», «веганское»). Ответ: verdict и цитата.",
       {"code": S_, "requirement": S_}, ["code", "requirement"]),
    fn("get_context", "Справка: состав рецепта, на что смотреть при выборе.", {"question": S_}, ["question"]),
    fn("remember", "Запомнить факт о пользователе (ограничение, вкус).", {"fact": S_}, ["fact"]),
    fn("forget", "Забыть факт о пользователе.", {"fact": S_}, ["fact"]),
    fn("schedule", "Создать регулярную задачу.", {"when": S_, "request": S_}, ["when", "request"]),
    fn("watch_price", "Уведомить, когда цена товара станет ниже порога.", {"code": S_, "below": {"type": "number"}}, ["code", "below"]),
]
SYS_TOOLS2 = ("Ты — ассистент продуктового магазина с доставкой. Помогаешь найти товары, собрать корзину, учитываешь память "
              "о пользователе и историю покупок, можешь уточнить запрос, создать задачу по расписанию или следить за ценой. "
              "Явные ограничения пользователя и факты из памяти обязательны: проверяй их по карточкам или check_product. "
              "Для набора продуктов ищи каждую позицию; продукт, нарушающий условие по природе, заменяй. Для сценарного запроса "
              "без известного состава используй get_context. Если без уточнения не подобрать — задай вопрос (без вызова инструментов). "
              "Для корзины назови итог по цене. Цены синтетические. В последней строке ответа с товарами: ВЫБОР: [\"код1\", ...]")


def card_full(tools, i, day):
    c = core.full_card(tools.cat.rows[i])
    c["price"] = tools.price(i, day)
    return c


async def run_tools2(tools, sess, msg, client, day=0, max_steps=10):
    usage, trace = Usage(), []
    msgs = [{"role": "system", "content": SYS_TOOLS2 + "\n" + user_block(sess)}] + sess.dialog[-6:] + [{"role": "user", "content": msg}]
    side = []
    for step in range(max_steps):
        m = await chat(client, msgs, usage, "tool-loop", tools=TOOLS2, max_tokens=900)
        calls = m.get("tool_calls") or []
        if not calls:
            text = m.get("content") or ""
            choice = parse_choice(text)
            if not choice and any(t["tool"] in ("search", "semantic_search", "image_search", "purchase_history") for t in trace):
                # модель нашла товары, но забыла строку с кодами: одна дозапрошенная реплика (её токены считаются)
                fix = await chat(client, msgs + [{"role": "assistant", "content": text},
                                                 {"role": "user", "content": "Выведи только строку ВЫБОР: [\"код1\", ...] с кодами товаров, которые ты предложил."}],
                                 usage, "tool-loop", max_tokens=200)
                choice = parse_choice(fix.get("content"))
                text = text + "\nВЫБОР: " + json.dumps(choice) if choice else text
            if choice:
                sess.last = {"выдача": [card_line(tools.cat, tools.cat.by_code[c], day) for c in choice if c in tools.cat.by_code]}
            sess.dialog += [{"role": "user", "content": msg}, {"role": "assistant", "content": re.sub(r"ВЫБОР\s*:.*$", "", text, flags=re.S).strip()}]
            return {"answer": text, "choice": choice, "kind": "answer" if choice else "reply", "usage": usage.total(), "calls": usage.calls,
                    "trace": trace, "side": side, "memory": list(sess.memory), "tasks": list(sess.tasks), "pipeline": "tools"}
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        for tc in calls:
            name = tc["function"]["name"]
            try:
                a = json.loads(tc["function"].get("arguments") or "{}")
            except Exception:
                a = {}
            k = max(1, min(int(a.get("k") or 10), 20))
            try:
                if name in ("search", "semantic_search", "image_search"):
                    f = {"search": tools.search, "semantic_search": tools.ann, "image_search": tools.image}[name]
                    out = [card_full(tools, i, day) for i in f(a.get("query", ""), k)]
                elif name == "purchase_history":
                    out = [card_full(tools, i, day) for i in tools.history(sess, k)]
                elif name == "get_product":
                    i = tools.cat.by_code.get(str(a.get("code")))
                    out = core.full_card(tools.cat.rows[i], 5000) if i is not None else {"error": "нет товара"}
                elif name == "check_product":
                    i = tools.cat.by_code.get(str(a.get("code")))
                    out = (await tools.verify(client, usage, [i], a.get("requirement", ""))).get(i) if i is not None else {"error": "нет товара"}
                elif name == "get_context":
                    out = await tools.context(client, usage, a.get("question", ""))
                elif name == "remember":
                    sess.memory.append(a.get("fact", "")); side.append(f"запомнил: {a.get('fact')}"); out = "ok"
                elif name == "forget":
                    sess.memory[:] = [x for x in sess.memory if a.get("fact", "").lower() not in x.lower()]; side.append(f"забыл: {a.get('fact')}"); out = "ok"
                elif name == "schedule":
                    sess.tasks.append({"kind": "schedule", "when": a.get("when"), "what": a.get("request")}); side.append(f"задача: {a.get('when')} — {a.get('request')}"); out = "ok"
                elif name == "watch_price":
                    i = tools.cat.by_code.get(str(a.get("code")))
                    sess.tasks.append({"kind": "price", "item": i, "below": a.get("below"), "what": f"цена {a.get('code')} ниже {a.get('below')}"}); side.append("слежу за ценой"); out = "ok"
                else:
                    out = {"error": "нет инструмента"}
            except Exception as e:
                out = {"error": str(e)}
            s = json.dumps(out, ensure_ascii=False)
            trace.append({"step": step, "tool": name, "args": a, "chars": len(s)})
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": s})
    msgs.append({"role": "user", "content": "Лимит шагов исчерпан, ответь по найденному."})
    m = await chat(client, msgs, usage, "tool-loop", max_tokens=700)
    text = m.get("content") or ""
    sess.dialog += [{"role": "user", "content": msg}, {"role": "assistant", "content": text}]
    return {"answer": text, "choice": parse_choice(text), "kind": "answer", "usage": usage.total(), "calls": usage.calls, "trace": trace,
            "side": side, "memory": list(sess.memory), "tasks": list(sess.tasks), "pipeline": "tools"}


# ---------------------------------------------------------------- scheduler
def task_message(sess, task, day):
    if task["kind"] == "price":
        return None
    prev = "; ".join(", ".join(c["name"] for c in r) for r in sess.runs[-3:])
    return f"[запуск по расписанию «{task['when']}», день {day}] {task['what']}" + (f". Прошлые меню (не повторяй): {prev}" if prev else "")


def check_prices(tools, sess, day):
    out = []
    for t in sess.tasks:
        if t["kind"] == "price" and t.get("item") is not None:
            p = tools.price(t["item"], day)
            if p <= (t["below"] or 0):
                out.append(f"{tools.cat.rows[t['item']]['name']}: {p} ₽ (порог {t['below']} ₽)")
    return out
