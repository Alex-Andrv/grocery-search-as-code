// Автотур («мультик») для демо: подсветка элементов с подписью. Запуск: кнопка «▶ Тур» или ?tour в адресе.
// Главная: подсказки → живые запросы обоих агентов → переход на /metrics?tour; там — цифры, графики, примеры.
(() => {
  const css = `.tour-spot{position:fixed;z-index:50;border-radius:14px;box-shadow:0 0 0 9999px rgba(10,10,9,.62);transition:all .55s cubic-bezier(.2,.8,.2,1);pointer-events:none}
  .tour-spot.ring{outline:2px solid #4fd1a1;outline-offset:3px}
  .tour-cap{position:fixed;z-index:51;max-width:420px;background:#141413;color:#f0efea;border-radius:12px;padding:12px 16px;font:14.5px/1.45 Inter,system-ui,sans-serif;box-shadow:0 10px 40px rgba(0,0,0,.3);transition:all .45s ease;opacity:0}
  .tour-cap b{color:#4fd1a1;font-weight:600}
  .tour-cap.show{opacity:1}
  .tour-stop{position:fixed;z-index:52;right:18px;bottom:18px;background:#141413;color:#f0efea;border:none;border-radius:999px;padding:8px 14px;font:13px Inter,system-ui;cursor:pointer;opacity:.8}`;
  const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  let spot, cap, stop, alive = false, cur = null;
  const place = (r, pad) => Object.assign(spot.style, {left: r.left - pad + "px", top: r.top - pad + "px", width: r.width + 2 * pad + "px", height: r.height + 2 * pad + "px"});
  addEventListener("scroll", () => { if (spot && cur && cur.el) place(cur.el.getBoundingClientRect(), cur.pad); }, {passive: true});
  addEventListener("resize", () => { if (spot && cur && cur.el) place(cur.el.getBoundingClientRect(), cur.pad); });

  function ensure() {
    if (spot) return;
    spot = Object.assign(document.createElement("div"), {className: "tour-spot ring"});
    cap = Object.assign(document.createElement("div"), {className: "tour-cap"});
    stop = Object.assign(document.createElement("button"), {className: "tour-stop", textContent: "Остановить тур ✕"});
    stop.onclick = end;
    document.body.append(spot, cap, stop);
  }
  function end() { alive = false; for (const e of [spot, cap, stop]) e && e.remove(); spot = cap = stop = null; history.replaceState(null, "", location.pathname); }

  async function show(el, html, ms = 3200, pad = 8) {
    if (!alive) throw "stop";
    ensure();
    if (typeof el === "string") el = document.querySelector(el);
    if (el) {   // прокручиваем и ждём, пока позиция элемента перестанет меняться
      el.scrollIntoView({behavior: "smooth", block: "center"});
      let prev = -1;
      for (let k = 0; k < 40; k++) { await sleep(50); const t = el.getBoundingClientRect().top; if (Math.abs(t - prev) < 0.5 && k > 3) break; prev = t; }
    }
    cur = {el, pad};
    const r = el ? el.getBoundingClientRect() : {left: innerWidth / 2, top: innerHeight / 2, width: 0, height: 0};
    place(r, pad);
    spot.classList.toggle("ring", !!el);
    cap.innerHTML = html; cap.classList.remove("show");
    await sleep(60);
    const cw = Math.min(420, innerWidth - 32), below = r.top + r.height + 20 + 120 < innerHeight;
    Object.assign(cap.style, {width: cw + "px", left: Math.max(16, Math.min(innerWidth - cw - 16, r.left + r.width / 2 - cw / 2)) + "px",
      top: el ? (below ? r.top + r.height + pad + 14 : Math.max(16, r.top - pad - 14 - cap.offsetHeight)) + "px" : (innerHeight / 2 - 40) + "px"});
    cap.classList.add("show");
    await sleep(ms);
  }
  async function type(text) { const q = document.querySelector("#q"); q.value = ""; for (const ch of text) { if (!alive) throw "stop"; q.value += ch; await sleep(38); } }
  const fmt = n => Math.round(n).toLocaleString("ru-RU");
  const last = s => document.querySelector(`#c-${s} .log .turn`);

  async function ask(text, waitCap) {
    await show("#f", `Запрос: <b>${text}</b>`, 400);
    await type(text);
    await sleep(400);
    const p = send();
    await show(".cols", waitCap || "Оба агента получают одну и ту же реплику и работают <b>параллельно</b>", 1500, 4);
    await p;
    await sleep(900);
  }

  async function demoPage() {
    localStorage.removeItem("hints_off"); hints();
    await show(null, "<b>Search as code</b> — агентный поиск товаров, где агент пишет поиск кодом.<br>Слева обычный агент с tool calling, справа — наш.", 3800);
    const H = document.querySelectorAll(".hint");
    for (let i = 0; i < H.length; i++) await show(H[i], `<b>${i + 1}/4</b> · ${H[i].querySelector("b").textContent}`, 3000);

    // 1. корзина под блюдо + память («без свинины» не говорим — агент помнит)
    $("#user").value = "anna"; await newDialog();
    await show("#user", "Покупатель — <b>Анна</b>: не ест свинину, следит за белком. Это в памяти агента.", 3000);
    await ask("всё для карбонары на четверых");
    const B = LAST.base || {}, O = LAST.ours || {}, ub = B.usage || {}, uo = O.usage || {};
    const tb = (ub.in || 0) + (ub.out || 0), to = (uo.in || 0) + (uo.out || 0);
    await show("#t-base", `Обычный агент: <b>${fmt(tb)}</b> токенов — он сам читает каждую карточку товара`, 3200);
    await show("#t-ours", `Search as code: <b>${fmt(to)}</b> токенов — <b>в ${(tb / Math.max(to, 1)).toFixed(1)}× меньше</b>`, 3200);
    const rub = u => ((u.in || 0) * 15 + (u.out || 0) * 60) / 1e6 * 1000;
    await show(".cols", `1000 таких запросов: <b>${rub(uo).toFixed(0)} ₽</b> против ${rub(ub).toFixed(0)} ₽ <span style="opacity:.6">(по тарифу 15/60 ₽ за 1M токенов вход/выход)</span>`, 3600, 4);
    const mo = last("ours") && last("ours").querySelector(".meta"), mb = last("base") && last("base").querySelector(".meta");
    if (mo && uo.calls < ub.calls) await show(mo, `<b>${uo.calls} вызова LLM</b> вместо ${ub.calls}: меньше задержек, меньше точек отказа`, 3400);
    if (O.wall && B.wall && O.wall < B.wall) await show(last("ours").querySelector(".q"), `И быстрее: <b>${O.wall.toFixed(1)} с</b> против ${B.wall.toFixed(1)} с`, 3000);
    const t = last("ours");
    if (t) {
      const sw = t.querySelector(".meta span"); if (sw) sw.click();
      await show(t.querySelector("pre") || t, "Вся логика — <b>одна программа</b>: поиск по слотам блюда, фильтры, ранкер, проверка состава. Runtime исполняет её, модель видит только сводку.", 5000);
      await show(t.querySelector("pre") || t, "Это ещё и <b>прозрачно</b>: видно, как именно агент искал — можно отладить, проверить и воспроизвести", 3600);
      if (sw) sw.click();
      await show(t.querySelector(".prods") || t, "Корзина собрана по слотам блюда — и <b>с учётом памяти</b>: «без свинины» в запросе не было", 3600);
    }

    // 2. уточнение по прошлой выдаче
    await ask("а что из этого дешевле всего?", "Уточнение по прошлой выдаче: наш агент отвечает <b>без нового поиска</b>");
    { const B = LAST.base || {}, O = LAST.ours || {}, ub = B.usage || {}, uo = O.usage || {};
      const tb = (ub.in || 0) + (ub.out || 0), to = (uo.in || 0) + (uo.out || 0);
      await show(last("ours"), `Ответ по прошлой выдаче: <b>${fmt(to)}</b> токенов за ${(O.wall || 0).toFixed(1)} с` + (tb > to ? ` — у обычного агента ${fmt(tb)} за ${(B.wall || 0).toFixed(1)} с` : ""), 3600); }

    // 3. поиск по фото + другой покупатель
    $("#user").value = "vegan"; await newDialog();
    await show("#user", "Другой покупатель — <b>Илья</b>, веган и сладкоежка", 2600);
    await ask("что-нибудь сладкое в яркой красивой упаковке в подарок", "Про упаковку в карточках не написано — нужен <b>поиск по фото</b> (SigLIP)");
    await show(last("ours"), (LAST.ours && (LAST.ours.trace || []).some(x => /image\(/.test(x.program || "")) ? "Агент сам включил <b>поиск по фото</b> — упаковки подобраны по картинке, а <b>QA-постфильтр</b> отсеял товары с молоком по составу" : "Состав каждого товара сверен с памятью «веган» — <b>QA-постфильтр</b> цитирует состав"), 4000);
    await show("#t-ours", `Итого за диалог: <b>${fmt(tot.base)}</b> против <b>${fmt(tot.ours)}</b> токенов`, 3200);
    await show(null, "А теперь — метрики на бенчмарке →", 1800);
    location.href = "/metrics?tour";
  }

  async function metricsPage() {
    await sleep(1400);
    const K = document.querySelectorAll(".kpi");
    const txt = ["<b>В 4× меньше токенов</b> на запрос — значит и в 4× дешевле", "Ответы <b>лучше</b> по оценке LLM-судьи: 3,6 против 3,0", "Наши кандгены находят нужное на <b>+42%</b> лучше обычного ANN"];
    for (let i = 0; i < K.length; i++) await show(K[i], txt[i], 3200);
    await show("#tok", "Разрыв растёт на <b>сложных запросах</b>: корзины, сценарии, широкие запросы", 4000);
    await show("#ret", "BM25 и обычный ANN против наших: <b>многовекторный MaxSim</b>, дообученный двухбашенный и их гибрид", 4000);
    await show("#extra", "Свой <b>дообученный ранкер</b>: +36% nDCG к готовому bge-reranker", 4600);
    await show(document.querySelector(".ex"), "И живые примеры: один запрос — два агента", 3600);
    await show(null, "<b>Search as code</b>: дешевле в разы и точнее.<br>Попробуйте сами на главной.", 4000);
    end();
  }

  window.startTour = async () => {
    alive = true;
    try { location.pathname.startsWith("/metrics") ? await metricsPage() : await demoPage(); } catch (e) { if (e !== "stop") console.error(e); end(); }
  };
  if (new URLSearchParams(location.search).has("tour")) addEventListener("load", () => setTimeout(startTour, 600));
})();
