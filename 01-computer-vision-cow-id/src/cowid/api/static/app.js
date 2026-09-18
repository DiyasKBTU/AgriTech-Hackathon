"use strict";
// COW ID — платформа. Одна страница, без внешних библиотек.
//
// Главное правило интерфейса: говорить словами зоотехника. Время — в часах
// за сутки, отклонение — «ест меньше, −44%», насколько это необычно —
// «заметно» или «сильно» по сравнению с обычными колебаниями этой же коровы.

const METRICS = [
  // ключ, название, что это, единица в таблице
  ["feeder_share", "Ест", "часов в сутки у кормового стола", "ч/сут"],
  ["resting_share", "Лежит", "часов в сутки лёжа и без движения", "ч/сут"],
  ["drinker_share", "Пьёт", "часов в сутки у поилки", "ч/сут"],
  ["activity_rate", "Двигается", "метров пути за час", "м/ч"],
  ["milk_kg", "Удой", "килограммов за сутки", "кг"],
];
const STATUS_LABEL = {
  alert: "проверить", watch: "на заметку", estrus: "признаки охоты", ok: "в норме",
  insufficient: "мало данных", learning: "набирается норма",
};
const EVENT_STATUS = { alert: "alert", warning: "watch", info: "estrus" };
const LEVEL = { strong: "сильно", notable: "заметно" };
const ANSWER = { new: "", confirmed: "подтверждено", false_alarm: "ложная тревога" };
const SOURCE = {
  tag: "номер прочитан с бирки",
  biometric: "узнана по внешности",
  photo: "зарегистрирована по фото",
  sensor: "номер известен: датчик на ошейнике",
};
const WEEKDAY = ["вс", "пн", "вт", "ср", "чт", "пт", "сб"];

const $app = document.getElementById("app");
let liveTimer = null;
let SIGNS = {};   // "метрика:сторона" → «Ест меньше»; приходит с сервера вместе со стадом

// ---------------------------------------------------------------- мелочи

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const enc = encodeURIComponent;
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (e) { /* пусто */ }
    throw new Error(detail);
  }
  return r.json();
}
const isShare = key => key.endsWith("_share");
// Число в единицах таблицы: доля суток — в часах.
function dec(v, digits) { return v.toFixed(digits).replace(".", ","); }
function num(key, v) {
  if (v == null || isNaN(v)) return "—";
  if (isShare(key)) return dec(v * 24, 1);
  if (key === "milk_kg") return dec(v, 1);
  return v.toFixed(0);
}
function perDay(key, v) {
  if (v == null || isNaN(v)) return "—";
  if (isShare(key)) return num(key, v) + " ч";
  return num(key, v) + (key === "milk_kg" ? " кг" : " м/ч");
}
function pct(v) {
  if (v == null || isNaN(v)) return "";
  const r = Math.round(v);
  return (r > 0 ? "+" : r < 0 ? "−" : "") + Math.abs(r) + "%";
}
function ddmm(day) { return day ? day.slice(8, 10) + "." + day.slice(5, 7) : ""; }
function fmtDay(day) {
  if (!day) return "";
  const wd = WEEKDAY[new Date(day + "T12:00:00").getDay()];
  return ddmm(day) + "." + day.slice(0, 4) + ", " + wd;
}
function fmtDur(s) {
  if (s == null) return "—";
  if (s < 120) return Math.round(s) + " с";
  if (s < 7200) return Math.round(s / 60) + " мин";
  return dec(s / 3600, 1) + " ч";
}
function badge(status, label) {
  return '<span class="st st-' + esc(status || "none") + '">' + esc(label || STATUS_LABEL[status] || status || "—") + "</span>";
}
function metricName(key) { const m = METRICS.find(x => x[0] === key); return m ? m[1] : key; }
function signName(metric, delta) {
  const side = delta < 0 ? "down" : "up";
  return SIGNS[metric + ":" + side] || (metricName(metric) + (delta < 0 ? " — меньше" : " — больше"));
}
function rememberSigns(names) { Object.assign(SIGNS, names || {}); }
function lower(s) { return s ? s[0].toLowerCase() + s.slice(1) : s; }

// Где сегодняшнее значение относительно нормы коровы: полоса — её обычные
// сутки (±2 обычных колебания), черта — середина нормы, точка — эти сутки.
function gauge(key, ind) {
  if (!ind || ind.norm == null || !ind.scale) return "";
  const W = 200, H = 24, P = 7;
  const half = Math.max(3 * ind.scale, Math.abs(ind.value - ind.norm) * 1.15);
  const lo = ind.norm - half, hi = ind.norm + half;
  const x = v => (P + (W - 2 * P) * (v - lo) / (hi - lo)).toFixed(1);
  const cls = ind.level ? " lv-" + ind.level : "";
  return '<svg class="gauge" viewBox="0 0 ' + W + " " + H + '" width="' + W + '" height="' + H + '" role="img" aria-label="' +
    esc("эти сутки " + perDay(key, ind.value) + ", норма " + perDay(key, ind.norm)) + '">' +
    '<line class="g-axis" x1="' + P + '" y1="' + H / 2 + '" x2="' + (W - P) + '" y2="' + H / 2 + '"/>' +
    '<rect class="g-band" x="' + x(ind.norm - 2 * ind.scale) + '" y="5" width="' +
    (x(ind.norm + 2 * ind.scale) - x(ind.norm - 2 * ind.scale)).toFixed(1) + '" height="' + (H - 10) + '"/>' +
    '<line class="g-norm" x1="' + x(ind.norm) + '" y1="3" x2="' + x(ind.norm) + '" y2="' + (H - 3) + '"/>' +
    '<circle class="g-val' + cls + '" cx="' + x(ind.value) + '" cy="' + H / 2 + '" r="5.5"/></svg>';
}
function distanceWords(ind) {
  if (!ind || ind.norm == null) return "";
  const side = ind.value < ind.norm ? "ниже" : "выше";
  if (ind.level === "strong") return "сильно " + side + " нормы";
  if (ind.level === "notable") return "заметно " + side + " нормы";
  return "в пределах нормы";
}

// ---------------------------------------------------------------- навигация

function route() {
  if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  stopBrowserCamera();
  const hash = location.hash.replace(/^#/, "") || "/";
  const parts = hash.split("/").filter(Boolean);
  const page = parts[0] || "today";
  document.querySelectorAll("nav a").forEach(a =>
    a.classList.toggle("on", a.dataset.page === (page === "cow" ? "herd" : page)));
  const view = { today: pageToday, herd: pageHerd, cow: pageCow, live: pageLive, quality: pageQuality }[page]
    || pageToday;
  $app.innerHTML = '<div class="loading">Загрузка…</div>';
  view(parts.slice(1)).then(bindDaySwitch).catch(err => {
    $app.innerHTML = '<div class="panel"><div class="body">Ошибка: ' + esc(err.message) + "</div></div>";
  });
}
window.addEventListener("hashchange", route);

async function barInfo() {
  try {
    const s = await api("/api/summary");
    const days = s.days || [];
    document.getElementById("bar-info").innerHTML =
      "коров <b>" + s.animals + "</b> · суток <b>" + days.length + "</b>" +
      (days.length ? " · последние данные <b>" + ddmm(days[days.length - 1]) + "</b>" : "");
  } catch (e) { /* шапка не обязательна */ }
}

// Переключатель суток: ◀ пред. · список · след. ▶
function daySwitch(days, day, prefix) {
  const i = days.indexOf(day);
  const prev = i > 0 ? days[i - 1] : null;
  const next = i >= 0 && i < days.length - 1 ? days[i + 1] : null;
  const opts = days.slice().reverse().map(d =>
    '<option value="' + d + '"' + (d === day ? " selected" : "") + ">" + fmtDay(d) + "</option>").join("");
  const link = (d, text) => d ? '<a class="btn" href="' + prefix + d + '">' + text + "</a>"
    : '<span class="btn off">' + text + "</span>";
  return '<div class="dayswitch">' + link(prev, "◀ " + (prev ? ddmm(prev) : "")) +
    '<select class="txt" data-prefix="' + esc(prefix) + '" aria-label="сутки">' + opts + "</select>" +
    link(next, (next ? ddmm(next) : "") + " ▶") + "</div>";
}
function bindDaySwitch() {
  document.querySelectorAll(".dayswitch select").forEach(s => {
    s.onchange = () => { location.hash = s.dataset.prefix + s.value; };
  });
}

function countsStrip(counts, total, fb) {
  const c = counts || {};
  const cell = (k, v, cls, hint) => '<div' + (hint ? ' title="' + esc(hint) + '"' : "") + '><div class="k">' + k +
    '</div><div class="v ' + (cls || "") + '">' + v + "</div></div>";
  let out = '<div class="strip">' +
    cell("Проверить", c.alert || 0, (c.alert ? "alert" : "zero"), "признаки держатся или их много — осмотреть сегодня") +
    cell("На заметку", c.watch || 0, (c.watch ? "warn" : "zero"), "одно сильное или несколько умеренных отклонений") +
    cell("Признаки охоты", c.estrus || 0, (c.estrus ? "info" : "zero"), "двигается больше, лежит меньше") +
    cell("В норме", c.ok || 0, "");
  if (c.learning) out += cell("Норма набирается", c.learning, "zero", "первые 5 спокойных суток");
  if (c.insufficient) out += cell("Мало данных", c.insufficient, "zero", "корова была видна меньше часа");
  out += cell("Всего коров", total, "");
  if (fb) {
    const checked = (fb.confirmed || 0) + (fb.false_alarm || 0);
    out += cell("Ответы зоотехника", checked + (checked ? " · " + Math.round(100 * (fb.confirmed || 0) / checked) + "% подтв." : ""), "",
      "доля подтверждённых событий — настоящая точность на ферме");
  }
  return out + "</div>";
}

// ---------------------------------------------------------------- события

const URGENCY = { alert: 0, warning: 1, info: 2 };
function byUrgency(a, b) {
  const ra = a.cow_id === "стадо" ? 3 : (URGENCY[a.severity] ?? 1);
  const rb = b.cow_id === "стадо" ? 3 : (URGENCY[b.severity] ?? 1);
  return ra - rb || String(a.cow_id).localeCompare(String(b.cow_id));
}

// Заголовок события словами зоотехника, из тех же чисел, что в базе.
function eventText(e) {
  const d = (e.deviations || [])[0];
  if (e.cow_id === "стадо") {
    return d ? "Всё стадо: " + lower(signName(d.metric, d.delta_pct)) + " — " + pct(d.delta_pct) +
      " у большинства коров" : e.title;
  }
  if (e.severity === "info") return "Признаки охоты: двигается больше, лежит меньше обычного";
  if (!d) return e.title;
  const run = /(\d+)-е сутки подряд/.exec(e.title || "");
  return signName(d.metric, d.delta_pct) + ": " + perDay(d.metric, d.value) +
    (isShare(d.metric) ? " в сутки" : "") + " вместо обычных " + perDay(d.metric, d.baseline) +
    " (" + pct(d.delta_pct) + ")" + (run ? ", " + run[0] : "");
}

function eventItem(e, i) {
  const herd = e.cow_id === "стадо";
  return '<div class="ev click" data-i="' + i + '">' +
    '<div class="ev-top"><span class="cow">' + (herd ? "Всё стадо" : esc(e.cow_id)) + "</span>" +
    badge(herd ? "herd" : EVENT_STATUS[e.severity], herd ? "всё стадо" : null) +
    '<span class="tag right">' + esc(ANSWER[e.status] || "") + "</span></div>" +
    '<div class="ev-text">' + esc(eventText(e)) + "</div></div>";
}

// ---------------------------------------------------------------- Сегодня

async function pageToday(args) {
  const summary = await api("/api/summary");
  const days = summary.days || [];
  if (!days.length) {
    $app.innerHTML = '<div class="panel"><div class="body">' +
      "<h1>Данных пока нет</h1>" +
      '<p class="lead">События появятся после обработки записей: норма коровы строится по нескольким ' +
      "спокойным суткам, поэтому первые дни система только учится.</p>" +
      "<p>Обработать запись камеры за сутки: <code>cowid process запись.mp4 --day 2026-09-16</code></p>" +
      "<p>Загрузить реальные суточные данные MmCows для показа: <code>cowid import-mmcows</code></p>" +
      '<p>Посмотреть, что видит камера прямо сейчас: <a href="#/live">Камера</a>.</p>' +
      "</div></div>";
    return;
  }
  const day = args[0] && days.includes(args[0]) ? args[0] : days[days.length - 1];
  const [events, herd] = await Promise.all([api("/api/events?limit=5000"), api("/api/herd?day=" + day)]);
  rememberSigns(herd.sign_names);
  const list = events.filter(e => e.day === day).sort(byUrgency);
  const cows = list.filter(e => e.cow_id !== "стадо").length;

  let empty = "";
  if (!list.length) {
    empty = (herd.counts.learning === herd.cows.length)
      ? "Первые сутки система запоминает обычный день каждой коровы. События начнутся, когда у коровы наберётся 5 спокойных суток."
      : "Осматривать некого: у всех коров признаки в пределах их собственной нормы.";
  }

  $app.innerHTML =
    '<div class="pagehead"><h1>Сегодня — ' + esc(fmtDay(day)) + "</h1>" + daySwitch(days, day, "#/today/") + "</div>" +
    countsStrip(herd.counts, herd.cows.length, summary.events_by_status) +
    '<div class="cols">' +
    '<div class="panel"><h2>Кого осмотреть · коров ' + cows + "</h2>" +
    (list.length ? '<div id="events">' + list.map(eventItem).join("") + "</div>"
      : '<div class="body calm">' + esc(empty) + "</div>") + "</div>" +
    '<div id="detail">' + (list.length ? "" :
      '<div class="panel"><div class="body note">Ниже — что изменилось у коров за эти сутки, даже если это не повод для осмотра.</div></div>') +
    "</div></div>" +
    boardPanel(herd);

  const items = [...document.querySelectorAll("#events .ev")];
  items.forEach(it => it.onclick = () => {
    items.forEach(x => x.classList.remove("on"));
    it.classList.add("on");
    showEvent(list[+it.dataset.i], it, herd);
  });
  if (items.length) items[0].click();
}

// Доска признаков: что изменилось у кого — и что камера пока не считает.
function boardPanel(herd) {
  const th = herd.thresholds || {};
  const learning = herd.cows.length && herd.counts.learning === herd.cows.length;
  const chips = arr => arr.length ? arr.map(h =>
    '<a class="chip lv-' + h.level + '" href="#/cow/' + enc(h.cow_id) + "/" + herd.day + '" title="' +
    esc(perDay(h.metric, h.value) + " при её норме " + perDay(h.metric, h.norm)) + '">' +
    esc(h.cow_id) + " <b>" + pct(h.delta_pct) + "</b></a>").join("") : '<span class="dash">—</span>';
  const herdRows = (herd.herd_events || []).map(e =>
    '<tr class="herdrow"><td><b>Всё стадо</b></td><td colspan="2">' + esc(eventText(e)) +
    '. Это не болезнь одной коровы — нормы коров на эти сутки поправлены.</td>' +
    '<td class="look">погода и вентиляция, корм, вода, перегруппировка, обработка</td></tr>').join("");
  const rows = herd.board.map(b => {
    const any = b.strong.length + b.notable.length;
    return '<tr class="' + (any ? "hit" : "quiet") + '"><td><b>' + esc(b.name) + "</b></td>" +
      "<td>" + chips(b.strong) + "</td><td>" + chips(b.notable) + "</td>" +
      '<td class="look">' + (any ? esc(b.look) : "") + "</td></tr>";
  }).join("");
  const nm = (herd.not_measured || []).map(i =>
    '<tr class="nm"><td><b>' + esc(i.name) + '</b></td><td colspan="2">' + badge("none", i.status) +
    " камера пока не считает: " + esc(i.what) + '</td><td class="look">' + esc(i.look_at) + "</td></tr>").join("");
  return '<div class="panel"><h2>Какие признаки есть у коров · ' + ddmm(herd.day) + "</h2>" +
    (learning ? '<div class="body calm">Норма ещё набирается — признаков пока нет.</div>' : "") +
    '<table class="t board"><thead><tr><th style="width:17%">признак</th>' +
    '<th style="width:24%">сильно — далеко от её нормы</th><th style="width:24%">заметно</th>' +
    "<th>что осмотреть</th></tr></thead><tbody>" + herdRows + rows + nm + "</tbody></table>" +
    '<div class="body note legend">Каждая корова сравнивается с собой. <b>Заметно</b> — отклонение не меньше ' +
    (th.min_effect_pct || 12) + "% и вдвое больше её обычного колебания; <b>сильно</b> — вчетверо. " +
    "Один признак — ещё не тревога: статус ставится по сумме признаков и по тому, держатся ли они несколько суток. " +
    "Нажмите на корову, чтобы открыть её карточку.</div></div>";
}

function certainty(e, dayInfo, cow, th) {
  const ev = (cow && cow.events.find(x => x.event_id === e.event_id)) || e;
  const evidence = ev.evidence || {};
  const src = evidence.id_source || (cow && cow.id_source);
  let who = src ? SOURCE[src] || src : "номер из базы; способ узнавания не сохранён";
  if (evidence.tag_text) who += ", бирка " + evidence.tag_text + " (" + evidence.tag_votes + " из " + evidence.tag_total + " кадров)";

  const zmax = Math.max(0, ...(e.deviations || []).map(d => Math.abs(d.robust_z || 0)));
  const h = th.cusum_h || 2.5;
  let chance;
  if (dayInfo && dayInfo.episode_days >= 2 && dayInfo.cusum > 0) {
    chance = "признаки держатся " + dayInfo.episode_days + "-е сутки подряд";
  } else {
    chance = "пока одни сутки";
  }
  if (zmax) chance += "; самое сильное отклонение в " + dec(zmax, 1) + " раза больше её обычного колебания";
  if (dayInfo) {
    chance += dayInfo.cusum >= h ? "; накопление дошло до порога «проверить»"
      : "; накоплено " + Math.round(100 * dayInfo.cusum / h) + "% от порога «проверить»";
  }
  const minH = th.min_observed_h || 1;
  const seen = dayInfo ? (dayInfo.observed_h >= minH ? "да: видна " : "нет: видна только ") +
    String(dayInfo.observed_h).replace(".", ",") + " ч за сутки (нужно от " + minH + " ч)" : "—";
  return '<div class="checks">' +
    '<div><div class="q">Это точно она?</div><div class="a">' + esc(who) + "</div></div>" +
    '<div><div class="q">Это не случайность?</div><div class="a">' + esc(chance) + "</div></div>" +
    '<div><div class="q">Данных достаточно?</div><div class="a">' + esc(seen) + "</div></div></div>";
}

function answerButtons(e, okText) {
  const done = e.status !== "new";
  return '<div class="actions">' +
    '<button class="btn primary" data-v="confirmed"' + (done ? " disabled" : "") + ">" + okText + "</button>" +
    '<button class="btn" data-v="false_alarm"' + (done ? " disabled" : "") + ">Ложная тревога</button>" +
    '<span class="tag">' + (done ? "ответ сохранён: " + esc(ANSWER[e.status]) : "ответ нужен, чтобы система знала свою точность") + "</span>";
}
function bindAnswer(box, e, item, redraw) {
  box.querySelectorAll("button[data-v]").forEach(b => b.onclick = async () => {
    await api("/api/events/" + e.event_id + "/feedback", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ verdict: b.dataset.v }),
    });
    e.status = b.dataset.v;
    if (item) item.querySelector(".tag").textContent = ANSWER[e.status];
    redraw();
  });
}

function showHerdEvent(e, item, box) {
  const rows = (e.deviations || []).map(d =>
    "<tr><td><b>" + esc(signName(d.metric, d.delta_pct)) + "</b></td>" +
    '<td class="n">' + perDay(d.metric, d.value) + "</td>" +
    '<td class="n">' + perDay(d.metric, d.baseline) + "</td>" +
    '<td class="n strong">' + pct(d.delta_pct) + "</td></tr>").join("");
  box.innerHTML =
    '<div class="panel"><div class="body">' +
    '<div class="head"><span class="id">Всё стадо</span>' + badge("herd", "всё стадо") +
    '<span class="tag">' + esc(fmtDay(e.day)) + "</span></div>" +
    '<p class="claim">' + esc(eventText(e)) + "</p>" +
    '<p class="lead">Сдвиг сразу у большинства коров. Так бывает из-за погоды, корма, воды, ' +
    "перегруппировки или обработки — это не болезнь одной коровы. Нормы отдельных коров на эти сутки " +
    "поправлены на этот сдвиг, поэтому ложных событий по коровам меньше.</p>" +
    '<table class="t"><thead><tr><th>признак</th><th class="n">середина по стаду</th>' +
    '<th class="n">обычно</th><th class="n">сдвиг</th></tr></thead><tbody>' + rows + "</tbody></table>" +
    '<div class="look-box"><b>Что проверить</b><ul class="look"><li>погоду и вентиляцию в коровнике</li>' +
    "<li>корм и воду: раздача, поилки</li><li>не было ли перегруппировки, обработки, проверки стельности</li></ul></div>" +
    answerButtons(e, "Причину нашли — подтверждаю") +
    '<a class="btn right" href="#/herd/' + e.day + '">Всё стадо за эти сутки</a></div>' +
    "</div></div>";
  bindAnswer(box, e, item, () => showHerdEvent(e, item, box));
}

async function showEvent(e, item, herd) {
  const box = document.getElementById("detail");
  if (e.cow_id === "стадо") { showHerdEvent(e, item, box); return; }
  let cow = null, card = null;
  try { cow = await api("/api/cows/" + enc(e.cow_id)); } catch (err) { /* нет истории */ }
  try { card = await api("/api/registry/" + enc(e.cow_id) + "?day=" + e.day); } catch (err) { /* реестра нет */ }
  const dayInfo = cow ? cow.days.find(d => d.day === e.day) : null;
  const th = (cow && cow.thresholds) || herd.thresholds || {};
  const ev = (cow && cow.events.find(x => x.event_id === e.event_id)) || e;
  const evidence = ev.evidence || {};

  // Все показатели суток, а не только отклонившиеся: видно, что остальное в норме.
  const inds = dayInfo ? dayInfo.indicators : {};
  const devKeys = (e.deviations || []).map(d => d.metric);
  const order = devKeys.concat(METRICS.map(m => m[0]).filter(k => !devKeys.includes(k)));
  const rows = order.map(k => {
    const ind = inds[k] || (e.deviations || []).filter(d => d.metric === k)
      .map(d => ({ value: d.value, norm: d.baseline, delta_pct: d.delta_pct }))[0];
    if (!ind || ind.value == null) return "";
    const lv = ind.level ? " lv-" + ind.level : "";
    return '<tr class="' + (devKeys.includes(k) ? "" : "dim") + '"><td><b>' + metricName(k) + "</b></td>" +
      '<td class="n' + lv + '">' + perDay(k, ind.value) + "</td>" +
      '<td class="n">' + perDay(k, ind.norm) + "</td>" +
      '<td class="n' + lv + '">' + pct(ind.delta_pct) + "</td>" +
      "<td>" + gauge(k, ind) + "</td>" +
      '<td class="words' + lv + '">' + esc(distanceWords(ind)) + "</td></tr>";
  }).join("");

  const lines = String(e.detail || "").split("\n");
  const look = lines.filter(l => l.includes(" — ")).map(l => "<li>" + esc(l.split(" — ").slice(1).join(" — ")) + "</li>").join("");
  const top = devKeys[0];
  const figs = [["animal_crop", "корова в кадре"], ["tag_crop", "крупный план бирки"], ["track_image", "путь за сутки"]]
    .filter(([k]) => evidence[k])
    .map(([k, c]) => '<figure><img src="/evidence/' + esc(evidence[k]) + '" alt=""><figcaption>' + c + "</figcaption></figure>").join("");

  box.innerHTML =
    '<div class="panel"><div class="body">' +
    '<div class="head"><a class="id" href="#/cow/' + enc(e.cow_id) + "/" + e.day + '">' + esc(e.cow_id) + "</a>" +
    badge(EVENT_STATUS[e.severity]) + '<span class="tag">' + esc(fmtDay(e.day)) + "</span></div>" +
    '<p class="claim">' + esc(eventText(e)) + "</p>" +
    registryLine(card) +
    certainty(e, dayInfo, cow, th) +
    '<table class="t dev"><thead><tr><th>признак</th><th class="n">эти сутки</th><th class="n">её обычный день</th>' +
    '<th class="n">разница</th><th>где относительно нормы</th><th></th></tr></thead><tbody>' + rows + "</tbody></table>" +
    '<div class="gauge-legend note">Полоса — её обычные сутки, черта — середина нормы, точка — эти сутки.</div>' +
    (look ? '<div class="look-box"><b>Что осмотреть</b><ul class="look">' + look + "</ul></div>" : "") +
    (cow && top ? '<div class="mini"><div class="mini-title">' + esc(metricName(top)) + " по дням — " +
      esc(METRICS.find(m => m[0] === top)[2]) + "</div>" + chart(cow.days, top, e.day, 150, 1000) + "</div>" : "") +
    (figs ? '<div class="proof">' + figs + "</div>" : "") +
    '<p class="note">Система не ставит диагноз: она показывает, что изменилось у коровы по сравнению с ней самой. Причину определяет ветврач.</p>' +
    answerButtons(e, "Осмотрел — подтверждаю") +
    '<a class="btn right" href="#/cow/' + enc(e.cow_id) + "/" + e.day + '">Карточка коровы</a></div>' +
    "</div></div>";
  bindAnswer(box, e, item, () => showEvent(e, item, herd));
}

// ---------------------------------------------------------------- Стадо

function signsText(row) {
  if (row.status === "learning") return '<span class="muted">система запоминает её обычный день</span>';
  if (row.status === "insufficient") return '<span class="muted">видна меньше часа — сутки не оцениваются</span>';
  if (!row.signs.length) return '<span class="muted">всё в пределах её нормы</span>';
  return row.signs.map(s => '<span class="sg lv-' + s.level + '">' + esc(lower(s.name)) + " <b>" +
    pct(s.delta_pct) + "</b></span>").join("");
}
function metricCell(key, ind) {
  ind = ind || {};
  const lv = ind.level ? " lv-" + ind.level : "";
  const sub = ind.norm != null
    ? '<div class="sub">норма ' + num(key, ind.norm) + " · " + pct(ind.delta_pct) + "</div>" : "";
  return '<td class="m' + lv + '"><div class="val">' + num(key, ind.value) + "</div>" + sub + "</td>";
}

async function pageHerd(args) {
  const want = args[0] ? "?day=" + enc(args[0]) : "";
  const h = await api("/api/herd" + want);
  rememberSigns(h.sign_names);
  if (!h.day) {
    $app.innerHTML = '<h1>Стадо</h1><p class="empty">Суточных данных пока нет.</p>';
    return;
  }
  const head = METRICS.map(m => '<th class="n">' + m[1] + ", " + m[3] + "</th>").join("");
  const rows = h.cows.map(c =>
    '<tr class="click" data-cow="' + esc(c.cow_id) + '">' +
    '<td><span class="cow">' + esc(c.cow_id) + "</span></td>" +
    "<td>" + badge(c.status, c.status_label) + "</td>" +
    '<td class="signs">' + signsText(c) + "</td>" +
    METRICS.map(([k]) => metricCell(k, c.indicators[k])).join("") +
    '<td class="n">' + String(c.observed_h).replace(".", ",") + "</td></tr>").join("");

  const catalog = h.indicators.map(i =>
    "<tr" + (i.status === "работает" ? "" : ' class="nm"') + "><td><b>" + esc(i.name) + "</b></td>" +
    "<td>" + esc(i.what) + "</td><td>" + esc(i.source) + "</td><td>" +
    badge(i.status === "работает" ? "ok" : i.status === "косвенно" ? "watch" : "none", i.status) + "</td>" +
    "<td>" + esc(i.look_at) + "</td></tr>").join("");

  $app.innerHTML =
    '<div class="pagehead"><h1>Стадо — ' + esc(fmtDay(h.day)) + "</h1>" + daySwitch(h.days, h.day, "#/herd/") + "</div>" +
    countsStrip(h.counts, h.cows.length) +
    '<div class="panel"><table class="t herd"><thead><tr><th>корова</th><th>статус</th><th>что изменилось</th>' +
    head + '<th class="n">видна, ч</th></tr></thead><tbody>' + rows + "</tbody></table>" +
    '<div class="body note legend">В каждой ячейке — значение за сутки, ниже — её собственная норма и разница. ' +
    '<span class="sg lv-notable">заметно</span> — вдвое дальше её обычного колебания, ' +
    '<span class="sg lv-strong">сильно</span> — вчетверо. Нажмите на строку, чтобы открыть корову.</div></div>' +
    boardPanel(h) +
    '<div class="panel"><h2>Какие признаки считает система</h2>' +
    '<table class="t"><thead><tr><th>признак</th><th>как считается</th><th>откуда</th><th>состояние</th>' +
    "<th>что осмотреть при отклонении</th></tr></thead><tbody>" + catalog + "</tbody></table></div>";

  document.querySelectorAll("tr[data-cow]").forEach(tr =>
    tr.onclick = () => { location.hash = "#/cow/" + enc(tr.dataset.cow) + "/" + h.day; });
}

// ---------------------------------------------------------------- Корова

function chart(days, key, markDay, height, width) {
  const W = width || 440, H = height || 170, L = 54, R = 10, T = 10, B = 22;
  const k = isShare(key) ? 24 : 1;               // доля суток → часы
  const pts = days.map((d, i) => ({ i, d, m: d.indicators[key] || {} })).filter(p => p.m.value != null);
  if (pts.length < 2) return '<div class="empty">мало данных</div>';
  const vals = [];
  pts.forEach(p => {
    vals.push(p.m.value * k);
    if (p.m.norm != null) vals.push((p.m.norm - 2 * p.m.scale) * k, (p.m.norm + 2 * p.m.scale) * k);
  });
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
  const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
  const x = i => L + (W - L - R) * (days.length === 1 ? 0 : i / (days.length - 1));
  const y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  const withNorm = pts.filter(p => p.m.norm != null);
  let band = "", median = "";
  if (withNorm.length >= 2) {
    const top = withNorm.map(p => x(p.i) + "," + y((p.m.norm + 2 * p.m.scale) * k));
    const bot = withNorm.slice().reverse().map(p => x(p.i) + "," + y((p.m.norm - 2 * p.m.scale) * k));
    band = '<polygon class="band" points="' + top.concat(bot).join(" ") + '"/>';
    median = '<polyline class="median" points="' + withNorm.map(p => x(p.i) + "," + y(p.m.norm * k)).join(" ") + '"/>';
  }
  const mi = days.findIndex(d => d.day === markDay);
  const mark = mi >= 0 ? '<line class="mark" x1="' + x(mi) + '" y1="' + T + '" x2="' + x(mi) + '" y2="' + (H - B) + '"/>' : "";
  const line = '<polyline class="line" points="' + pts.map(p => x(p.i) + "," + y(p.m.value * k)).join(" ") + '"/>';
  const dots = pts.map(p => {
    const flagged = ["alert", "watch", "estrus"].includes(p.d.status);
    return '<circle class="dot ' + esc(p.d.status) + '" cx="' + x(p.i) + '" cy="' + y(p.m.value * k) +
      '" r="' + (flagged ? 5 : 3) + '"><title>' + esc(fmtDay(p.d.day) + ": " + perDay(key, p.m.value) +
      (p.m.norm != null ? ", норма " + perDay(key, p.m.norm) : "") + " — " + STATUS_LABEL[p.d.status]) + "</title></circle>";
  }).join("");
  const unit = isShare(key) ? " ч" : key === "milk_kg" ? " кг" : "";
  const ticks = [lo + pad, (lo + hi) / 2, hi - pad].map(v =>
    '<text x="' + (L - 6) + '" y="' + (y(v) + 4) + '" text-anchor="end">' + dec(v, k > 1 || key === "milk_kg" ? 1 : 0) + unit + "</text>").join("");
  const step = Math.max(1, Math.ceil(days.length / 7));
  const last = days.length - 1;
  const xl = days.map((d, i) => (i % step === 0 || i === last) ?
    '<text x="' + x(i) + '" y="' + (H - 5) + '" text-anchor="' +
    (i === 0 ? "start" : i === last ? "end" : "middle") + '">' + ddmm(d.day) + "</text>" : "").join("");
  return '<svg class="chart" viewBox="0 0 ' + W + " " + H + '">' +
    '<line class="axis" x1="' + L + '" y1="' + (H - B) + '" x2="' + (W - R) + '" y2="' + (H - B) + '"/>' +
    band + median + mark + line + dots + ticks + xl + "</svg>";
}

async function pageCow(args) {
  const id = decodeURIComponent(args[0] || "");
  const c = await api("/api/cows/" + enc(id));
  rememberSigns(c.sign_names);
  if (!c.days.length) {
    $app.innerHTML = '<h1>' + esc(id) + '</h1><p class="empty">Суточных данных нет — только события.</p>';
    return;
  }
  const allDays = c.days.map(d => d.day);
  const cur = c.days.find(d => d.day === args[1]) || c.days[c.days.length - 1];
  const prefix = "#/cow/" + enc(c.cow_id) + "/";
  let card = null;
  try { card = await api("/api/registry/" + enc(c.cow_id) + "?day=" + cur.day); } catch (e) { /* реестра нет */ }

  let said;
  if (cur.status === "learning") said = "Норма ещё набирается: системе нужно 5 спокойных суток этой коровы.";
  else if (cur.status === "insufficient") said = "Корова была видна меньше часа — эти сутки не оцениваются.";
  else if (!cur.signs.length) said = "Все признаки в пределах её собственной нормы.";
  const signs = cur.signs.map(s =>
    '<li><b class="lv-' + s.level + '">' + esc(s.name) + " " + pct(s.delta_pct) + "</b> (" + LEVEL[s.level] + "): " +
    esc(perDay(s.metric, s.value)) + (isShare(s.metric) ? " в сутки" : "") + " вместо обычных " +
    esc(perDay(s.metric, s.norm)) + (s.look ? ' <span class="muted">— ' + esc(s.look) + "</span>" : "") +
    "</li>").join("");
  const normRows = METRICS.map(([k, name]) => {
    const ind = cur.indicators[k] || {};
    if (ind.value == null) return "";
    const lv = ind.level ? " lv-" + ind.level : "";
    return "<tr><td><b>" + name + "</b></td>" +
      '<td class="n">' + perDay(k, ind.norm) + "</td>" +
      '<td class="n' + lv + '">' + perDay(k, ind.value) + "</td>" +
      '<td class="n' + lv + '">' + pct(ind.delta_pct) + "</td>" +
      "<td>" + gauge(k, ind) + '</td><td class="words' + lv + '">' + esc(distanceWords(ind)) + "</td></tr>";
  }).join("");

  const charts = METRICS.map(([k, name, what]) =>
    '<div class="panel"><h2>' + esc(name) + " — " + esc(what) + "</h2>" +
    '<div class="body">' + chart(c.days, k, cur.day) + "</div></div>").join("");
  const dayRows = c.days.slice().reverse().map(d =>
    '<tr class="click' + (d.day === cur.day ? " on" : "") + '" data-day="' + d.day + '">' +
    '<td class="n">' + ddmm(d.day) + "</td><td>" + badge(d.status, d.status_label) + "</td>" +
    '<td class="signs">' + signsText(d) + "</td>" +
    METRICS.map(([k]) => metricCell(k, d.indicators[k])).join("") +
    '<td class="n">' + String(d.observed_h).replace(".", ",") + "</td></tr>").join("");
  const evRows = c.events.map(e =>
    '<tr class="click" data-day="' + e.day + '"><td class="n">' + ddmm(e.day) + "</td><td>" + badge(EVENT_STATUS[e.severity]) + "</td>" +
    "<td>" + esc(eventText(e)) + '</td><td class="tag">' + esc(ANSWER[e.status]) + "</td></tr>").join("");

  $app.innerHTML =
    '<div class="pagehead"><div class="head"><span class="id">' + esc(c.cow_id) + "</span>" + badge(cur.status, cur.status_label) +
    '<span class="tag">' + esc(c.id_source ? SOURCE[c.id_source] || c.id_source : "") + "</span>" +
    '<a class="tag" href="#/herd/' + cur.day + '">← всё стадо</a></div>' + daySwitch(allDays, cur.day, prefix) + "</div>" +
    '<div class="cols cowpage"><div>' +
    '<div class="panel"><h2>Что изменилось · ' + esc(fmtDay(cur.day)) + "</h2><div class=\"body\">" +
    (said ? '<p class="calm">' + esc(said) + "</p>" : '<ul class="signs-list">' + signs + "</ul>") +
    '<p class="note">Видна ' + String(cur.observed_h).replace(".", ",") + " ч за сутки" +
    (cur.episode_days >= 2 ? " · признаки держатся " + cur.episode_days + "-е сутки подряд" : "") + "</p></div></div>" +
    (card ? '<div class="panel"><h2>Из реестра фермы · на ' + esc(fmtDay(card.as_of)) + '</h2><div class="body">' +
      registryHtml(card) + "</div></div>" : "") + "</div>" +
    '<div class="panel"><h2>Её норма и эти сутки</h2><table class="t dev"><thead><tr><th>признак</th>' +
    '<th class="n">её обычный день</th><th class="n">эти сутки</th><th class="n">разница</th>' +
    "<th>где относительно нормы</th><th></th></tr></thead><tbody>" + normRows + "</tbody></table>" +
    '<div class="body note">Норма — по её последним спокойным суткам (до 14). Полоса — её обычные сутки, черта — середина, точка — эти сутки.</div></div>' +
    "</div>" +
    '<p class="lead">Графики: линия — её значение по суткам, зелёная полоса — её собственная норма, пунктир — середина нормы, ' +
    "вертикальная черта — выбранные сутки. Крупные точки — отмеченные сутки (наведите, чтобы увидеть статус).</p>" +
    '<div class="charts">' + charts + "</div>" +
    '<div class="panel"><h2>События</h2>' + (evRows ? '<table class="t"><tbody>' + evRows + "</tbody></table>" :
      '<div class="body empty">Событий нет</div>') + "</div>" +
    '<div class="panel"><h2>Сутки — нажмите, чтобы выбрать</h2><table class="t herd"><thead><tr><th class="n">сутки</th><th>статус</th>' +
    "<th>что изменилось</th>" + METRICS.map(m => '<th class="n">' + m[1] + ", " + m[3] + "</th>").join("") +
    '<th class="n">видна, ч</th></tr></thead><tbody>' + dayRows + "</tbody></table></div>";

  document.querySelectorAll("tr[data-day]").forEach(tr =>
    tr.onclick = () => { location.hash = prefix + tr.dataset.day; });
}

// ---------------------------------------------------------------- Камера

async function pageLive() {
  const [sources, state, phone] = await Promise.all([api("/api/live/sources"), api("/api/live/state"),
    api("/api/live/phone").catch(() => ({}))]);
  tracksMode = "";
  selectedCow = null;
  const running = state.source && sources.find(s => s.id === state.source.id || (s.kind === state.source.kind && s.kind !== "playlist"));
  let chosen = (running && running.id) || (sources[0] && sources[0].id);
  const kindOf = id => (sources.find(s => s.id === id) || {}).kind;
  const opts = sources.map(s =>
    '<label class="opt' + (s.id === chosen ? " on" : "") + '" data-id="' + esc(s.id) + '">' +
    '<input type="radio" name="src" value="' + esc(s.id) + '"' + (s.id === chosen ? " checked" : "") + ">" +
    '<div class="ttl">' + esc(s.title) + '</div><div class="sub">' + esc(s.note) + "</div></label>").join("");

  $app.innerHTML =
    '<div class="cols live">' +
    '<div><div class="panel"><h2>Источник</h2>' + opts +
    '<div class="body"><div id="src-extra"></div>' +
    '<div class="actions"><button class="btn primary" id="start">Включить</button>' +
    '<button class="btn danger" id="stop">Выключить</button></div>' +
    '<p class="note" style="margin-top:8px">Живой режим — для установки камеры и показа. ' +
    "Суточные признаки считаются ночной обработкой записи.</p></div></div></div>" +
    '<div><div class="panel"><h2 id="src-title">Кадр</h2><div class="stream-wrap"><div class="stream" id="stream">' +
    '<div class="off">Камера выключена</div></div>' +
    '<video id="cam-preview" class="cam-preview" playsinline muted hidden></video></div>' +
    '<div class="body note" id="clip"></div></div></div>' +
    '<div><div class="panel"><h2>Сейчас</h2><div class="body">' +
    '<div class="bigrow"><div><div class="bignum" id="n-in">0</div><div class="k">коров в кадре</div></div>' +
    '<div id="known-box"><div class="bignum" id="n-known">0</div><div class="k">узнано</div></div>' +
    '<div id="pose-box" hidden><div class="bignum lying" id="n-lie">0</div><div class="k">лежат</div></div></div>' +
    '<div class="kv" id="id-box"><span class="k">«не знаю»</span><span class="v" id="n-unknown">0</span>' +
    '<span class="k">коров в реестре камеры</span><span class="v" id="n-gallery">—</span>' +
    '<span class="k">время на ферме</span><span class="v" id="n-clock">—</span></div>' +
    '<p class="note" id="no-gallery" hidden>Коровы этого источника не зарегистрированы — номера не ставятся, ' +
    "считаются коровы в кадре, движение и время.</p></div></div>" +
    '<div class="panel"><h2>Камера</h2><div class="body">' +
    '<div class="health" id="c-problems">—</div><div class="kv">' +
    '<span class="k">разрешение</span><span class="v" id="c-res">—</span>' +
    '<span class="k">кадров/с источника</span><span class="v" id="c-fps">—</span>' +
    '<span class="k">обработка, кадров/с</span><span class="v" id="c-proc">—</span>' +
    '<span class="k">яркость (0–255)</span><span class="v" id="c-bright">—</span>' +
    '<span class="k">резкость</span><span class="v" id="c-sharp">—</span></div></div></div>' +
    "</div></div>" +
    '<div class="panel" id="cows-box" hidden><h2 id="cows-title">Узнанные коровы — что делали и что это значит</h2>' +
    '<table class="t cows"><thead><tr><th>корова</th><th>из реестра фермы</th><th>сейчас</th>' +
    '<th class="n">на виду</th><th class="n">лежит</th><th class="n">лежит подряд</th>' +
    '<th class="n">у корма</th><th class="n">у поилки</th><th>что это значит</th></tr></thead>' +
    '<tbody id="cows"></tbody></table>' +
    '<div class="body note legend" id="cows-note"></div></div>' +
    '<div id="cow-card"></div>' +
    '<details class="panel tracks-box"><summary><h2 id="tracks-title">Дорожки в кадре (для проверки)</h2></summary>' +
    '<table class="t"><thead id="tracks-head"></thead><tbody id="tracks"></tbody></table>' +
    '<div class="body note" id="tracks-more"></div>' +
    '<div class="body note">Номер дорожки ставится голосованием её кадров: хотя бы 3 голоса и больше половины за одну корову. ' +
    "До этого — «?». Серая рамка — честное «не знаю», синяя — корова лежит.</div></details>";

  // Поля под выбранным источником: номер камеры, ссылка, как стоит камера, адрес для телефона.
  function renderExtra() {
    const kind = kindOf(chosen);
    const view = '<div class="tag">Как стоит камера</div><select class="txt" id="view">' +
      '<option value="side"' + (kind === "browser" ? " selected" : "") + ">сбоку или под углом — готовая модель, без номеров</option>" +
      '<option value="top"' + (kind !== "browser" ? " selected" : "") + ">сверху над проходом — наш детектор и галерея</option></select>";
    let html = "";
    if (kind === "camera") {
      html = '<div class="tag">Номер камеры: 0 — встроенная, 1 — USB</div>' +
        '<input class="txt mono" id="camidx" value="0">' + view;
    } else if (kind === "url") {
      html = '<div class="tag">Ссылка на поток</div>' +
        '<input class="txt mono" id="url" placeholder="rtsp://… или http://адрес-телефона:8080/video">' + view;
    } else if (kind === "browser") {
      html = view + phoneHint(phone);
    }
    const box = document.getElementById("src-extra");
    box.innerHTML = html;
    box.hidden = !html;
  }
  renderExtra();

  document.querySelectorAll("label.opt").forEach(l => l.onclick = () => {
    document.querySelectorAll("label.opt").forEach(x => x.classList.remove("on"));
    l.classList.add("on");
    if (chosen !== l.dataset.id) { chosen = l.dataset.id; renderExtra(); }
  });
  document.getElementById("start").onclick = async () => {
    const kind = kindOf(chosen);
    const view = document.getElementById("view");
    const body = view ? { view: view.value } : {};
    if (kind === "url") {
      body.url = document.getElementById("url").value.trim();
      if (!body.url) { alert("Вставьте ссылку на поток камеры."); return; }
    } else if (kind === "camera") {
      body.camera = parseInt(document.getElementById("camidx").value, 10) || 0;
    } else {
      body.source = chosen;
    }
    stopBrowserCamera();
    try {
      await api("/api/live/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      selectedCow = null;
      if (!document.getElementById("cow-card")) return;     // пока включалось, ушли со страницы
      document.getElementById("cow-card").innerHTML = "";
      attachStream();
      if (kind === "browser") await startBrowserCamera();
    } catch (err) {
      if (kind === "browser") await api("/api/live/stop", { method: "POST" }).catch(() => {});
      alert("Не удалось включить: " + err.message);
    }
  };
  document.getElementById("stop").onclick = async () => {
    stopBrowserCamera();
    await api("/api/live/stop", { method: "POST" });
    document.getElementById("stream").innerHTML = '<div class="off">Камера выключена</div>';
  };
  document.getElementById("cows").onclick = ev => {
    const tr = ev.target.closest("tr[data-cow]");
    if (!tr) return;
    selectedCow = tr.dataset.cow;
    renderCowCard();
  };
  if (state.running) attachStream();
  liveTimer = setInterval(updateLive, 1000);
  updateLive();
}

// Как открыть камеру телефона: браузер даёт камеру только по https или на самом компьютере.
function phoneHint(phone) {
  const local = ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
  const https = phone && phone.enabled && phone.https_port && (phone.addresses || []).length;
  let h = "";
  if (!window.isSecureContext) {
    h += '<p class="warn-text">Страница открыта по http — браузер телефона не даст камеру. ' +
      (https ? "Откройте https-адрес ниже." : "Запустите сервер командой <code>cowid serve --phone</code>.") + "</p>";
  }
  if (https) {
    h += '<div class="tag">На телефоне (та же сеть Wi-Fi) откройте:</div>' +
      phone.addresses.map(ip => '<div class="phone-url mono">https://' + esc(ip) + ":" + phone.https_port + "/#/live</div>").join("") +
      '<p class="note">Браузер предупредит о сертификате: «Дополнительно» → «Перейти на сайт». ' +
      "Затем этот же пункт → «Включить» → разрешить камеру. Кадр с рамками виден и здесь, и на телефоне.</p>";
  } else if (local) {
    h += '<p class="note">На этом компьютере камера работает сразу. Для телефона запустите сервер командой ' +
      "<code>cowid serve --phone</code> — здесь появится адрес для телефона.</p>";
  } else if (phone && phone.enabled) {
    h += '<p class="note">https не поднят (нет openssl). Телефон можно подключить приложением IP Webcam — ' +
      "пункт «IP-камера или телефон по ссылке».</p>";
  }
  return h;
}

// ---------------------------------------------------------------- камера браузера

let camStream = null;
let camToken = 0;

async function startBrowserCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(window.isSecureContext ? "этот браузер не даёт доступ к камере"
      : "браузер даёт камеру только по https или на самом компьютере — запустите cowid serve --phone");
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } } });
  } catch (e) {
    throw new Error(e.name === "NotAllowedError" ? "доступ к камере запрещён — разрешите его в настройках сайта"
      : e.name === "NotFoundError" ? "камера не найдена"
      : e.name === "NotReadableError" ? "камера занята другой программой" : e.message);
  }
  camStream = stream;
  const token = ++camToken;
  const video = document.getElementById("cam-preview");
  if (!video) { stopBrowserCamera(); return; }
  video.srcObject = stream;
  video.hidden = false;
  await video.play().catch(() => {});
  pushFrames(video, token);
}

function stopBrowserCamera() {
  camToken++;
  if (camStream) camStream.getTracks().forEach(t => t.stop());
  camStream = null;
  const video = document.getElementById("cam-preview");
  if (video) { video.srcObject = null; video.hidden = true; }
}

// Кадр за кадром: следующий уходит, когда сервер принял предыдущий (не чаще 8 в секунду).
async function pushFrames(video, token) {
  const canvas = document.createElement("canvas");
  while (token === camToken) {
    const started = performance.now();
    if (video.videoWidth) {
      const scale = Math.min(1, 960 / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(done => canvas.toBlob(done, "image/jpeg", 0.8));
      if (blob && token === camToken) {
        try {
          const r = await fetch("/api/live/frame", { method: "POST", headers: { "Content-Type": "image/jpeg" }, body: blob });
          if (r.status === 409) { stopBrowserCamera(); return; }   // включили другой источник
        } catch (e) { /* сеть моргнула — отправим следующий кадр */ }
      }
    }
    await new Promise(done => setTimeout(done, Math.max(20, 125 - (performance.now() - started))));
  }
}

// Кадр идёт потоком mjpg. Если за 6 с не пришло ни одного кадра (антивирус или прокси
// держат поток целиком, медленный первый кадр), берём последний кадр снимками.
const STREAM_WAIT_MS = 6000;
const SNAPSHOT_MS = 300;
let snapshotToken = 0;

function attachStream() {
  const box = document.getElementById("stream");
  if (!box) return;
  const token = ++snapshotToken;
  const img = new Image();
  img.alt = "кадр с камеры с рамками коров";
  img.src = "/api/live/stream.mjpg?t=" + Date.now();
  box.replaceChildren(img);
  setTimeout(() => {
    if (token === snapshotToken && img.isConnected && !img.naturalWidth) pollSnapshots(img, token);
  }, STREAM_WAIT_MS);
}

function pollSnapshots(img, token) {
  const next = new Image();
  next.alt = img.alt;
  const again = (cur, ms) => setTimeout(() => {
    if (token === snapshotToken && cur.isConnected) pollSnapshots(cur, token);
  }, ms);
  next.onload = () => {
    if (token !== snapshotToken || !img.isConnected) return;
    img.replaceWith(next);
    img.removeAttribute("src");                     // обрывает поток mjpg, если он ещё открыт
    again(next, SNAPSHOT_MS);
  };
  next.onerror = () => again(img, 1000);            // кадра ещё нет — сервер ответил 204
  next.src = "/api/live/latest.jpg?t=" + Date.now();
}

const TRACK_ROWS = 15;
let tracksMode = "";
let selectedCow = null;
let lastCows = [];
let lastFarm = null;

// Доля времени и обычная доля: «41% · обычно 18%».
function shareCell(pct, norm, level, own) {
  if (pct == null) return '<td class="n muted">—</td>';
  const cls = level ? " lv-" + level : "";
  return '<td class="n' + cls + '"><div class="val">' + pct + "%</div>" +
    (norm != null ? '<div class="sub">' + (own ? "обычно " : "у стада ") + norm + "%</div>" : "") + "</td>";
}
function signLevel(row, word) {
  const s = (row.signs || []).find(x => x.text.includes(word));
  return s ? s.level : "";
}
function signsCell(row) {
  if ((row.signs || []).length) {
    return row.signs.map(s => '<div class="sign lv-' + s.level + '">' + esc(s.text) + "</div>").join("");
  }
  return row.seen_min < 20 ? '<span class="muted">мало наблюдений (нужно от 20 мин)</span>'
    : '<span class="calm-text">как обычно в эти часы</span>';
}

function updateCows(s) {
  const box = document.getElementById("cows-box");
  const cows = s.cows || [];
  lastCows = cows;
  lastFarm = s.farm || null;
  box.hidden = !cows.length;
  if (!cows.length) return;
  document.getElementById("cows").innerHTML = cows.map(r => {
    const reg = r.registry;
    const leg = reg && reg.last_leg_problem;
    return '<tr class="click' + (r.cow === selectedCow ? " on" : "") + '" data-cow="' + esc(r.cow) + '">' +
      '<td><span class="cow">' + esc(r.cow) + "</span></td>" +
      "<td>" + (reg ? esc(reg.summary || "—") + (leg ? '<div class="sub warn-text">' + esc(leg.what) + " " +
        esc(ddmm(leg.day)) + "</div>" : "") : '<span class="muted">нет в реестре</span>') + "</td>" +
      "<td>" + esc(r.state || "—") + (r.zone ? '<div class="sub">' + esc(r.zone) + "</div>" : "") + "</td>" +
      '<td class="n">' + fmtDur(60 * r.seen_min) + "</td>" +
      shareCell(r.lying_pct, r.lying_norm_pct, signLevel(r, "лежит больше") || signLevel(r, "не ложится"), r.norm_own) +
      '<td class="n' + (r.bout_min >= 120 ? " lv-notable" : "") + '">' + (r.bout_min ? fmtDur(60 * r.bout_min) : "—") + "</td>" +
      shareCell(r.feeder_pct, r.feeder_norm_pct, signLevel(r, "у кормового"), r.norm_own) +
      shareCell(r.drinker_pct, r.drinker_norm_pct, signLevel(r, "у поилки"), r.norm_own) +
      '<td class="signs">' + signsCell(r) + "</td></tr>";
  }).join("");
  const farm = s.farm || {};
  document.getElementById("cows-note").innerHTML =
    "Время — сколько корова была на виду за сеанс (часы фермы). «Обычно» — " +
    (farm.norms ? "её же обычные часы по датчикам фермы (13 суток); у коров без датчиков — стадо в эти же часы. "
      : "стадо в эти же часы. ") +
    (farm.registry ? "Карточка — реальные записи фермы на " + esc(fmtDay(farm.as_of)) + ": лактация, отёл, стельность, болезни. " : "") +
    "Хромоту камера здесь не видит напрямую — это косвенный признак по поведению. Нажмите на корову, чтобы открыть карточку.";
  if (selectedCow) renderCowCard(true);
}

async function renderCowCard(refreshOnly) {
  const box = document.getElementById("cow-card");
  const row = lastCows.find(r => r.cow === selectedCow);
  if (!row || !box) return;
  document.querySelectorAll("#cows tr").forEach(tr => tr.classList.toggle("on", tr.dataset.cow === selectedCow));
  let card = refreshOnly && box.dataset.cow === selectedCow ? JSON.parse(box.dataset.card || "null") : null;
  if (!refreshOnly || !card) {
    try {
      card = await api("/api/registry/" + enc(selectedCow) + (lastFarm && lastFarm.as_of ? "?day=" + lastFarm.as_of : ""));
    } catch (e) { card = null; }
  }
  box.dataset.cow = selectedCow;
  box.dataset.card = JSON.stringify(card);
  box.innerHTML = cowCardHtml(row, card);
}

// Одна строка «что известно о корове» над событием.
function registryLine(card) {
  if (!card) return "";
  const recent = (card.health || []).slice(0, 3).map(h => h.what + " " + ddmm(h.day));
  return '<p class="reg-line"><b>Из реестра:</b> ' + esc(card.summary || "") +
    (card.milk_kg != null ? " · удой " + dec(card.milk_kg, 1) + " кг" : "") +
    (recent.length ? ' · <span class="warn-text">' + esc(recent.join(", ")) + "</span>" : "") + "</p>";
}

function fmtDate(day) { return day ? ddmm(day) + "." + day.slice(2, 4) : ""; }

function registryHtml(card) {
  if (!card) return '<p class="muted">В реестре фермы этой коровы нет.</p>';
  const kv = [
    ["электронная бирка", card.eid], ["загон", card.pen],
    ["лактация", card.lactation], ["отёл", card.calved ? fmtDay(card.calved) : null],
    ["дней после отёла", card.days_in_milk], ["стельность", card.pregnancy],
    ["удой, кг/сут (7 дней)", card.milk_kg != null ? dec(card.milk_kg, 1) : null],
  ].filter(x => x[1] != null && x[1] !== "");
  const events = (list, empty) => list && list.length ? '<table class="t"><tbody>' + list.map(e =>
    '<tr><td class="n nowrap">' + esc(fmtDate(e.day)) + "</td><td>" + esc(e.what) +
    (e.detail ? ' <span class="muted">— ' + esc(e.detail) + "</span>" : "") + "</td></tr>").join("") +
    "</tbody></table>" : '<p class="muted">' + empty + "</p>";
  return '<div class="kv reg">' + kv.map(([k, v]) => '<span class="k">' + k + '</span><span class="v">' + esc(v) + "</span>").join("") + "</div>" +
    '<h3 class="sub-h">Здоровье за 120 дней</h3>' + events(card.health, "записей нет") +
    '<h3 class="sub-h">Воспроизводство</h3>' + events((card.reproduction || []).slice(0, 5), "записей нет");
}

function cowCardHtml(row, card) {
  const signs = (row.signs || []).length ? '<ul class="signs-list">' + row.signs.map(s =>
    '<li class="lv-' + s.level + '">' + esc(s.text) + "</li>").join("") + "</ul>"
    : '<p class="calm">' + (row.seen_min < 20 ? "Мало наблюдений — нужно хотя бы 20 минут." : "Всё как обычно в эти часы.") + "</p>";
  const line = (label, pct, norm) => pct == null ? "" :
    "<tr><td>" + label + '</td><td class="n">' + pct + '%</td><td class="n">' + (norm != null ? norm + "%" : "—") + "</td></tr>";
  return '<div class="cols cowpage">' +
    '<div class="panel"><h2>' + esc(row.cow) + " — что видит камера</h2><div class=\"body\">" +
    '<div class="head"><span class="id">' + esc(row.cow) + "</span><span class=\"tag\">" + esc(row.state || "") +
    (row.zone ? " · " + esc(row.zone) : "") + "</span></div>" + signs +
    '<table class="t"><thead><tr><th>за сеанс</th><th class="n">доля времени</th><th class="n">обычно</th></tr></thead><tbody>' +
    line("лежит", row.lying_pct, row.lying_norm_pct) + line("у кормового стола", row.feeder_pct, row.feeder_norm_pct) +
    line("у поилки", row.drinker_pct, row.drinker_norm_pct) + "</tbody></table>" +
    '<p class="note">На виду ' + fmtDur(60 * row.seen_min) + (row.bout_min ? " · лежит без перерыва " + fmtDur(60 * row.bout_min) : "") +
    (row.longest_bout_min ? " · самый долгий период лёжа " + fmtDur(60 * row.longest_bout_min) : "") +
    ". «Обычно» — " + esc(row.norm_source) + ".</p>" +
    (row.cow.match(/^C\d\d$/) ? '<a class="btn" href="#/cow/' + enc(row.cow) + "/" + (card ? card.as_of : "") + '">Сутки по датчикам</a>' : "") +
    "</div></div>" +
    '<div class="panel"><h2>Из реестра фермы' + (card ? " · на " + esc(fmtDay(card.as_of)) : "") + "</h2><div class=\"body\">" +
    registryHtml(card) + "</div></div></div>";
}

async function updateLive() {
  let s;
  try { s = await api("/api/live/state"); } catch (e) { return; }
  const $ = id => document.getElementById(id);
  const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
  if (!$("tracks")) return;
  set("src-title", s.running ? "Кадр — " + (s.source ? s.source.title : "") : "Кадр");
  const fromBrowser = s.source && s.source.kind === "browser";
  set("clip", !s.running ? "" : s.error ? (fromBrowser && !s.processed ? s.error : "Ошибка: " + s.error)
    : fromBrowser ? (camStream ? "кадры с камеры этого устройства" : "кадры присылает другое устройство")
    : s.clip ? "ролик " + s.clip : "");
  set("n-in", s.in_frame || 0);
  set("n-clock", s.clock || "—");

  const poses = s.lying != null;
  $("pose-box").hidden = !poses;
  if (poses) set("n-lie", s.lying);
  const registered = !!s.gallery;
  $("known-box").hidden = s.running && !registered;
  $("id-box").hidden = s.running && !registered;
  $("no-gallery").hidden = !(s.running && !registered);
  set("n-known", s.known || 0);
  set("n-unknown", s.unknown || 0);
  set("n-gallery", s.gallery == null ? "—" : s.gallery);

  set("c-res", s.frame ? s.frame[0] + "×" + s.frame[1] : "—");
  set("c-fps", s.fps_in || "—");
  set("c-proc", s.fps_proc || "—");
  set("c-bright", s.health ? s.health.brightness : "—");
  set("c-sharp", s.health ? s.health.sharpness : "—");
  const prob = $("c-problems");
  const bad = s.health && s.health.problems.length;
  prob.className = "health " + (bad ? "bad" : s.running && s.health ? "good" : "");
  prob.textContent = !s.running ? "камера выключена" : bad ? "Внимание: " + s.health.problems.join(", ")
    : s.health ? "картинка в порядке" : "ждём кадр…";

  updateCows(s);

  // Таблица дорожек зависит от того, узнаёт ли источник номера и различает ли позу.
  const mode = (registered ? "id" : "noid") + (poses ? "-pose" : "");
  if (mode !== tracksMode) {
    tracksMode = mode;
    $("tracks-head").innerHTML = "<tr><th class=\"n\">дорожка</th>" +
      (registered ? '<th>корова</th><th class="n">голоса кадров</th>' : "") +
      '<th class="n">в кадре</th><th>сейчас</th>' +
      (poses ? '<th class="n">стоит</th><th class="n">лежит</th>'
        : '<th class="n">идёт</th><th class="n">стоит</th>') + "<th>зона</th></tr>";
  }
  let tracks = (s.tracks || []).slice().sort((a, b) => b.seconds - a.seconds);
  const more = Math.max(0, tracks.length - TRACK_ROWS);
  tracks = tracks.slice(0, TRACK_ROWS);
  const cols = 3 + (registered ? 2 : 0) + 2 + 1;
  $("tracks").innerHTML = tracks.map(t =>
    '<tr><td class="n">' + t.track + "</td>" +
    (registered ? "<td>" + (t.cow ? '<span class="cow">' + esc(t.cow) + "</span>" : '<span class="tag">?</span>') +
      '</td><td class="n">' + t.votes + " из " + t.total + "</td>" : "") +
    '<td class="n">' + fmtDur(t.seconds) + "</td><td>" + esc(t.state) + "</td>" +
    (poses ? '<td class="n">' + fmtDur(t.still_s) + '</td><td class="n">' + fmtDur(t.lying_s) + "</td>"
      : '<td class="n">' + fmtDur(t.moving_s) + '</td><td class="n">' + fmtDur(t.still_s) + "</td>") +
    "<td>" + esc(t.zone || "—") + "</td></tr>").join("") ||
    '<tr><td colspan="' + cols + '" class="muted">' + (s.running ? "коров в кадре нет" : "камера выключена") + "</td></tr>";
  set("tracks-more", more ? "и ещё дорожек: " + more : "");
}

// ---------------------------------------------------------------- Качество

function qrow(cells, cls) {
  return "<tr" + (cls ? ' class="' + cls + '"' : "") + ">" +
    cells.map((c, i) => "<td" + (i ? ' class="n"' : "") + ">" + c + "</td>").join("") + "</tr>";
}
function p1(v) { return v == null ? "—" : dec(100 * v, 1) + "%"; }
function p0(v) { return v == null ? "—" : Math.round(100 * v) + "%"; }
function verdict(text, good) { return '<p class="verdict ' + (good ? "good" : "bad") + '">' + esc(text) + "</p>"; }

async function pageQuality() {
  const q = await api("/api/quality");
  const parts = [];
  parts.push("<h1>Как проверено качество</h1>" +
    '<p class="lead">Только реальные данные. Проверка всегда на коровах или днях, которых модель не видела. ' +
    "Рядом — контрольные проверки: если необученная сеть почти так же хороша, цифре верить нельзя.</p>");

  const det = q.detector;
  if (det) {
    const t = det.trained, c = det.coco_baseline;
    parts.push('<div class="panel"><h2>Найти корову на кадре (детектор)</h2><div class="body">' +
      verdict("Находит " + p1(t.oriented.recall) + " коров; готовая модель — " + p1(c.iou50.recall) + ".", true) +
      "<p>Cows2021: " + t.oriented.images + " кадров 4–11 марта, коров " + t.oriented.cows + ". Детектор учился на кадрах 5–29 февраля.</p>" +
      '<table class="t"><thead><tr><th>модель</th><th class="n">найдено</th><th class="n">пропущено</th><th class="n">лишних</th><th class="n">найдено, доля</th><th class="n">точность</th></tr></thead><tbody>' +
      qrow(["обученная (повёрнутые рамки)", t.oriented.found, t.oriented.missed, t.oriented.extra, p1(t.oriented.recall), p1(t.oriented.precision)], "best") +
      qrow(["готовая COCO", c.iou50.found, c.iou50.missed, c.iou50.extra, p1(c.iou50.recall), p1(c.iou50.precision)]) +
      '</tbody></table><p class="note">Почти все «лишние» рамки обученной модели — настоящие коровы у края кадра, которых авторы не размечали.</p></div></div>');
  }
  const ctr = (x, title, text, good, note) => x ? '<div class="panel"><h2>' + title + '</h2><div class="body">' +
    verdict(text, good) +
    '<table class="t"><thead><tr><th>вариант</th><th class="n">узнана с первой попытки (Rank-1)</th><th class="n">mAP</th></tr></thead><tbody>' +
    qrow(["обученная модель", p1(x.trained_mask0.rank1), dec(x.trained_mask0.mAP, 3)], "best") +
    qrow(["необученная сеть (ImageNet) — контроль", p1(x.untrained_mask0.rank1), dec(x.untrained_mask0.mAP, 3)]) +
    qrow(["обученная, центр снимка закрыт — контроль", p1(x.trained_mask60.rank1), dec(x.trained_mask60.mAP, 3)]) +
    '</tbody></table><p class="note">' + note + "</p></div></div>" : "";
  parts.push(ctr(q.reid_controls, "Узнать корову по спине (53 коровы, которых модель не видела)",
    "Проверка осмысленная: обучение даёт прирост с ~62% до ~98%.", true,
    "Контроль: без обучения сеть узнаёт заметно хуже — значит, цифра не от «лёгкого» датасета."));
  parts.push(ctr(q.face_controls, "Узнать корову по морде (84 коровы, которых модель не видела)",
    "Не доказано: необученная сеть почти так же хороша.", false,
    "Фото одной коровы в этом датасете сняты за секунды, поэтому задача слишком лёгкая. Нужны фото разных дней."));
  if (q.video) {
    const v = q.video;
    parts.push('<div class="panel"><h2>Вся цепочка на реальных видео (Cows2021, 8–11 марта)</h2><div class="body">' +
      verdict("Ошибок «одна корова в двух местах» — " + v.conflicts + "; узнано " +
        Math.round(100 * v.identified / Math.max(1, v.tracks)) + "% дорожек, новых коров — хуже.", v.conflicts === 0) +
      '<table class="t"><tbody>' +
      qrow(["роликов", v.clips]) + qrow(["дорожек", v.tracks]) +
      qrow(["узнано", v.identified + " (" + Math.round(100 * v.identified / Math.max(1, v.tracks)) + "%)"]) +
      qrow(["«не знаю»", v.unknown]) + qrow(["одна корова в двух местах сразу", v.conflicts]) +
      qrow(["узнано: коровы из обучения модели", v.identified_seen_in_training]) +
      qrow(["узнано: коровы, которых модель не видела", v.identified_unseen_in_training]) +
      qrow(["скорость, кадров/с", dec(v.fps, 1)]) +
      '</tbody></table><p class="note">Ответов «кто на видео» в датасете нет. Сверка глазами 40 дорожек: 35 совпали, 5 спорных, явных ошибок нет.</p></div></div>');
  }
  const lc = q.lying_check;
  if (lc && lc.after) {
    const rows = (lc.before || []).map(b => qrow([esc(b.model), p0(b.standing_recall), p0(b.lying_recall), p0(b.precision), "—"]))
      .join("") + qrow([esc(lc.after.model), p0(lc.after.standing_recall), p0(lc.after.lying_recall),
        p0(lc.after.precision), p1(lc.after.class_accuracy)], "best");
    parts.push('<div class="panel"><h2>Другая ферма: стоит корова или лежит (MmCows, США)</h2><div class="body">' +
      verdict("Без дообучения детектор на другой ферме почти слеп; после дообучения на 12 часах записи — " +
        p0(lc.after.recall) + " коров, поза верна в " + p1(lc.after.class_accuracy) + ".", true) +
      "<p>Проверка — на " + lc.after.images + " кадрах 14:00–24:00, которых модель не видела (учили на 00:00–12:00). " +
      "Коров стоя " + lc.after.cows_standing + ", лёжа " + lc.after.cows_lying + ".</p>" +
      '<table class="t"><thead><tr><th>модель</th><th class="n">стоящих найдено</th><th class="n">лежащих найдено</th>' +
      '<th class="n">точность</th><th class="n">поза верна</th></tr></thead><tbody>' + rows + "</tbody></table>" +
      '<p class="note">Лежащих находит реже (88%), поэтому время лёжа по камере — по найденным рамкам.</p></div></div>');
  }
  const sc = q.sensor_check;
  if (sc) {
    const names = { feeder_share: "ест", drinker_share: "пьёт", resting_share: "лежит", activity_rate: "двигается", milk_kg: "удой" };
    const spread = Object.entries(sc.relative_spread || {}).map(([k, v]) => qrow([names[k] || k, p0(v)])).join("");
    const evs = (sc.events || []).map(e => "<tr><td class=\"n\">" + ddmm(e.day) + "</td><td>" +
      (e.cow === "стадо" ? "всё стадо" : '<span class="cow">' + esc(e.cow) + "</span>") + "</td><td>" +
      (e.cow === "стадо" ? badge("herd", "всё стадо") : badge(EVENT_STATUS[e.severity])) + "</td><td>" + esc(e.title) + "</td></tr>").join("");
    parts.push('<div class="panel"><h2>Выявление отклонений на реальных сутках (MmCows: датчики вместо камеры)</h2><div class="body">' +
      verdict("Ложных «проверить» — " + sc.alerts + " за " + sc.cow_days_assessed + " коров-суток здорового стада.", sc.alerts === 0) +
      '<div class="two"><div><table class="t"><tbody>' +
      qrow(["коров", sc.cows]) + qrow(["суток", sc.days]) +
      qrow(["оценено коров-суток (первые 5 суток — на норму)", sc.cow_days_assessed]) +
      qrow(["«проверить»", sc.alerts]) + qrow(["«на заметку»", sc.watch]) + qrow(["«признаки охоты»", sc.estrus]) +
      qrow(["события «всё стадо»", sc.herd_events + " из " + sc.days + " суток"]) +
      "</tbody></table></div>" +
      '<div><table class="t"><thead><tr><th>признак</th><th class="n">обычный разброс суток у коровы</th></tr></thead><tbody>' +
      spread + "</tbody></table>" +
      (sc.zones_vs_labels_0725 ? '<p class="note">Зона кормового стола ловит ' + p0(sc.zones_vs_labels_0725.feed_recall) +
        " еды, из времени в зоне " + p0(sc.zones_vs_labels_0725.feed_precision) + " — еда. Поилка: питьё — лишь " +
        p0(sc.zones_vs_labels_0725.drink_precision) + " времени у поилки.</p>" : "") + "</div></div>" +
      '<table class="t"><thead><tr><th class="n">сутки</th><th>кто</th><th>событие</th><th>что</th></tr></thead><tbody>' +
      evs + "</tbody></table>" +
      '<p class="note">Диагнозов за время записи в журнале фермы нет, поэтому это проверка ложных тревог, а не выявления болезней.</p></div></div>');
  }
  if (q.farm_chain) parts.splice(1, 0, farmChainPanel(q.farm_chain));
  $app.innerHTML = parts.join("");
}

// Вся цепочка ТЗ на ферме MmCows: узнавание → tracking → активность, с ответами.
function farmChainPanel(fc) {
  const pc = v => v == null ? "—" : dec(100 * v, 1) + "%";
  const trk = fc.tracking || {};
  const names = Object.keys(trk);
  const ours = names.find(n => n.includes("(наш)")) || names[names.length - 1];
  const careful = names.find(n => n.includes("осторожный"));
  const o = trk[ours] || {};
  const frameId = o["номер в кадре"] || {};
  const idt = o["номер коровы во времени"] || {};
  const cf = careful ? (trk[careful]["номер в кадре"] || {}) : null;
  const trkRows = names.map(n => {
    const v = trk[n], f = v["номер в кадре"] || {}, i = v["номер коровы во времени"] || {}, t = v["дорожки (номер трека)"];
    return qrow([esc(n), pc(f["верно"]), pc(f["не знаю"]), pc(f["ошибка"]), pc(i.idf1), i.num_switches,
      t ? pc(t.idf1) : "—", t ? pc(t.mota) : "—"], n === ours ? "best" : "");
  }).join("");
  const cut = o["разрез"] || {};
  const cutRows = Object.keys(cut).map(g => qrow([esc(g), pc(cut[g]["верно"]), pc(cut[g]["не знаю"]),
    pc(cut[g]["ошибка"]), cut[g].n])).join("");
  const reid = fc["узнавание"] || {};
  const reidRows = Object.keys(reid).filter(k => reid[k] && reid[k].rank1 != null)
    .map(k => qrow([esc(k), pc(reid[k].rank1), dec(reid[k].mAP, 3)], k === "дообученная" ? "best" : "")).join("");
  const act = fc["активность"] || {}, perfect = fc["активность при номере из разметки"] || {};
  const actRow = (label, a) => ["лёжа", "у корма"].map(k => a[k] ? qrow([esc(label + ": " + k),
    dec(a[k]["средняя ошибка, ч"], 2) + " ч", dec(a[k]["наибольшая ошибка, ч"], 2) + " ч",
    a[k]["связь камера—люди (r)"] == null ? "—" : dec(a[k]["связь камера—люди (r)"], 2),
    "±" + dec(a[k]["разброс между коровами у людей, ч"], 2) + " ч"]) : "").join("");
  return '<div class="panel"><h2>Вся цепочка на другой ферме с ответами (MmCows: 16 коров, 4 камеры, кадр раз в 15 с)</h2><div class="body">' +
    verdict("Номер коровы во времени: IDF1 " + pc(idt.idf1) + ", в кадре верен в " + pc(frameId["верно"]) + " случаев" +
      (cf ? "; с осторожным порогом чужой номер — только " + pc(cf["ошибка"]) + ", остальное — «не знаю»" : "") + ".", true) +
    "<p>Обучение и регистрация — 03–12 ч (номер из разметки играет роль бирки), проверка — 14–24 ч, " +
    "включая вечер и ночь при лампах. Те же 16 коров в тот же день — это проверка «узнать через несколько часов при другом свете», а не «через месяц».</p>" +
    '<table class="t"><thead><tr><th>вариант</th><th class="n">номер в кадре верно</th><th class="n">«не знаю»</th>' +
    '<th class="n">ошибка</th><th class="n">IDF1 номера</th><th class="n">подмен номера</th>' +
    '<th class="n">IDF1 дорожек</th><th class="n">MOTA</th></tr></thead><tbody>' + trkRows + "</tbody></table>" +
    '<p class="note">IDF1 — доля времени, когда у коровы правильный номер (метрика из ТЗ). «Подмен номера» — сколько раз номер коровы сменился на чужой. ' +
    "MOTA учитывает и пропуски детектора.</p>" +
    '<div class="two"><div><table class="t"><thead><tr><th>когда и где</th><th class="n">верно</th><th class="n">«не знаю»</th>' +
    '<th class="n">ошибка</th><th class="n">рамок</th></tr></thead><tbody>' + cutRows + "</tbody></table></div>" +
    '<div><table class="t"><thead><tr><th>узнавание по одной вырезке</th><th class="n">Rank-1</th><th class="n">mAP</th></tr></thead><tbody>' +
    reidRows + "</tbody></table>" +
    '<p class="note">«Центр закрыт» — контроль: если модель узнаёт корову без самой коровы, она узнаёт место.</p></div></div>' +
    '<table class="t"><thead><tr><th>активность конкретной коровы за 10 ч</th><th class="n">средняя ошибка</th>' +
    '<th class="n">наибольшая</th><th class="n">связь с людьми (r)</th><th class="n">коровы различаются на</th></tr></thead><tbody>' +
    actRow("наш номер", act) + actRow("номер из разметки", perfect) + "</tbody></table>" +
    '<p class="note">Ошибка меньше, чем различие между коровами, — значит, камера различает «свою» норму каждой коровы.</p>' +
    "</div></div>";
}

barInfo();
route();
