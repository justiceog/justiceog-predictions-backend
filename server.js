"use strict";
// ʝµรƭเcεσɠ predictions backend.
// 1) fetches pre-match data from API-Football and caches it
// 2) saves every prediction before kick-off and settles it after full time (results tracker)
// 3) learns per-league goal scaling from settled matches (self-correcting model)
const http = require("http"), fs = require("fs"), path = require("path");

const KEY = process.env.API_FOOTBALL_KEY;
const PORT = process.env.PORT || 3000;
const TZ = process.env.DISPLAY_TZ || "UTC";
const LEAGUES = (process.env.LEAGUES || "39,140,78,135").split(",").map(s => +s.trim());
const MAX_MATCHES = +(process.env.MAX_MATCHES || 8);
const DAILY_CAP = +(process.env.DAILY_CAP || 90);
const DELAY = +(process.env.CALL_DELAY_MS || 6500);
const TTL = 12 * 3600e3;
const MIN_EDGE = 0.03, PRIOR = 20, AVG = 1.4;
const DIR = process.env.DATA_DIR || __dirname;
const FILE = path.join(DIR, "cache.json");     // disposable: fixtures, odds, injuries
const HFILE = path.join(DIR, "history.json");  // precious: saved predictions, results, learned model

let C = { day: "", used: 0, updated: 0, matches: [], details: {}, warnings: [], fx: {} };
let Hs = { log: {}, model: {} };
try { C = { ...C, ...JSON.parse(fs.readFileSync(FILE, "utf8")) }; } catch {}
try { Hs = { ...Hs, ...JSON.parse(fs.readFileSync(HFILE, "utf8")) }; } catch {}
const write = (f, o) => { try { fs.writeFileSync(f, JSON.stringify(o)); } catch (e) { console.error("save failed:", e.message); } };
const sleep = ms => new Promise(r => setTimeout(r, ms));
const num = x => { const n = parseFloat(x); return isNaN(n) ? null : n; };
const clamp = (x, a = 0.3, b = 3.5) => Math.min(b, Math.max(a, x));
const iso = t => new Date(t).toISOString().slice(0, 10);
const season = () => { const d = new Date(); return d.getUTCMonth() >= 6 ? d.getUTCFullYear() : d.getUTCFullYear() - 1; };

// ---------- model ----------
function pmf(l) { const a = [Math.exp(-l)]; for (let k = 1; k <= 12; k++) a.push(a[k - 1] * l / k); return a; }
function probs(hx, ax) {
  const ph = pmf(hx), pa = pmf(ax), p = { H: 0, D: 0, A: 0, O25: 0, BY: 0 };
  for (let i = 0; i <= 12; i++) for (let j = 0; j <= 12; j++) {
    const w = ph[i] * pa[j];
    if (i > j) p.H += w; else if (i === j) p.D += w; else p.A += w;
    if (i + j > 2.5) p.O25 += w;
    if (i > 0 && j > 0) p.BY += w;
  }
  p.U25 = 1 - p.O25; p.BN = 1 - p.BY;
  return p;
}
// Learned scaling: compares goals actually scored with the raw expected goals, shrunk towards 1 until enough matches exist.
function scales(lgId) {
  const m = Hs.model[lgId];
  if (!m || !m.n) return { sh: 1, sa: 1 };
  const f = (act, raw) => Math.min(1.25, Math.max(0.8, (act + PRIOR * AVG) / (raw + PRIOR * AVG)));
  return { sh: f(m.ah, m.rh), sa: f(m.aa, m.ra) };
}
function applyResult(x, hg, ag) {
  x.settled = true; x.score = [hg, ag];
  const out = { H: hg > ag, D: hg === ag, A: hg < ag, O25: hg + ag > 2.5, U25: hg + ag < 2.5, BY: hg > 0 && ag > 0, BN: !(hg > 0 && ag > 0) };
  x.out = out;
  x.bets.forEach(b => { b.win = out[b.k]; b.profit = b.win ? +(b.o - 1).toFixed(4) : -1; });
  const m = Hs.model[x.lgId] = Hs.model[x.lgId] || { name: x.lg, n: 0, ah: 0, rh: 0, aa: 0, ra: 0 };
  m.n++; m.ah += hg; m.rh += x.rawH; m.aa += ag; m.ra += x.rawA;
}
function summary() {
  const S = Object.values(Hs.log).filter(x => x.settled);
  const bets = S.flatMap(x => x.bets.map(b => ({ ...b, lg: x.lg, m: x.h + " v " + x.a, score: x.score.join("-") })));
  const agg = a => { const n = a.length, w = a.filter(b => b.win).length, pr = a.reduce((s, b) => s + b.profit, 0); return { n, wins: w, hit: n ? w / n : null, profit: +pr.toFixed(2), roi: n ? pr / n : null }; };
  const group = f => { const g = {}; bets.forEach(b => (g[f(b)] = g[f(b)] || []).push(b)); return Object.entries(g).map(([k, a]) => ({ k, ...agg(a) })); };
  const bins = Array.from({ length: 10 }, () => ({ n: 0, p: 0, hits: 0 }));
  let brier = 0;
  S.forEach(x => {
    for (const k of Object.keys(x.P)) { const b = bins[Math.min(9, Math.floor(x.P[k] * 10))]; b.n++; b.p += x.P[k]; b.hits += x.out[k] ? 1 : 0; }
    brier += ["H", "D", "A"].reduce((s, k) => s + Math.pow(x.P[k] - (x.out[k] ? 1 : 0), 2), 0);
  });
  return {
    matches: S.length, pending: Object.values(Hs.log).filter(x => !x.settled).length, all: agg(bets),
    byMarket: group(b => b.k), byLeague: group(b => b.lg),
    calibration: bins.map((b, i) => ({ from: i * 10, to: i * 10 + 10, n: b.n, pred: b.n ? b.p / b.n : null, actual: b.n ? b.hits / b.n : null })).filter(b => b.n),
    brier: S.length ? brier / S.length : null,
    model: Object.entries(Hs.model).map(([id, m]) => ({ lg: m.name, n: m.n, ...scales(id) })),
    recent: bets.slice(-15).reverse()
  };
}

// ---------- API-Football ----------
async function api(ep, params) {
  const day = iso(Date.now());
  if (C.day !== day) { C.day = day; C.used = 0; }
  if (C.used >= DAILY_CAP) throw new Error("Daily request cap reached (" + DAILY_CAP + ")");
  C.used++;
  await sleep(DELAY);
  const r = await fetch("https://v3.football.api-sports.io/" + ep + "?" + new URLSearchParams(params), { headers: { "x-apisports-key": KEY } });
  if (!r.ok) throw new Error(ep + ": HTTP " + r.status);
  const j = await r.json();
  const errs = Array.isArray(j.errors) ? j.errors : Object.values(j.errors || {});
  if (errs.length) throw new Error(ep + ": " + errs.join(" "));
  return j.response || [];
}
async function tryApi(ep, p) { try { return await api(ep, p); } catch (e) { C.warnings.push(e.message); return null; } }

function bestOdds(resp) {
  const o = {}, up = (k, v) => { v = parseFloat(v); if (k && v > (o[k] || 0)) o[k] = v; };
  for (const b of ((resp && resp[0] && resp[0].bookmakers) || [])) for (const bet of b.bets) {
    if (bet.id === 1) bet.values.forEach(v => up({ Home: "H", Draw: "D", Away: "A" }[v.value], v.odd));
    if (bet.id === 5) bet.values.forEach(v => { if (v.value === "Over 2.5") up("O25", v.odd); if (v.value === "Under 2.5") up("U25", v.odd); });
    if (bet.id === 8) bet.values.forEach(v => up(v.value === "Yes" ? "BY" : v.value === "No" ? "BN" : null, v.odd));
  }
  return o;
}

async function enrich(f) {
  const id = f.fixture.id;
  let fx = C.fx[id] || {};
  if (!fx.at || Date.now() - fx.at > TTL) {
    const p = await tryApi("predictions", { fixture: id });
    const o = await tryApi("odds", { fixture: id });
    const i = await tryApi("injuries", { fixture: id });
    const t = p && p[0] && p[0].teams;
    let xg = null;
    if (t) {
      const g = (s, k, w) => num(t[s].league.goals[k].average[w]);
      const hf = g("home", "for", "home"), ha = g("home", "against", "home"), af = g("away", "for", "away"), aa = g("away", "against", "away");
      if ([hf, ha, af, aa].every(x => x !== null)) {
        let rawH = (hf + aa) / 2, rawA = (af + ha) / 2, form = false;
        const l5 = (s, k) => num(t[s].last_5 && t[s].last_5.goals && t[s].last_5.goals[k] && t[s].last_5.goals[k].average);
        const a1 = l5("home", "for"), a2 = l5("home", "against"), b1 = l5("away", "for"), b2 = l5("away", "against");
        if ([a1, a2, b1, b2].every(x => x !== null)) { rawH = 0.75 * rawH + 0.25 * (a1 + b2) / 2; rawA = 0.75 * rawA + 0.25 * (b1 + a2) / 2; form = true; }
        xg = { rawH: clamp(rawH), rawA: clamp(rawA), hf, ha, af, aa, form };
      }
    }
    fx = { ...fx, at: Date.now(), xg, advice: p && p[0] && p[0].predictions && p[0].predictions.advice,
      o: bestOdds(o), inj: (i || []).map(x => ({ t: x.team.id, n: x.player.name, r: x.player.reason })) };
  }
  const mins = (new Date(f.fixture.date) - Date.now()) / 60000;
  if (mins < 120 && mins > -10 && !(fx.lu && fx.lu.length) && (!fx.luAt || Date.now() - fx.luAt > 20 * 60e3)) {
    const l = await tryApi("fixtures/lineups", { fixture: id });
    fx.lu = (l || []).map(x => ({ t: x.team.id, fm: x.formation, xi: x.startXI.map(p => p.player.name) }));
    fx.luAt = Date.now();
  }
  C.fx[id] = fx;
  return fx;
}

function shape(f, fx) {
  const h = f.teams.home, a = f.teams.away, id = "f" + f.fixture.id;
  const raw = fx.xg ? { h: fx.xg.rawH, a: fx.xg.rawA } : { h: 1.4, a: 1.1 };
  const sc = scales(f.league.id);
  const hx = +clamp(raw.h * sc.sh).toFixed(2), ax = +clamp(raw.a * sc.sa).toFixed(2);
  const hasOdds = Object.keys(fx.o || {}).length > 0, hasInfo = (fx.lu && fx.lu.length) || (fx.inj && fx.inj.length);
  const q = +((fx.xg ? 0.6 : 0.3) + (hasOdds ? 0.25 : 0) + (hasInfo ? 0.15 : 0)).toFixed(2);
  const inj = t => (fx.inj || []).filter(x => x.t === t.id).map(x => x.n + (x.r ? " (" + x.r + ")" : "")).join(", ") || "None reported";
  const lu = t => (fx.lu || []).find(x => x.t === t.id);
  const luS = [lu(h), lu(a)];
  const ko = new Date(f.fixture.date).toLocaleString("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZone: TZ });
  return {
    raw, hx, ax, q,
    match: { id, lg: f.league.name, h: h.name, a: a.name, hx, ax, q, o: fx.o || {},
      sub: fx.advice ? "The data provider’s advice is: " + fx.advice + "." : "" },
    detail: {
      ko, v: [f.fixture.venue.name, f.fixture.venue.city].filter(Boolean).join(", ") || "Not available", r: f.fixture.referee || "Not announced",
      pre: fx.xg ? `Season goal averages: ${h.name} score ${fx.xg.hf} and concede ${fx.xg.ha} per home game, while ${a.name} score ${fx.xg.af} and concede ${fx.xg.aa} per away game.${fx.xg.form ? " Recent form is blended in." : ""} The model turns these into the expected goals shown below.` : "There isn’t enough season data for a preview yet.",
      inj: [inj(h), inj(a)], pl: [], bk: [], cr: null,
      lu: luS[0] && luS[1] ? [luS[0].xi, luS[1].xi] : null, fm: [luS[0] && luS[0].fm, luS[1] && luS[1].fm]
    }
  };
}

// Save the prediction before kick-off so results can be judged fairly afterwards.
function record(f, fx, s) {
  const id = f.fixture.id, old = Hs.log[id];
  if (!fx.xg || (old && old.settled) || new Date(f.fixture.date) <= Date.now()) return;
  const P = probs(s.hx, s.ax), bets = [];
  for (const k of Object.keys(fx.o || {})) {
    const ev = P[k] * fx.o[k] - 1;
    if (ev >= MIN_EDGE && s.q >= 0.6) bets.push({ k, p: +P[k].toFixed(4), o: fx.o[k], ev: +ev.toFixed(4) });
  }
  const rd = {}; for (const k of Object.keys(P)) rd[k] = +P[k].toFixed(4);
  Hs.log[id] = { id, lgId: f.league.id, lg: f.league.name, h: f.teams.home.name, a: f.teams.away.name, ko: f.fixture.timestamp * 1000,
    rawH: s.raw.h, rawA: s.raw.a, hx: s.hx, ax: s.ax, q: s.q, P: rd, bets, settled: false };
}

async function settle() {
  const now = Date.now();
  for (const [id, x] of Object.entries(Hs.log)) if (!x.settled && x.ko < now - 10 * 864e5) delete Hs.log[id]; // postponed or never reported
  const pend = Object.values(Hs.log).filter(x => !x.settled && x.ko < now - 3 * 3600e3);
  const byLg = {};
  pend.forEach(x => (byLg[x.lgId] = byLg[x.lgId] || []).push(x));
  for (const [lg, arr] of Object.entries(byLg)) {
    const r = await tryApi("fixtures", { league: lg, season: season(), from: iso(Math.min(...arr.map(x => x.ko))), to: iso(now), status: "FT" });
    if (!r) continue;
    const fin = new Map(r.map(f => [f.fixture.id, f]));
    for (const x of arr) { const f = fin.get(x.id); if (f && f.goals.home !== null) applyResult(x, f.goals.home, f.goals.away); }
  }
  write(HFILE, Hs);
}

let busy = false;
async function refresh() {
  if (busy) return; busy = true; C.warnings = [];
  try {
    if (!KEY) { C.warnings.push("API_FOOTBALL_KEY is not set"); return; }
    await settle();
    const now = Date.now();
    let fixtures = [];
    for (const lg of LEAGUES) { const r = await tryApi("fixtures", { league: lg, season: season(), from: iso(now), to: iso(now + 7 * 864e5) }); if (r) fixtures.push(...r); }
    fixtures = fixtures.filter(f => ["NS", "TBD"].includes(f.fixture.status.short)).sort((a, b) => a.fixture.timestamp - b.fixture.timestamp).slice(0, MAX_MATCHES);
    const matches = [], details = {};
    for (const f of fixtures) { const fx = await enrich(f), s = shape(f, fx); record(f, fx, s); matches.push(s.match); details[s.match.id] = s.detail; }
    C.matches = matches; C.details = details; C.updated = Date.now();
    const keep = new Set(fixtures.map(f => f.fixture.id));
    for (const k of Object.keys(C.fx)) if (!keep.has(+k)) delete C.fx[k];
  } catch (e) { C.warnings.push(e.message); }
  finally { C.warnings = [...new Set(C.warnings)].slice(0, 5); write(FILE, C); write(HFILE, Hs); busy = false; console.log("refresh done; requests used today:", C.used, "warnings:", C.warnings); }
}

// ---------- server ----------
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css" };
const ROOT = path.join(__dirname, "public");
const json = (res, o) => { res.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" }); res.end(JSON.stringify(o)); };
function handler(req, res) {
  const u = new URL(req.url, "http://x");
  if (u.pathname === "/api/matches") return json(res, { matches: C.matches, details: C.details, updated: C.updated, warnings: C.warnings });
  if (u.pathname === "/api/results") return json(res, summary());
  if (u.pathname === "/api/status") return json(res, { keySet: !!KEY, season: season(), requestsToday: C.used, cap: DAILY_CAP, lastRefresh: C.updated, matches: C.matches.length, saved: Object.keys(Hs.log).length, warnings: C.warnings });
  const p = path.join(ROOT, path.normalize(u.pathname === "/" ? "index.html" : u.pathname));
  if (!p.startsWith(ROOT)) { res.writeHead(403); return res.end("Forbidden"); }
  fs.readFile(p, (e, d) => {
    if (e) { res.writeHead(404); return res.end("Not found"); }
    res.writeHead(200, { "content-type": MIME[path.extname(p)] || "application/octet-stream" }); res.end(d);
  });
}
module.exports = { probs, scales, applyResult, summary, Hs, record, shape };
if (require.main === module) {
  http.createServer(handler).listen(PORT, () => console.log("ʝµรƭเcεσɠ predictions running on port " + PORT));
  if (Date.now() - C.updated > TTL) refresh();
  setInterval(() => { if (Date.now() - C.updated > TTL) refresh(); }, 10 * 60e3);
}
