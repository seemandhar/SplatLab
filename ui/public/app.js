/* SplatLab — frontend. Zero dependencies. */
'use strict';

const S = {
  token: localStorage.getItem('gss_token') || '',
  config: null,
  runs: [],
  session: { alive: false, active_run_id: null },
  runId: null,
  status: null,
  events: [],
  evtTotal: 0,
  es: null,
  clockBase: 0,
  clockAt: Date.now(),
  clockRunning: false,
  tmuxPoll: null,
  activeTab: 'overview',
};

// ───────────────────────── utils ─────────────────────────
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtH = (s) => {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
  return `${h}:${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`;
};
const ago = (iso) => {
  if (!iso) return '';
  const d = (Date.now() - new Date(iso).getTime()) / 1000;
  if (d < 60) return `${Math.floor(d)}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
};
const num = (v, d = 3) => (typeof v === 'number' ? (v >= 0 ? '+' : '') + v.toFixed(d) : '—');
const deltaCls = (v) => (typeof v !== 'number' ? 'delta-zero' : v > 0.001 ? 'delta-pos' : v < -0.001 ? 'delta-neg' : 'delta-zero');

function toast(msg, kind = '') {
  const t = $('#toast');
  t.textContent = msg; t.className = 'toast ' + kind; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), 4200);
}

// ───────────────────────── api ─────────────────────────
async function api(path, opts = {}) {
  const headers = Object.assign({}, opts.headers);
  if (S.token) headers['Authorization'] = 'Bearer ' + S.token;
  if (opts.body) headers['Content-Type'] = 'application/json';
  const res = await fetch(path, Object.assign({}, opts, { headers }));
  if (res.status === 401) { showTokenModal(); throw new Error('unauthorized'); }
  const ct = res.headers.get('content-type') || '';
  const data = ct.includes('json') ? await res.json() : await res.text();
  if (!res.ok) { const e = new Error((data && data.error) || res.statusText); e.data = data; e.status = res.status; throw e; }
  return data;
}

function showTokenModal() { $('#tokenModal').hidden = false; }
$('#tokenSave').onclick = () => {
  S.token = $('#tokenInput').value.trim();
  localStorage.setItem('gss_token', S.token);
  $('#tokenModal').hidden = true;
  boot();
};

// ───────────────────────── tabs ─────────────────────────
$('#tabs').addEventListener('click', (e) => {
  const t = e.target.closest('.tab'); if (!t) return;
  selectTab(t.dataset.tab);
});
function selectTab(name) {
  S.activeTab = name;
  $$('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
  $$('.panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === name));
  if (name === 'live') startTmuxPoll(); else stopTmuxPoll();
  if (name === 'paper') renderPaper();
  if (name === 'new') refreshLaunchBlocked();
}

// ───────────────────────── boot ─────────────────────────
async function boot() {
  try { S.config = await api('/api/config'); fillLaunchForm(); } catch (e) { if (e.message === 'unauthorized') return; }
  await refreshRuns();
  // choose run: active tmux run > previously-selected > most recent
  let pick = S.session.active_run_id || S.runId || (S.runs[0] && S.runs[0].run_id);
  if (pick) await selectRun(pick);
  pollGpus();
  setInterval(pollGpus, 4000);
  setInterval(tickClock, 1000);
}

async function refreshRuns() {
  const d = await api('/api/runs');
  S.runs = d.runs; S.session = d.session;
  renderRunSelect(); renderRunsGrid(); refreshLaunchBlocked(); renderActionButtons();
}

async function selectRun(id) {
  if (!id) return;
  S.runId = id;
  try { S.status = await api('/api/runs/' + id); } catch { return; }
  // reset event stream
  S.events = []; S.evtTotal = 0;
  openStream(id);
  syncClockFromStatus();
  renderAll();
}

// ───────────────────────── SSE ─────────────────────────
function openStream(id) {
  if (S.es) { S.es.close(); S.es = null; }
  const url = '/api/runs/' + id + '/stream' + (S.token ? '?token=' + encodeURIComponent(S.token) : '');
  const es = new EventSource(url);
  S.es = es;
  setConn('connecting…', false);
  es.onopen = () => setConn('live', true);
  es.onerror = () => setConn('reconnecting…', false);
  es.addEventListener('status', (e) => { S.status = JSON.parse(e.data); syncClockFromStatus(); renderAll(); });
  es.addEventListener('tick', (e) => { const d = JSON.parse(e.data); S.clockBase = d.wall_elapsed_s; S.clockAt = Date.now(); S.clockRunning = d.status === 'running'; if (S.status) S.status.status = d.status; renderPill(); });
  es.addEventListener('events', (e) => {
    const d = JSON.parse(e.data);
    if (d.total < S.evtTotal) S.events = []; // run switched / truncated
    for (const ev of d.events) S.events.push(ev);
    if (S.events.length > 500) S.events = S.events.slice(-500);
    S.evtTotal = d.total;
    renderEvents();
  });
  es.addEventListener('session', () => { refreshRuns(); renderActionButtons(); });
}
function setConn(text, live) { const c = $('#connDot'); c.classList.toggle('live', live); $('#connText').textContent = text; }

// ───────────────────────── clock ─────────────────────────
function syncClockFromStatus() {
  if (!S.status) return;
  S.clockBase = S.status.wall_elapsed_s || 0;
  S.clockAt = Date.now();
  S.clockRunning = S.status.status === 'running';
  // if running but ledger wall is stale (0), fall back to created_at elapsed
  if (S.clockRunning && S.clockBase === 0 && S.status.created_at) {
    S.clockBase = Math.max(0, (Date.now() - new Date(S.status.created_at).getTime()) / 1000);
  }
}
function tickClock() {
  if (!S.status) return;
  renderPill();
}
function currentWall() {
  let w = S.clockBase;
  if (S.clockRunning) w += (Date.now() - S.clockAt) / 1000;
  return w;
}

// ───────────────────────── render: top bar ─────────────────────────
function renderPill() {
  const st = S.status;
  const dot = $('#pillDot'), runEl = $('#pillRun'), clk = $('#pillClock'), cap = $('#pillCap');
  if (!st) { runEl.textContent = 'no run'; clk.textContent = '—'; cap.textContent = ''; dot.className = 'dot'; return; }
  dot.className = 'dot ' + st.status;
  runEl.textContent = st.run_id;
  clk.textContent = fmtH(currentWall());
  cap.textContent = '/ ' + (st.wall_hours_cap || 24) + 'h · ' + st.status;
}
function renderRunSelect() {
  const sel = $('#runSelect');
  sel.innerHTML = S.runs.map((r) => {
    const tag = r.run_id === S.session.active_run_id && S.session.alive ? ' ● live' : '';
    return `<option value="${r.run_id}" ${r.run_id === S.runId ? 'selected' : ''}>${r.run_id} · ${r.status}${tag}</option>`;
  }).join('');
  sel.onchange = () => selectRun(sel.value);
}
function pollGpus() {
  api('/api/gpus').then((g) => {
    const strip = $('#gpuStrip');
    if (!g.available) { strip.innerHTML = '<span style="color:var(--text-faint);font-size:11px">no gpu info</span>'; return; }
    strip.innerHTML = g.gpus.map((gp) => {
      const pct = gp.mem_total ? Math.round((gp.mem_used / gp.mem_total) * 100) : 0;
      const cls = pct > 70 ? 'full' : pct > 8 ? 'busy' : '';
      return `<span class="gpu-bar ${cls}" title="GPU ${gp.index}: ${gp.mem_used}/${gp.mem_total} MiB${gp.util != null ? ' · ' + gp.util + '% util' : ''}"><i style="height:${Math.max(4, pct)}%"></i></span>`;
    }).join('');
  }).catch(() => {});
}

// ───────────────────────── render: action buttons ─────────────────────────
function renderActionButtons() {
  const alive = S.session.alive;
  const st = S.status ? S.status.status : null;
  const isActive = S.runId && S.runId === S.session.active_run_id;
  $('#btnStop').disabled = !(alive && isActive && (st === 'running' || st === 'stopping'));
  $('#btnResume').disabled = !(!alive && st && ['stopping', 'stopped', 'failed', 'running'].includes(st));
  $('#btnKill').disabled = !alive;
}
$('#btnStop').onclick = async () => {
  if (!confirm('Write a graceful stop signal? The master finishes in-flight scenes, then halts.')) return;
  try { const r = await api('/api/runs/' + S.runId + '/stop', { method: 'POST', body: '{}' }); toast(r.message, 'ok'); refreshRuns(); } catch (e) { toast(e.message, 'err'); }
};
$('#btnResume').onclick = async () => {
  try { const r = await api('/api/runs/' + S.runId + '/resume', { method: 'POST', body: '{}' }); toast(r.message, 'ok'); setTimeout(refreshRuns, 2000); } catch (e) { toast((e.data && e.data.message) || e.message, 'err'); }
};
$('#btnKill').onclick = async () => {
  if (!confirm('Hard-kill the tmux session? The ledger survives; use Resume to continue.')) return;
  try { const r = await api('/api/runs/' + S.runId + '/kill', { method: 'POST', body: '{}' }); toast(r.message, 'ok'); setTimeout(refreshRuns, 1500); } catch (e) { toast(e.message, 'err'); }
};

// ───────────────────────── render: overview ─────────────────────────
function renderAll() { renderPill(); renderRunSelect(); renderActionButtons(); renderOverview(); renderExperiments(); renderEvents(); renderPaper(); }

function renderOverview() {
  const st = S.status;
  if (!st) { $('#ovEmpty').hidden = false; $('#ovBody').hidden = true; return; }
  $('#ovEmpty').hidden = true; $('#ovBody').hidden = false;

  $('#cardStatus').innerHTML = `<h4>Status</h4>
    <div class="big neutral" style="display:flex;align-items:center;gap:10px"><span class="dot ${st.status}"></span>${esc(st.status)}</div>
    <div class="kv"><span class="k">wall</span><span class="v">${fmtH(currentWall())} / ${st.wall_hours_cap}h</span></div>
    <div class="kv"><span class="k">created</span><span class="v">${ago(st.created_at)}</span></div>
    ${st.regime_hint ? `<div class="kv"><span class="k">regime</span><span class="v">${esc(st.regime_hint)}</span></div>` : ''}
    ${st.stop_reason ? `<div class="sub">${esc(st.stop_reason)}</div>` : ''}`;

  const bh = st.best_headline, best = st.best;
  $('#cardBest').innerHTML = `<h4>Best result</h4>` + (best ? `
    <div class="big ${bh ? deltaCls(bh.value).replace('delta-', '') : 'neutral'}">${bh ? num(bh.value) : '—'}</div>
    <div class="sub">${bh ? esc(bh.key) : 'no metric'}</div>
    <div class="kv"><span class="k">method</span><span class="v" style="text-align:right;max-width:160px">${esc((best.method_name || '').slice(0, 60))}</span></div>
    ${best.mechanism_class ? `<div class="kv"><span class="k">class</span><span class="v"><span class="pill purple">${esc(best.mechanism_class)}</span></span></div>` : ''}
    ${best.cross_scene_winner ? `<div class="kv"><span class="k">cross-scene</span><span class="v"><span class="pill mint">winner</span></span></div>` : ''}`
    : `<div class="sub">No winner yet.</div>`);

  $('#cardIdeas').innerHTML = `<h4>Idea pipeline</h4>
    <div class="kv"><span class="k">tried</span><span class="v">${st.ideas.tried}</span></div>
    <div class="kv"><span class="k">queued</span><span class="v">${st.ideas.queued}</span></div>
    <div class="kv"><span class="k">rejected</span><span class="v">${st.ideas.rejected}</span></div>
    <div class="kv"><span class="k">experiments</span><span class="v">${st.n_experiments}</span></div>`;

  const b = st.budget || {};
  $('#cardBudget').innerHTML = `<h4>Budget</h4>
    ${b.usd_spent_est != null ? `<div class="kv"><span class="k">$ spent</span><span class="v">${(+b.usd_spent_est).toFixed(2)}</span></div>` : ''}
    ${b.gpu_hours != null ? `<div class="kv"><span class="k">gpu-hours</span><span class="v">${(+b.gpu_hours).toFixed(1)}</span></div>` : ''}
    ${b.opus_tokens != null ? `<div class="kv"><span class="k">opus tok</span><span class="v">${(b.opus_tokens / 1e6).toFixed(2)}M</span></div>` : ''}
    ${b.sonnet_tokens != null ? `<div class="kv"><span class="k">sonnet tok</span><span class="v">${(b.sonnet_tokens / 1e6).toFixed(2)}M</span></div>` : ''}
    <div class="kv"><span class="k">cap</span><span class="v">${st.stop_condition && st.stop_condition.max_budget_usd ? '$' + st.stop_condition.max_budget_usd : 'wall-clock'}</span></div>`;

  $('#cardTarget').innerHTML = `<h4>Target</h4>
    <div class="proseblock" style="border-left-color:var(--cyan)">${esc(st.target.user_intent || '—')}</div>
    <div style="display:flex;gap:18px;margin-top:10px;flex-wrap:wrap;font-size:12px" class="mono">
      ${st.target.paper_url ? `<a href="${esc(st.target.paper_url)}" target="_blank">paper ↗</a>` : ''}
      ${st.target.git_url ? `<a href="${esc(st.target.git_url)}" target="_blank">repo ↗</a>` : ''}
      ${st.target.track ? `<span class="pill gray">track ${esc(st.target.track)}</span>` : ''}
      ${st.dataset ? `<span class="pill gray">${esc(st.dataset)}</span>` : ''}
      ${st.scenes && st.scenes.length ? `<span class="pill gray">${st.scenes.length} scenes</span>` : ''}
    </div>`;

  $('#cardThesis').innerHTML = st.thesis ? `<h4>Research thesis</h4><div class="proseblock">${esc(st.thesis)}</div>` : '';
  $('#cardThesis').style.display = st.thesis ? 'block' : 'none';

  // milestones
  const ms = st.milestones;
  if (ms && typeof ms === 'object') {
    const items = Array.isArray(ms) ? ms.map((m, i) => ['M' + (i + 1), m]) : Object.entries(ms);
    $('#cardMilestones').style.display = 'block';
    $('#cardMilestones').innerHTML = `<h4>Milestones</h4><div class="mstones">` +
      items.map(([k, v]) => `<div class="mstone"><div class="mlabel">${esc(k)}</div><div class="mtext">${esc(typeof v === 'string' ? v : JSON.stringify(v))}</div></div>`).join('') + `</div>`;
  } else $('#cardMilestones').style.display = 'none';

  // baselines
  const rep = st.target.reported_metrics || {}, repro = st.target.reproduced_metrics || {};
  const keys = [...new Set([...Object.keys(rep), ...Object.keys(repro || {})])];
  if (keys.length) {
    $('#cardBaselines').style.display = 'block';
    $('#cardBaselines').innerHTML = `<h4>Baselines (reported vs reproduced)</h4>` +
      keys.map((k) => `<div class="kv"><span class="k">${esc(k)}</span><span class="v">${rep[k] != null ? JSON.stringify(rep[k]) : '—'} → ${repro && repro[k] != null ? JSON.stringify(repro[k]) : '—'}</span></div>`).join('');
  } else $('#cardBaselines').style.display = 'none';
}

// ───────────────────────── render: experiments ─────────────────────────
function stageOf(s) {
  s = (s || '').toLowerCase();
  if (/(done|winner|multi|near.?miss|fail-train|regression)/.test(s)) return 'multi';
  if (/probe/.test(s)) return 'probe';
  return 'smoke';
}
function statusPill(s) {
  s = s || 'unknown';
  const l = s.toLowerCase();
  let c = 'gray';
  if (/pass|winner|done|promoted/.test(l)) c = 'mint';
  else if (/probe|running|pending/.test(l)) c = 'amber';
  else if (/fail|reject|regress|crash/.test(l)) c = 'red';
  else if (/smoke|implement/.test(l)) c = 'purple';
  return `<span class="pill ${c}">${esc(s)}</span>`;
}
function renderExperiments() {
  const st = S.status; if (!st) return;
  const exps = st.experiments || [];
  const buckets = { smoke: [], probe: [], multi: [] };
  for (const e of exps) buckets[stageOf(e.status)].push(e);
  const col = (key, cls, label) => `<div class="casc-col casc-${cls}"><h5>${label} <span class="casc-count">${buckets[key].length}</span></h5>` +
    buckets[key].slice(0, 8).map((e) => `<div class="casc-item"><span>${esc(e.exp_id.replace('exp_', ''))}</span><span class="${deltaCls(e.delta)}">${e.delta != null ? num(e.delta) : ''}</span></div>`).join('') + `</div>`;
  $('#expCascade').innerHTML = col('smoke', 'smoke', 'Smoke') + col('probe', 'probe', 'Probe') + col('multi', 'multi', 'Multi-scene');

  $('#expTable thead').innerHTML = `<tr><th>exp</th><th>idea</th><th>status</th><th>scenes</th><th>Δ</th><th>mechanism</th><th>smoke</th><th>diff</th></tr>`;
  $('#expTable tbody').innerHTML = exps.length ? exps.map((e) => `<tr>
      <td class="mono">${esc(e.exp_id.replace('exp_', ''))}</td>
      <td class="mono">${esc(e.idea_id || '—')}</td>
      <td>${statusPill(e.status)}</td>
      <td class="mono">${e.scenes_total ? `${e.scenes_done}/${e.scenes_total}` : (e.scene || '—')}</td>
      <td class="mono ${deltaCls(e.delta)}">${e.delta != null ? num(e.delta) : '—'}</td>
      <td>${e.mechanism_class ? `<span class="pill purple">${esc(e.mechanism_class)}</span>` : '—'}</td>
      <td class="mono">${e.smoke_verdict ? esc(e.smoke_verdict) : '—'}</td>
      <td class="mono">${e.diff_lines != null ? e.diff_lines + 'L' : '—'}</td>
    </tr>`).join('') : `<tr><td colspan="8" style="color:var(--text-faint);padding:20px">No experiments yet.</td></tr>`;
}

// ───────────────────────── render: events / live ─────────────────────────
function evtClass(type) {
  const t = (type || '').toLowerCase();
  if (/win|best|promot/.test(t)) return 'win';
  if (/fail|reject|regress|error|crash|stall/.test(t)) return 'fail';
  if (/paper|figure|latex|review|rebuttal|revis/.test(t)) return 'paper';
  if (/idea|ideat|scout/.test(t)) return 'idea';
  return '';
}
function renderEvents() {
  const feed = $('#eventFeed');
  if (!S.events.length) { feed.innerHTML = `<div style="color:var(--text-faint);padding:16px">No events yet.</div>`; return; }
  const rows = [...S.events].reverse().map((e) => {
    const { type, ts, idx, ...rest } = e;
    const body = Object.entries(rest).filter(([k]) => k !== 'raw').map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`).join('  ') || (e.raw || '');
    const time = ts ? new Date(ts).toLocaleTimeString() : '';
    return `<div class="evt ${evtClass(type)}"><span class="etime">${esc(time)}</span><span class="etype">${esc(type || 'event')}</span><span class="ebody">${esc(body.slice(0, 280))}</span></div>`;
  }).join('');
  feed.innerHTML = rows;
}

function startTmuxPoll() {
  stopTmuxPoll();
  const fetchTmux = () => {
    if (!S.runId) return;
    api('/api/runs/' + S.runId + '/tmux').then((d) => {
      const stEl = $('#consoleState'), con = $('#tmuxConsole');
      stEl.textContent = d.alive ? 'live' : 'offline';
      stEl.className = 'console-state ' + (d.alive ? 'on' : 'off');
      if (d.alive && d.text) { con.textContent = d.text; con.scrollTop = con.scrollHeight; }
      else if (!d.alive) con.textContent = 'No live session. Launch or resume a run to see the master console.';
    }).catch(() => {});
  };
  fetchTmux();
  S.tmuxPoll = setInterval(fetchTmux, 3000);
}
function stopTmuxPoll() { if (S.tmuxPoll) { clearInterval(S.tmuxPoll); S.tmuxPoll = null; } }

// ───────────────────────── render: paper ─────────────────────────
function renderPaper() {
  if (S.activeTab !== 'paper') return;
  const st = S.status, body = $('#paperBody');
  if (!st) { body.innerHTML = `<div class="empty">No run selected.</div>`; return; }
  const p = st.paper, base = '/api/runs/' + st.run_id;
  let html = '';
  if (p.pdf_exists) {
    html += `<div class="artifacts"><a class="btn btn-mint" href="${base}/pdf${S.token ? '?token=' + S.token : ''}" target="_blank">Open PDF ↗</a></div>`;
    html += `<iframe class="paper-frame" src="${base}/pdf${S.token ? '?token=' + S.token : ''}"></iframe>`;
  } else {
    html += `<div class="banner banner-warn">No compiled paper yet for this run.</div>`;
  }
  if (p.artifacts && p.artifacts.length) {
    html += `<div class="artifacts">` + p.artifacts.map((a) => `<a class="btn" href="${base}/file?path=${encodeURIComponent(a)}${S.token ? '&token=' + S.token : ''}" target="_blank">${esc(a)} ↗</a>`).join('') + `</div>`;
  }
  if (p.figures && p.figures.length) {
    html += `<h3>Figures (${p.figures.length})</h3><div class="fig-gallery">` +
      p.figures.map((f) => `<div class="fig"><img loading="lazy" src="${base}/file?path=${encodeURIComponent(f)}${S.token ? '&token=' + S.token : ''}"><div class="cap">${esc(f.split('/').pop())}</div></div>`).join('') + `</div>`;
  }
  body.innerHTML = html;
}

// ───────────────────────── render: runs grid ─────────────────────────
function renderRunsGrid() {
  const grid = $('#runsGrid');
  grid.innerHTML = S.runs.map((r) => {
    const live = r.run_id === S.session.active_run_id && S.session.alive;
    const bd = r.best_delta;
    return `<div class="run-card ${r.run_id === S.runId ? 'selected' : ''}" data-run="${r.run_id}">
      <div class="rc-top"><span class="dot ${r.status}"></span><span class="rc-id">${r.run_id}</span>${live ? '<span class="pill mint">live</span>' : ''}${r.has_pdf ? '<span class="pill cyan">pdf</span>' : ''}</div>
      <div class="rc-intent">${esc((r.user_intent || '').slice(0, 160) || 'no description')}</div>
      <div class="rc-foot"><span>${esc(r.status)} · ${fmtH(r.wall_elapsed_s)}</span><span class="${bd ? deltaCls(bd.value) : ''}">${bd ? num(bd.value) + ' ' + bd.key.slice(0, 10) : ''}</span></div>
    </div>`;
  }).join('') || `<div class="empty">No runs yet.</div>`;
  $$('.run-card', grid).forEach((c) => (c.onclick = () => { selectRun(c.dataset.run); selectTab('overview'); }));
}

// ───────────────────────── launch form ─────────────────────────
function fillLaunchForm() {
  const c = S.config; if (!c) return;
  // Don't dump a huge configured idea into the box; show the one-line placeholder
  // and offer a button to load the full configured idea on demand.
  const fullIdea = c.idea || '';
  const loadBtn = $('#loadCfgIdea');
  if (fullIdea.length > 280) {
    $('#fIdea').value = '';
    loadBtn.hidden = false;
    loadBtn.textContent = `▸ Load the configured idea (${fullIdea.length.toLocaleString()} chars)`;
    loadBtn.onclick = () => { $('#fIdea').value = fullIdea; loadBtn.hidden = true; $('#fIdea').focus(); };
  } else {
    $('#fIdea').value = fullIdea;
    loadBtn.hidden = true;
  }
  $('#fHours').value = c.hours || 24;
  $('#fPaper').value = c.paper_url || '';
  $('#fGit').value = c.git_url || '';
  $('#fScenes').value = (c.scenes || []).join(',');
  $('#fBudget').value = c.max_budget_usd || '';
  $('#fPreset').innerHTML = (c.presets || ['all_opus']).map((p) => `<option ${p === c.model_preset ? 'selected' : ''}>${p}</option>`).join('');
}
function refreshLaunchBlocked() {
  const blocked = S.session.alive;
  $('#launchBlocked').hidden = !blocked;
  $('#btnLaunch').disabled = blocked;
}
$('#launchForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const body = {
    idea: $('#fIdea').value.trim(),
    hours: parseInt($('#fHours').value, 10),
    paper_url: $('#fPaper').value.trim(),
    git_url: $('#fGit').value.trim(),
    scenes: $('#fScenes').value.split(',').map((s) => s.trim()).filter(Boolean),
    model_preset: $('#fPreset').value,
    max_budget_usd: $('#fBudget').value ? Number($('#fBudget').value) : null,
  };
  const msg = $('#launchMsg');
  if (!body.idea) { msg.textContent = 'idea is required'; msg.style.color = 'var(--red)'; return; }
  $('#btnLaunch').disabled = true; msg.style.color = 'var(--text-dim)'; msg.textContent = 'launching…';
  const before = new Set(S.runs.map((r) => r.run_id));
  try {
    const r = await api('/api/launch', { method: 'POST', body: JSON.stringify(body) });
    msg.style.color = 'var(--mint)'; msg.textContent = r.message;
    toast('Run launching in tmux…', 'ok');
    // poll for the new run dir to appear (master bootstraps in ~15-60s)
    let tries = 0;
    const poll = setInterval(async () => {
      tries++;
      await refreshRuns();
      const fresh = S.runs.find((x) => !before.has(x.run_id));
      if (fresh) { clearInterval(poll); msg.textContent = 'started: ' + fresh.run_id; await selectRun(fresh.run_id); selectTab('overview'); toast('Run started: ' + fresh.run_id, 'ok'); }
      else if (tries > 40) { clearInterval(poll); msg.textContent = 'launched — check the Live tab for the master console.'; selectTab('live'); }
    }, 3000);
  } catch (err) {
    $('#btnLaunch').disabled = false;
    msg.style.color = 'var(--red)';
    msg.textContent = (err.data && (err.data.message || (err.data.detail && err.data.detail.join(', ')))) || err.message;
  }
});

boot();
