#!/usr/bin/env node
/*
 * SplatLab — zero-dependency control panel for the autonomous GS research lab.
 *
 * The backend holds NO run state of its own. Everything is derived from disk
 * (workspace run_<id> ledger.json + events.jsonl) and from tmux (has-session,
 * capture-pane). The `claude` master lives in a persistent tmux session,
 * independent of both the browser and this server — so refresh / laptop-close /
 * server-restart all reconnect to the same live job for free.
 *
 * Each API endpoint mirrors one /research-family behavior. See ui/README.md and
 * the in-app Help tab for the full mapping.
 *
 * Env:
 *   PORT        (default 8420)
 *   HOST        (default 127.0.0.1; set 0.0.0.0 for LAN — then set GSS_TOKEN)
 *   GSS_TOKEN   bearer token required on /api when set
 *   GSS_PYTHON  python interpreter (default python3) — needs pyyaml
 */
'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const { execFileSync, spawn } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const WORKSPACE = path.join(ROOT, 'workspace');
const PUBLIC = path.join(__dirname, 'public');
const UI_SCRIPTS = path.join(__dirname, 'scripts');
const LEDGER_HELPERS = path.join(ROOT, 'scripts', 'ledger_helpers.py');
const CONFIG_YAML = path.join(ROOT, 'config.yaml');

const TMUX_SESSION = process.env.GSS_SESSION || 'splatlab';
const PY = process.env.GSS_PYTHON || 'python3';
const HOST = process.env.HOST || '127.0.0.1';
const PORT = parseInt(process.env.PORT || '8420', 10);
const TOKEN = process.env.GSS_TOKEN || '';

const RUN_RE = /^run_[0-9_]+$/;
const URL_RE = /^https?:\/\/[^\s"'`$;|&<>\\]+$/;
const SCENE_RE = /^[A-Za-z0-9_]+$/;
const PRESETS = ['optimized', 'all_opus', 'all_sonnet'];

// ───────────────────────────── shell helpers ─────────────────────────────
function run(cmd, args, opts = {}) {
  try {
    const out = execFileSync(cmd, args, { encoding: 'utf8', timeout: 15000, ...opts });
    return { ok: true, code: 0, stdout: out, stderr: '' };
  } catch (e) {
    return { ok: false, code: e.status == null ? -1 : e.status, stdout: e.stdout || '', stderr: e.stderr || String(e) };
  }
}

function tmuxAlive() {
  try {
    execFileSync('tmux', ['has-session', '-t', TMUX_SESSION], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

function tmuxCapture(lines = 200) {
  if (!tmuxAlive()) return { alive: false, text: '' };
  const r = run('tmux', ['capture-pane', '-t', TMUX_SESSION, '-p', '-S', `-${lines}`]);
  return { alive: true, text: r.ok ? r.stdout : '' };
}

// ───────────────────────────── disk helpers ──────────────────────────────
function safeRun(id) {
  if (!RUN_RE.test(id)) return null;
  const dir = path.join(WORKSPACE, id);
  if (!dir.startsWith(WORKSPACE + path.sep)) return null;
  if (!fs.existsSync(path.join(dir, 'ledger.json'))) return null;
  return dir;
}

function readLedger(id) {
  const dir = safeRun(id);
  if (!dir) return null;
  const p = path.join(dir, 'ledger.json');
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      return JSON.parse(fs.readFileSync(p, 'utf8'));
    } catch {
      // mid-write (atomic replace makes this rare); brief retry
    }
  }
  return null;
}

function readEvents(id, since = 0) {
  const dir = safeRun(id);
  if (!dir) return { total: 0, events: [] };
  const p = path.join(dir, 'events.jsonl');
  if (!fs.existsSync(p)) return { total: 0, events: [] };
  let lines;
  try {
    lines = fs.readFileSync(p, 'utf8').split('\n').filter((l) => l.trim());
  } catch {
    return { total: 0, events: [] };
  }
  const total = lines.length;
  const slice = since > 0 ? lines.slice(since) : lines;
  const events = slice.map((l, i) => {
    try {
      return { idx: (since > 0 ? since : 0) + i, ...JSON.parse(l) };
    } catch {
      return { idx: (since > 0 ? since : 0) + i, type: 'raw', raw: l };
    }
  });
  return { total, events };
}

function listRuns() {
  let entries = [];
  try {
    entries = fs.readdirSync(WORKSPACE).filter((d) => RUN_RE.test(d));
  } catch {
    return [];
  }
  const alive = tmuxAlive();
  const runs = entries
    .map((id) => {
      const led = readLedger(id);
      if (!led) return null;
      const b = led.best || {};
      return {
        run_id: id,
        status: led.status || 'unknown',
        stop_reason: led.stop_reason || null,
        created_at: led.created_at || null,
        resumed_at: led.resumed_at || null,
        wall_elapsed_s: (led.budget && led.budget.wall_elapsed_s) || 0,
        user_intent: (led.target && led.target.user_intent) || '',
        best_method: b.method_name || null,
        best_delta: bestHeadlineDelta(b),
        n_experiments: Array.isArray(led.experiments) ? led.experiments.length : 0,
        has_pdf: !!findPdf(id),
      };
    })
    .filter(Boolean)
    .sort((a, b) => (a.created_at < b.created_at ? 1 : -1));

  // The single tmux session drives the most-recent run that is running/stopping
  // (or, if none, the most recent overall).
  let activeRunId = null;
  if (alive) {
    const live = runs.find((r) => r.status === 'running' || r.status === 'stopping');
    activeRunId = (live || runs[0] || {}).run_id || null;
  }
  return { runs, session: { alive, name: TMUX_SESSION, active_run_id: activeRunId } };
}

function bestHeadlineDelta(best) {
  if (!best || !best.delta_vs_target) return null;
  const d = best.delta_vs_target;
  // prefer obvious headline keys, else first numeric
  const prefer = ['avg_psnr', 'avg_human_2scene', 'psnr', 'avg_psnr_delta'];
  for (const k of prefer) if (typeof d[k] === 'number') return { key: k, value: d[k] };
  for (const k of Object.keys(d)) if (typeof d[k] === 'number') return { key: k, value: d[k] };
  return null;
}

function findPdf(id) {
  const dir = safeRun(id);
  if (!dir) return null;
  for (const rel of ['paper.pdf', 'paper/main.pdf', 'paper/paper.pdf', 'main.pdf']) {
    const p = path.join(dir, rel);
    if (fs.existsSync(p)) return rel;
  }
  return null;
}

// ───────────────────────────── status builder ────────────────────────────
function buildStatus(id) {
  const led = readLedger(id);
  if (!led) return null;
  const dir = safeRun(id);
  const cfg = led.config_snapshot || {};
  const scenes = cfg.scenes || led.scenes || led.primary_scenes || [];
  const sc = led.stop_condition || {};
  const wallCap = sc.wall_hours || cfg.hours || (led.config_snapshot && led.config_snapshot.hours) || 24;

  const exps = Array.isArray(led.experiments) ? led.experiments : [];
  const experiments = exps
    .slice(-14)
    .reverse()
    .map((e) => {
      const mps = e.metrics_per_scene || {};
      const scenesDone = Object.keys(mps).length;
      return {
        exp_id: e.exp_id || '?',
        idea_id: e.idea_id || null,
        method_name: e.method_name || null,
        status: e.status || 'unknown',
        scene: e.scene || null,
        scenes_done: scenesDone,
        scenes_total: scenes.length || null,
        mechanism_class: e.mechanism_class || null,
        smoke_verdict: e.smoke_verdict || null,
        diff_lines: e.diff_lines != null ? e.diff_lines : null,
        files_changed: Array.isArray(e.files_changed) ? e.files_changed.length : null,
        delta: pickExpDelta(e),
      };
    });

  const ideas = led.ideas || {};
  const ideaList = (arr) =>
    (Array.isArray(arr) ? arr : []).slice(0, 8).map((i) => ({
      idea_id: i.idea_id || i.id || null,
      hypothesis: i.hypothesis || i.title || i.idea || null,
      mechanism_class: i.mechanism_class || i.hook || null,
      delta: typeof i.psnr_delta === 'number' ? i.psnr_delta : (i.delta_vs_target && i.delta_vs_target.psnr) || null,
    }));

  const pdfRel = findPdf(id);
  const paper = led.paper || {};
  let figures = [];
  try {
    const fdir = path.join(dir, 'paper', 'figures');
    if (fs.existsSync(fdir)) figures = fs.readdirSync(fdir).filter((f) => /\.(png|jpg|jpeg|svg)$/i.test(f)).map((f) => 'paper/figures/' + f);
    else {
      const fdir2 = path.join(dir, 'figures');
      if (fs.existsSync(fdir2)) figures = fs.readdirSync(fdir2).filter((f) => /\.(png|jpg|jpeg|svg)$/i.test(f)).map((f) => 'figures/' + f);
    }
  } catch {}

  const artifacts = [];
  for (const f of ['RUN_REPORT.md', 'thesis.md', 'bottleneck_report.md', 'limitations_report.md', 'research_synthesis.md'])
    if (fs.existsSync(path.join(dir, f))) artifacts.push(f);

  const { session } = listRuns();
  return {
    run_id: led.run_id || id,
    status: led.status || 'unknown',
    stop_reason: led.stop_reason || null,
    created_at: led.created_at || null,
    resumed_at: led.resumed_at || null,
    research_phase: led.research_phase || null,
    wall_elapsed_s: (led.budget && led.budget.wall_elapsed_s) || 0,
    wall_hours_cap: wallCap,
    tmux_active: session.alive && session.active_run_id === (led.run_id || id),
    target: {
      user_intent: (led.target && led.target.user_intent) || '',
      paper_url: (led.target && led.target.paper_url) || null,
      git_url: (led.target && led.target.git_url) || null,
      track: (led.target && led.target.track) || led.preferred_track || null,
      reported_metrics: (led.target && led.target.reported_metrics) || {},
      reproduced_metrics: (led.target && led.target.reproduced_metrics) || null,
    },
    thesis: led.thesis || (paper && paper.thesis) || null,
    regime_hint: led.regime_hint || null,
    key_metric: led.key_metric || null,
    dataset: led.dataset || cfg.dataset || null,
    scenes,
    ideas: {
      tried: Array.isArray(ideas.tried) ? ideas.tried.length : 0,
      queued: Array.isArray(ideas.queued) ? ideas.queued.length : 0,
      rejected: Array.isArray(ideas.rejected) ? ideas.rejected.length : 0,
      queued_list: ideaList(ideas.queued),
      tried_list: ideaList(ideas.tried),
    },
    experiments,
    n_experiments: exps.length,
    best: led.best || null,
    best_headline: bestHeadlineDelta(led.best || {}),
    milestones: led.milestones || null,
    noise_floor: led.noise_floor || null,
    budget: led.budget || {},
    stop_condition: sc,
    paper: {
      pdf_exists: !!pdfRel,
      pdf_rel: pdfRel,
      last_written_at: paper.last_written_at || null,
      last_verified_at: paper.last_verified_at || null,
      open_issues: Array.isArray(paper.open_issues) ? paper.open_issues.length : 0,
      reviewer_rounds: paper.reviewer_rounds != null ? paper.reviewer_rounds : null,
      scope: paper.scope || null,
      figures,
      artifacts,
    },
  };
}

function pickExpDelta(e) {
  const cands = [e.delta_vs_baseline, e.delta_vs_target, e.delta];
  for (const d of cands) {
    if (typeof d === 'number') return d;
    if (d && typeof d.psnr === 'number') return d.psnr;
    if (d && typeof d.avg_psnr === 'number') return d.avg_psnr;
  }
  if (typeof e.avg_psnr_delta === 'number') return e.avg_psnr_delta;
  if (typeof e.smoke_train_psnr_final === 'number' && typeof e.smoke_train_psnr_start === 'number')
    return null; // smoke train psnr isn't a delta; leave for the row's smoke column
  return null;
}

// ───────────────────────────── gpu (best-effort) ─────────────────────────
let gpuCache = { at: 0, data: null };
function gpuInfo() {
  const now = Date.now();
  if (gpuCache.data && now - gpuCache.at < 3000) return gpuCache.data;
  let data = { available: false, gpus: [] };
  const r = run('nvidia-smi', ['--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits']);
  if (r.ok && r.stdout.trim()) {
    const gpus = r.stdout
      .trim()
      .split('\n')
      .map((line) => {
        const [idx, used, total, util] = line.split(',').map((s) => s.trim());
        return { index: +idx, mem_used: +used, mem_total: +total, util: +util, free: +used < 500 };
      });
    data = { available: true, gpus };
  }
  gpuCache = { at: now, data };
  return data;
}

// ───────────────────────────── lifecycle actions ─────────────────────────
function ledgerCmd(dir, args) {
  return run(PY, [LEDGER_HELPERS, ...args, '--workspace', dir]);
}

function patchConfig(payload) {
  return run(PY, [path.join(UI_SCRIPTS, 'patch_config.py'), CONFIG_YAML], { input: JSON.stringify(payload) });
}

function tmuxSend(sendCmd) {
  // Detached so the ~13s bootstrap+send sequence doesn't block the HTTP response.
  const child = spawn('bash', [path.join(UI_SCRIPTS, 'tmux_start.sh')], {
    cwd: ROOT,
    env: { ...process.env, GSS_SESSION: TMUX_SESSION, GSS_SEND_CMD: sendCmd },
    detached: true,
    stdio: 'ignore',
  });
  child.unref();
}

function buildResearchCmd(b) {
  let cmd = `/research --hours ${b.hours}`;
  if (b.paper_url) cmd += ` --paper-url "${b.paper_url}"`;
  if (b.git_url) cmd += ` --git-url "${b.git_url}"`;
  if (b.scenes && b.scenes.length) cmd += ` --scenes ${b.scenes.join(',')}`;
  if (b.model_preset) cmd += ` --model-preset ${b.model_preset}`;
  if (b.max_budget_usd) cmd += ` --max-budget-usd ${b.max_budget_usd}`;
  return cmd;
}

function validateLaunch(b) {
  const errs = [];
  if (!b || typeof b.idea !== 'string' || !b.idea.trim()) errs.push('idea is required');
  if (b.idea && b.idea.length > 20000) errs.push('idea too long');
  let hours = parseInt(b.hours, 10);
  if (!Number.isFinite(hours) || hours < 1 || hours > 168) errs.push('hours must be 1..168');
  if (b.paper_url && !URL_RE.test(b.paper_url)) errs.push('paper_url must be a clean http(s) URL');
  if (b.git_url && !URL_RE.test(b.git_url)) errs.push('git_url must be a clean http(s) URL');
  let scenes = [];
  if (b.scenes) {
    scenes = (Array.isArray(b.scenes) ? b.scenes : String(b.scenes).split(',')).map((s) => String(s).trim()).filter(Boolean);
    if (!scenes.every((s) => SCENE_RE.test(s))) errs.push('scenes must be alphanumeric');
  }
  if (b.model_preset && !PRESETS.includes(b.model_preset)) errs.push('invalid model_preset');
  let budget = null;
  if (b.max_budget_usd != null && b.max_budget_usd !== '') {
    budget = Number(b.max_budget_usd);
    if (!Number.isFinite(budget) || budget < 0) errs.push('max_budget_usd must be a number');
  }
  return { errs, clean: { idea: b.idea, hours, paper_url: b.paper_url || '', git_url: b.git_url || '', scenes, model_preset: b.model_preset || '', max_budget_usd: budget } };
}

// ───────────────────────────── http plumbing ─────────────────────────────
function sendJSON(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
  res.end(body);
}

function authed(req, query) {
  if (!TOKEN) return true;
  const h = req.headers['authorization'] || '';
  if (h === `Bearer ${TOKEN}`) return true;
  if (query.get('token') === TOKEN) return true;
  return false;
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.pdf': 'application/pdf',
  '.md': 'text/markdown; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.log': 'text/plain; charset=utf-8',
  '.tex': 'text/plain; charset=utf-8',
};

function serveStatic(req, res, pathname) {
  let rel = pathname === '/' ? '/index.html' : pathname;
  const fp = path.normalize(path.join(PUBLIC, rel));
  if (!fp.startsWith(PUBLIC)) {
    res.writeHead(403); return res.end('forbidden');
  }
  fs.readFile(fp, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      return res.end('not found');
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(fp)] || 'application/octet-stream', 'Cache-Control': 'no-cache, no-store, must-revalidate' });
    res.end(data);
  });
}

function serveFile(res, absPath, downloadName) {
  fs.stat(absPath, (err, st) => {
    if (err || !st.isFile()) {
      res.writeHead(404, { 'Content-Type': 'text/plain' }); return res.end('not found');
    }
    const headers = { 'Content-Type': MIME[path.extname(absPath)] || 'application/octet-stream', 'Content-Length': st.size };
    if (downloadName) headers['Content-Disposition'] = `inline; filename="${downloadName}"`;
    res.writeHead(200, headers);
    fs.createReadStream(absPath).pipe(res);
  });
}

function readBody(req) {
  return new Promise((resolve) => {
    let data = '';
    req.on('data', (c) => {
      data += c;
      if (data.length > 1e6) req.destroy();
    });
    req.on('end', () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch {
        resolve(null);
      }
    });
  });
}

// ───────────────────────────── SSE stream ────────────────────────────────
function startStream(req, res, id) {
  if (!safeRun(id)) return sendJSON(res, 404, { error: 'no such run' });
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  res.write('retry: 3000\n\n');

  let lastTotal = 0;
  let lastStatus = null;
  let lastAlive = null;
  let pings = 0;

  const send = (event, obj) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(obj)}\n\n`);
  };

  // initial snapshot
  const snap = buildStatus(id);
  if (snap) send('status', snap);
  const ev0 = readEvents(id, 0);
  lastTotal = ev0.total;
  send('events', { events: ev0.events, total: ev0.total });

  const tick = () => {
    try {
      const ev = readEvents(id, lastTotal);
      if (ev.total > lastTotal) {
        send('events', { events: ev.events, total: ev.total });
        lastTotal = ev.total;
      }
      const led = readLedger(id);
      if (led) {
        const st = led.status || 'unknown';
        const wall = (led.budget && led.budget.wall_elapsed_s) || 0;
        if (st !== lastStatus) {
          send('status', buildStatus(id));
          lastStatus = st;
        } else {
          send('tick', { wall_elapsed_s: wall, status: st });
        }
      }
      const alive = tmuxAlive();
      if (alive !== lastAlive) {
        send('session', { alive });
        lastAlive = alive;
      }
      if (++pings % 10 === 0) res.write(': ping\n\n');
    } catch {
      /* keep the stream alive through transient read errors */
    }
  };
  const iv = setInterval(tick, 1500);
  req.on('close', () => clearInterval(iv));
}

// ───────────────────────────── router ────────────────────────────────────
const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const pathname = decodeURIComponent(u.pathname);
  const q = u.searchParams;

  if (!pathname.startsWith('/api/')) return serveStatic(req, res, pathname);

  if (!authed(req, q)) return sendJSON(res, 401, { error: 'unauthorized', hint: 'set Authorization: Bearer <GSS_TOKEN> or ?token=' });

  const parts = pathname.split('/').filter(Boolean); // ['api', ...]
  try {
    // GET /api/health
    if (req.method === 'GET' && parts[1] === 'health') return sendJSON(res, 200, { ok: true, root: ROOT, session: TMUX_SESSION });

    // GET /api/config
    if (req.method === 'GET' && parts[1] === 'config') {
      const r = run(PY, ['-c', "import yaml,json,sys;print(json.dumps(yaml.safe_load(open(sys.argv[1]))))", CONFIG_YAML]);
      if (!r.ok) return sendJSON(res, 500, { error: 'config read failed', detail: r.stderr });
      let cfg = {};
      try { cfg = JSON.parse(r.stdout); } catch {}
      return sendJSON(res, 200, {
        idea: cfg.idea || '',
        hours: cfg.hours || 24,
        paper_url: cfg.paper_url || '',
        git_url: cfg.git_url || '',
        scenes: cfg.scenes || [],
        dataset: cfg.dataset || '',
        model_preset: cfg.model_preset || 'all_opus',
        max_budget_usd: cfg.max_budget_usd || null,
        gpu_indices: cfg.gpu_indices || [],
        presets: PRESETS,
      });
    }

    // GET /api/gpus
    if (req.method === 'GET' && parts[1] === 'gpus') return sendJSON(res, 200, gpuInfo());

    // GET /api/session
    if (req.method === 'GET' && parts[1] === 'session') {
      const { session } = listRuns();
      return sendJSON(res, 200, session);
    }

    // GET /api/runs   (list)
    if (req.method === 'GET' && parts[1] === 'runs' && parts.length === 2) return sendJSON(res, 200, listRuns());

    // POST /api/launch
    if (req.method === 'POST' && parts[1] === 'launch') {
      if (tmuxAlive()) return sendJSON(res, 409, { error: 'session_exists', session: TMUX_SESSION, message: 'A run is already active (single-active-run). Stop or kill it first.' });
      const body = await readBody(req);
      if (!body) return sendJSON(res, 400, { error: 'bad json' });
      const { errs, clean } = validateLaunch(body);
      if (errs.length) return sendJSON(res, 400, { error: 'validation', detail: errs });
      const pc = patchConfig(clean);
      if (!pc.ok) return sendJSON(res, 500, { error: 'config patch failed', detail: pc.stderr });
      tmuxSend(buildResearchCmd(clean));
      return sendJSON(res, 202, { status: 'launching', session: TMUX_SESSION, message: 'Master bootstrapping in tmux; a new run_* will appear within ~60s.' });
    }

    // routes under /api/runs/:id/...
    if (parts[1] === 'runs' && parts.length >= 3) {
      const id = parts[2];
      if (!safeRun(id)) return sendJSON(res, 404, { error: 'no such run' });
      const sub = parts[3] || '';

      if (req.method === 'GET' && sub === '') return sendJSON(res, 200, buildStatus(id));
      if (req.method === 'GET' && sub === 'ledger') return sendJSON(res, 200, readLedger(id));
      if (req.method === 'GET' && sub === 'events') {
        const since = parseInt(q.get('since') || '0', 10) || 0;
        return sendJSON(res, 200, readEvents(id, since));
      }
      if (req.method === 'GET' && sub === 'stream') return startStream(req, res, id);
      if (req.method === 'GET' && sub === 'tmux') return sendJSON(res, 200, tmuxCapture(parseInt(q.get('lines') || '200', 10)));
      if (req.method === 'GET' && sub === 'pdf') {
        const rel = findPdf(id);
        if (!rel) return sendJSON(res, 404, { error: 'no pdf yet' });
        return serveFile(res, path.join(safeRun(id), rel), 'paper.pdf');
      }
      if (req.method === 'GET' && sub === 'file') {
        const rel = q.get('path') || '';
        const dir = safeRun(id);
        const abs = path.normalize(path.join(dir, rel));
        if (!abs.startsWith(dir + path.sep)) return sendJSON(res, 403, { error: 'path escapes run dir' });
        if (!['.md', '.png', '.jpg', '.jpeg', '.svg', '.json', '.txt', '.log', '.tex', '.pdf'].includes(path.extname(abs)))
          return sendJSON(res, 403, { error: 'extension not allowed' });
        return serveFile(res, abs);
      }

      // lifecycle mutations
      if (req.method === 'POST' && sub === 'stop') {
        const dir = safeRun(id);
        const r1 = ledgerCmd(dir, ['set-status', '--status', 'stopping', '--reason', 'user_requested (SplatLab UI)']);
        ledgerCmd(dir, ['log-event', '--type', 'stop_requested', '--json', JSON.stringify({ source: 'gss_ui' })]);
        if (!r1.ok) return sendJSON(res, 500, { error: 'set-status failed', detail: r1.stderr });
        return sendJSON(res, 200, { ok: true, status: 'stopping', message: 'Stop signal written. Master halts at its next iteration; in-flight scenes finish.' });
      }
      if (req.method === 'POST' && sub === 'resume') {
        if (tmuxAlive()) return sendJSON(res, 409, { error: 'session_live', message: 'A session is already live; the master will pick up. Use Live tab to watch.' });
        tmuxSend('/research-resume');
        return sendJSON(res, 202, { status: 'resuming', message: 'Started a session and sent /research-resume. It continues the most recent run from its ledger.' });
      }
      if (req.method === 'POST' && sub === 'kill') {
        const r = run('tmux', ['kill-session', '-t', TMUX_SESSION]);
        return sendJSON(res, 200, { ok: r.ok, message: r.ok ? 'tmux session killed. The ledger survives; Resume to continue.' : 'No live session to kill.' });
      }
    }

    return sendJSON(res, 404, { error: 'unknown route', path: pathname });
  } catch (e) {
    return sendJSON(res, 500, { error: 'server error', detail: String(e && e.stack || e) });
  }
});

server.listen(PORT, HOST, () => {
  const banner = `
  ┌────────────────────────────────────────────────────────────┐
  │  SplatLab  ·  autonomous GS research lab control panel   │
  └────────────────────────────────────────────────────────────┘
   url:      http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}
   bind:     ${HOST}:${PORT}
   project:  ${ROOT}
   session:  ${TMUX_SESSION} (tmux)
   auth:     ${TOKEN ? 'token required (GSS_TOKEN set)' : 'none (localhost)'}
`;
  process.stdout.write(banner + '\n');
  if (HOST === '0.0.0.0' && !TOKEN)
    process.stdout.write('  ⚠  Binding 0.0.0.0 with NO token — anyone on your LAN can control runs. Set GSS_TOKEN.\n\n');
});
