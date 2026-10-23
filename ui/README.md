# SplatLab UI

A zero-dependency web control panel for the autonomous Gaussian-Splatting research
lab. It mirrors the full `/research` slash-command family from the browser:
**launch · observe · stop · resume · kill**.

There is no separate engine. The intelligence *is* Claude Code running the
`/research` master prompt inside a persistent `tmux` session. All durable state
lives on disk in `workspace/run_<ts>/ledger.json` + `events.jsonl`. This panel
reads that state and drives the same lifecycle commands — so refresh / laptop-close
/ server-restart all reconnect to the same live job. The backend keeps **no run
state of its own**.

## Run it

```bash
bash start-ui.sh                                   # http://localhost:8420
PORT=9000 bash start-ui.sh                          # custom port
HOST=0.0.0.0 GSS_TOKEN=secret bash start-ui.sh      # LAN / ssh-tunnel access, token-gated
```

Requirements: `node` (v18+), `tmux`, `python3` + `pyyaml`, and the `claude` CLI on
PATH (only needed to *launch* runs; observing works without it).

## How the controls map to commands

| UI control | Command | Effect |
|---|---|---|
| **Launch Run** | `/research "idea" --hours N` | Seeds idea+flags into `config.yaml`, starts `claude` in the `splatlab` tmux session, sends `/research`. |
| **Stop** | `/research-stop` | Sets `status: stopping` in the ledger; master halts gracefully. |
| **Resume** | `/research-resume` | Starts a session, sends `/research-resume`; continues most recent run from its ledger. |
| **Kill** | `tmux kill-session` | Hard-stops the session; ledger survives. |
| **Overview** | `/research-status` | Status, thesis, ideas, best result, milestones, budget. |

## API

All endpoints are under `/api`. When `GSS_TOKEN` is set they require
`Authorization: Bearer <token>` (or `?token=`).

| Method · Path | Purpose |
|---|---|
| `GET /api/config` | config.yaml defaults for the launch form |
| `GET /api/runs` | list all runs + tmux session state |
| `GET /api/runs/:id` | full status object (the `/research-status` view) |
| `GET /api/runs/:id/ledger` | raw ledger JSON |
| `GET /api/runs/:id/events?since=N` | events.jsonl lines from index N |
| `GET /api/runs/:id/stream` | SSE: live events + status + session |
| `GET /api/runs/:id/tmux` | tmux pane tail (master console) |
| `GET /api/runs/:id/pdf` | compiled paper PDF |
| `GET /api/runs/:id/file?path=` | whitelisted artifact (figures, reports) |
| `GET /api/gpus` | best-effort GPU strip |
| `POST /api/launch` | start a run (409 if one is already active) |
| `POST /api/runs/:id/stop` · `/resume` · `/kill` | lifecycle controls |

## Design

- **Single active run** — one run at a time in the fixed `splatlab`
  tmux session (matches `scripts/launch.sh`). History of all runs is browsable.
- **Safety** — free-form idea text is written into `config.yaml` (the documented
  `/research` positional fallback), never interpolated into a shell/tmux command.
  Only sanitized flags (numeric hours, validated http(s) URLs, scene whitelist)
  reach the command line. Artifact serving is path-traversal-guarded and
  extension-whitelisted.

Files:

```
ui/server.js              zero-dep Node http backend (API + static + SSE)
ui/public/index.html      dashboard shell
ui/public/styles.css      dark theme
ui/public/app.js          fetch + SSE client
ui/scripts/patch_config.py   seeds config.yaml from a launch (stdin JSON)
ui/scripts/tmux_start.sh     tmux orchestration (new-session → claude → send-keys)
```
