"""
Durable job state for the GS Research Autopilot web UI.

Provides:
- EventSink: queue.Queue-compatible shim that mirrors every event to a
  per-job events.jsonl file on disk (append-only, monotonic seq, UTC ts).
- Replay helpers so a browser connecting to /api/events/<job_id> can
  catch up on prior events before switching to live tailing.
- Rehydration: on Flask startup, scan workspace/ for job.json sidecars
  and rebuild in-memory job metadata so refreshing the UI always shows
  prior runs.

This is the "close the tab, come back later, see everything" layer.
The pipeline itself still lives in-process, so Flask restart → jobs
become 'orphaned' (process is gone, but logs remain fully replayable).
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# Event types that mark a job as finished. The SSE generator stops
# tailing after seeing one of these.
TERMINAL_EVENTS: frozenset[str] = frozenset({
    "pipeline_done", "pipeline_error", "pipeline_stopped",
    "done", "error", "discover_done",
    "cascade_benchmark_done",
})


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


class EventSink:
    """queue.Queue-compatible event sink that mirrors to disk.

    Drop-in replacement for the `event_queue` arg currently threaded
    through pipeline.py and splatify/discover.py. Every .put() assigns
    a monotonic sequence number, stamps a UTC iso timestamp, writes a
    JSON line to <workspace>/events.jsonl, and forwards the event to
    an in-memory Queue so live SSE subscribers still wake up.

    Thread-safe: the internal counter and file handle are guarded by
    a lock since pipeline/discover run in threads and call .put() from
    arbitrary workers.
    """

    def __init__(self, workspace: Path, start_seq: int = 0):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._path = self.workspace / "events.jsonl"
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._seq = start_seq

    @property
    def path(self) -> Path:
        return self._path

    @property
    def live_queue(self) -> queue.Queue:
        """The in-memory queue live SSE subscribers tail."""
        return self._queue

    def put(self, event: dict, block: bool = True, timeout: float | None = None) -> None:
        """Append to JSONL, then hand off to the live queue."""
        with self._lock:
            self._seq += 1
            enriched = {
                "seq": self._seq,
                "ts": _utc_iso(),
                **event,
            }
            try:
                with self._path.open("a") as f:
                    f.write(json.dumps(enriched, default=str) + "\n")
            except Exception:
                # Disk write should never block the pipeline; log to
                # stderr and still deliver to the live queue.
                pass
        self._queue.put(enriched, block=block, timeout=timeout)

    def put_nowait(self, event: dict) -> None:
        self.put(event, block=False)

    def get(self, block: bool = True, timeout: float | None = None):
        return self._queue.get(block=block, timeout=timeout)

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()


def replay_events(workspace: Path, after_seq: int = 0) -> Iterator[dict]:
    """Yield previously-recorded events from events.jsonl.

    Used by the SSE endpoint on cold connect to catch a client up to
    wherever they were before switching to live tailing. `after_seq`
    lets a client pass its highest-seen seq so a reconnect doesn't
    replay events it already has.
    """
    path = Path(workspace) / "events.jsonl"
    if not path.exists():
        return
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("seq", 0) > after_seq:
                yield event


def last_seq_in_log(workspace: Path) -> int:
    """Fast lookup of the highest seq already on disk (for rehydration)."""
    path = Path(workspace) / "events.jsonl"
    if not path.exists():
        return 0
    last = 0
    try:
        with path.open("r") as f:
            for line in f:
                try:
                    seq = json.loads(line).get("seq", 0)
                    if seq > last:
                        last = seq
                except Exception:
                    pass
    except Exception:
        pass
    return last


def write_sidecar(workspace: Path, meta: dict) -> None:
    """Write job.json next to events.jsonl."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "job.json").write_text(json.dumps(meta, indent=2, default=str))


def read_sidecar(workspace: Path) -> dict | None:
    path = Path(workspace) / "job.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def rehydrate_jobs(workspace_root: Path) -> list[dict]:
    """Scan workspace/ for all job.json sidecars.

    Returns a list of job metadata dicts sorted newest-first. Any job
    whose recorded PID is no longer alive is marked status='orphaned'
    — the events.jsonl is still fully replayable, but the producer
    thread is gone.
    """
    workspace_root = Path(workspace_root)
    if not workspace_root.exists():
        return []

    jobs: list[dict] = []
    for run_dir in workspace_root.iterdir():
        if not run_dir.is_dir():
            continue
        meta = read_sidecar(run_dir)
        if not meta:
            continue
        # Mark orphaned if the pipeline process is gone.
        if meta.get("status") == "running" and not _pid_alive(meta.get("pid")):
            meta["status"] = "orphaned"
        jobs.append(meta)

    jobs.sort(
        key=lambda m: m.get("started_at", ""),
        reverse=True,
    )
    return jobs


def make_sink_for_job(
    workspace: Path,
    job_id: str,
    run_id: str,
    kind: str,
    extra_meta: dict | None = None,
) -> EventSink:
    """Create an EventSink and write the initial job.json sidecar.

    `kind` is 'pipeline' or 'experiment' so rehydration knows which
    UI page to link to.
    """
    workspace = Path(workspace)
    meta = {
        "job_id": job_id,
        "run_id": run_id,
        "kind": kind,
        "workspace": str(workspace),
        "status": "running",
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
    }
    if extra_meta:
        meta.update(extra_meta)
    write_sidecar(workspace, meta)
    start_seq = last_seq_in_log(workspace)
    return EventSink(workspace, start_seq=start_seq)


def update_sidecar_status(
    workspace: Path,
    status: str,
    extra: dict | None = None,
) -> None:
    """Patch job.json with a new status (and optional extra fields)."""
    meta = read_sidecar(workspace) or {}
    meta["status"] = status
    if status in ("completed", "failed", "stopped", "orphaned"):
        meta["finished_at"] = datetime.now().isoformat()
    if extra:
        meta.update(extra)
    write_sidecar(workspace, meta)
