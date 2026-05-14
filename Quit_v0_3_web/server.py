from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent

# QUIT_DIR: resolved from env var, CLI --quit-dir, or default sibling directory
def _resolve_quit_dir() -> Path:
    if "QUIT_DIR" in os.environ:
        return Path(os.environ["QUIT_DIR"]).resolve()
    # check CLI before FastAPI/uvicorn consume argv
    for i, arg in enumerate(sys.argv):
        if arg == "--quit-dir" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).resolve()
    return (ROOT.parent / "Quit_v0_3").resolve()

import os
QUIT_DIR = _resolve_quit_dir()

# Python interpreter: prefer project venv, then active venv, then system
def _resolve_python() -> str:
    candidates = [
        QUIT_DIR.parent / ".venv" / "bin" / "python",
        ROOT.parent / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return sys.executable

PYTHON = _resolve_python()

app = FastAPI(title="QUIT Agent Web UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# job_id -> {proc, logs, status, returncode}
_jobs: dict[str, dict[str, Any]] = {}


class RunRequest(BaseModel):
    config: dict[str, Any]
    start_at: str = "PLAN"
    stop_after: str | None = None
    max_steps: int | None = None
    run_id: str | None = None


@app.get("/api/config")
def get_config():
    p = QUIT_DIR / "config.json"
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        # strip _comment keys
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    return {}


@app.get("/api/browse")
def browse_dir(path: str | None = None):
    """Return subdirectories at `path` for the model path picker.
    Defaults to QUIT_DIR.parent so users can navigate from the project root.
    """
    root = Path(path).resolve() if path else QUIT_DIR.parent.resolve()
    if not root.exists() or not root.is_dir():
        raise HTTPException(400, f"Not a directory: {root}")
    entries = []
    try:
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith('.'):
                entries.append({"name": child.name, "path": str(child)})
    except PermissionError:
        pass
    return {
        "current": str(root),
        "parent": str(root.parent) if root != root.parent else None,
        "entries": entries,
    }


@app.get("/api/runs/{run_id}/files")
def list_run_files(run_id: str):
    run_dir = (QUIT_DIR / "runs" / run_id).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise HTTPException(404, "Run not found")

    # important files to highlight in the UI
    HIGHLIGHT = {
        "paper_gene/main.pdf", "paper_gene/main.tex",
        "BuildSpec.json", "ResearchBrief.json", "IdeaDecision.json",
        "EXPERIMENT_LOG.md", "CodeRunReport.json", "ExperimentAudit.json",
    }

    def _tree(path: Path, rel: str) -> list:
        items = []
        try:
            for child in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name)):
                child_rel = f"{rel}/{child.name}" if rel else child.name
                if child.is_dir():
                    items.append({
                        "name": child.name, "rel": child_rel, "type": "dir",
                        "children": _tree(child, child_rel),
                    })
                else:
                    items.append({
                        "name": child.name, "rel": child_rel, "type": "file",
                        "ext": child.suffix.lower(),
                        "size": child.stat().st_size,
                        "highlight": child_rel in HIGHLIGHT,
                    })
        except PermissionError:
            pass
        return items

    return {"run_id": run_id, "files": _tree(run_dir, "")}


@app.get("/api/runs/{run_id}/file")
def get_run_file(run_id: str, path: str = Query(...)):
    run_dir = (QUIT_DIR / "runs" / run_id).resolve()
    target  = (run_dir / path).resolve()
    # security: must stay inside run_dir
    if not str(target).startswith(str(run_dir)):
        raise HTTPException(403, "Access denied")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File not found")

    if target.suffix.lower() == ".pdf":
        return FileResponse(str(target), media_type="application/pdf",
                            headers={"Content-Disposition": "inline"})

    if target.stat().st_size > 2 * 1024 * 1024:
        return Response("(file too large to preview — > 2 MB)", media_type="text/plain")

    return Response(target.read_text(errors="replace"), media_type="text/plain; charset=utf-8")


@app.get("/api/runs")
def list_runs():
    data_dir = QUIT_DIR / "runs"
    if not data_dir.exists():
        return []
    items = []
    for d in sorted(data_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir():
            items.append(d.name)
        if len(items) >= 30:
            break
    return items


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. override wins on conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        elif val is not None:
            result[key] = val
    return result


@app.post("/api/run")
async def start_run(req: RunRequest):
    job_id = uuid.uuid4().hex[:12]

    # Base: config.json in QUIT_DIR; UI params are merged on top (config.json never modified)
    base_config: dict = {}
    config_path = QUIT_DIR / "config.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            base_config = {k: v for k, v in raw.items() if not k.startswith("_")}
        except Exception:
            pass
    merged = _deep_merge(base_config, req.config)

    tmp_name = f".ui_config_{job_id}.json"
    tmp_path = QUIT_DIR / tmp_name
    tmp_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    cmd = [PYTHON, "-m", "quit_agent.main", "--config", tmp_name, "--start-at", req.start_at]
    if req.stop_after:
        cmd += ["--stop-after", req.stop_after]
    if req.max_steps is not None:
        cmd += ["--max-steps", str(req.max_steps)]
    if req.run_id:
        cmd += ["--run-id", req.run_id]

    env = os.environ.copy()
    src_dir = str(QUIT_DIR / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_dir if not existing else f"{src_dir}:{existing}"

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(QUIT_DIR),
        env=env,
    )
    _jobs[job_id] = {"proc": proc, "logs": [], "status": "running"}
    asyncio.create_task(_drain(job_id, tmp_path))
    return {"job_id": job_id}


async def _drain(job_id: str, tmp_path: Path) -> None:
    job = _jobs[job_id]
    async for raw in job["proc"].stdout:
        job["logs"].append(raw.decode("utf-8", errors="replace").rstrip())
    await job["proc"].wait()
    rc = job["proc"].returncode
    job["returncode"] = rc
    job["status"] = "done" if rc == 0 else "error"
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "job not found")
    j = _jobs[job_id]
    return {"status": j["status"], "log_count": len(j["logs"]), "returncode": j.get("returncode")}


@app.get("/api/stream/{job_id}")
async def stream_logs(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "job not found")
    job = _jobs[job_id]

    async def generate():
        sent = 0
        while True:
            logs = job["logs"]
            while sent < len(logs):
                yield f"data: {json.dumps(logs[sent])}\n\n"
                sent += 1
            if job["status"] != "running" and sent >= len(logs):
                yield f"data: [DONE:{job['status']}:{job.get('returncode', '')}]\n\n"
                return
            await asyncio.sleep(0.15)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "job not found")
    job = _jobs[job_id]
    try:
        job["proc"].kill()
    except Exception:
        pass
    job["status"] = "cancelled"
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="QUIT Agent Web UI")
    parser.add_argument("--host",     default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",     default=7860, type=int, help="Port (default: 7860)")
    parser.add_argument("--quit-dir", default=None, help="Path to Quit_v0_3 project directory")
    args, _ = parser.parse_known_args()

    if args.quit_dir:
        QUIT_DIR = Path(args.quit_dir).resolve()

    print(f"QUIT_DIR : {QUIT_DIR}")
    print(f"Python   : {PYTHON}")
    print(f"UI       : http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
