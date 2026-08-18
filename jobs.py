"""Simple single-worker job queue so only one GPU generation runs at a time.

Jobs are persisted to data/jobs.json so history survives server restarts.
Params blobs are kept small (no raw image bytes) - uploaded reference images
are saved to disk and only their paths are stored in the job record.
"""

import json
import threading
import queue
import time
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
JOBS_FILE = DATA_DIR / "jobs.json"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

_lock = threading.Lock()
_queue = queue.Queue()
_workers = {}


def _load():
    if JOBS_FILE.exists():
        try:
            return json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


_jobs = _load()


def _save():
    JOBS_FILE.write_text(
        json.dumps(_jobs, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def create_job(kind, params):
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "kind": kind,
        "params": params,
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "result": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }
    with _lock:
        _jobs[job_id] = job
        _save()
    _queue.put(job_id)
    return job


def update_job(job_id, **fields):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)
            _jobs[job_id]["updated_at"] = time.time()
            _save()


def get_job(job_id):
    with _lock:
        return _jobs.get(job_id)


def list_jobs(limit=50):
    with _lock:
        items = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)
    return items[:limit]


def cancel_job(job_id):
    with _lock:
        job = _jobs.get(job_id)
        if job and job["status"] == "queued":
            job["status"] = "cancelled"
            job["message"] = "Cancelled before it started"
            _save()
            return True
        return False


def register_worker(kind, fn):
    """fn(job: dict) must call update_job(...) itself, including on success."""
    _workers[kind] = fn


def _worker_loop():
    while True:
        job_id = _queue.get()
        job = get_job(job_id)
        if not job or job["status"] == "cancelled":
            continue
        fn = _workers.get(job["kind"])
        if not fn:
            update_job(
                job_id,
                status="error",
                message=f"No worker registered for '{job['kind']}'",
            )
            continue
        update_job(job_id, status="running", message="Starting generation...")
        try:
            fn(job)
        except Exception as exc:  # keep the queue alive even if a job blows up
            update_job(job_id, status="error", message=str(exc))


def start_worker():
    thread = threading.Thread(target=_worker_loop, daemon=True, name="gen-worker")
    thread.start()
    return thread
