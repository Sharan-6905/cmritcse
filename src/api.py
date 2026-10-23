"""FastAPI app: chat with the waste agent, optionally attaching an image."""
import collections
import json
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import agent, classifier

app = FastAPI(title="Waste Agent")
STATIC = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")
ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}
MAX_BYTES = 8 * 1024 * 1024
MAX_SESSIONS = 50
sessions: dict[str, agent.ChatSession] = {}

# Public demo protection: every chat request spends Gemini quota, so cap each visitor and the whole site.
RATE_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "40"))
DAILY_CAP = int(os.getenv("DAILY_REQUEST_CAP", "250"))
_hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
_day = {"date": "", "n": 0}
_rate_lock = threading.Lock()


def _limit(request: Request) -> None:
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else "?")
    now, today = time.time(), time.strftime("%Y-%m-%d")
    with _rate_lock:
        hits = _hits[ip]
        while hits and now - hits[0] > 3600:
            hits.popleft()
        if _day["date"] != today:
            _day.update(date=today, n=0)
        if _day["n"] >= DAILY_CAP:
            raise HTTPException(429, "This demo has reached its daily limit. Please try again tomorrow.")
        if len(hits) >= RATE_PER_HOUR:
            raise HTTPException(429, "You've reached the hourly limit for this demo. Please try again a little later.")
        hits.append(now)
        _day["n"] += 1


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


def _save(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, "Upload a .jpg, .png or .webp image")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(upload.file, tmp)
    if Path(tmp.name).stat().st_size > MAX_BYTES:
        Path(tmp.name).unlink()
        raise HTTPException(413, "Image larger than 8 MB")
    return tmp.name


@app.post("/api/classify")
def classify_only(file: UploadFile = File(...)):
    path = _save(file)
    try:
        return classifier.classify(path)
    finally:
        Path(path).unlink(missing_ok=True)


def _session(session_id: str) -> tuple[str, agent.ChatSession]:
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.pop(next(iter(sessions)))
        session_id = uuid.uuid4().hex
        try:
            sessions[session_id] = agent.ChatSession()
        except KeyError:
            raise HTTPException(500, "GEMINI_API_KEY is not set (see .env.example)")
    return session_id, sessions[session_id]


@app.post("/api/chat")
def chat(request: Request, message: str = Form(""), session_id: str = Form(""), file: UploadFile | None = File(None)):
    if not message.strip() and file is None:
        raise HTTPException(400, "Send a message or an image")
    _limit(request)
    session_id, session = _session(session_id)
    path = _save(file) if file is not None else None
    try:
        result = session.send(message[:500], path)
    except Exception as e:
        raise HTTPException(502, f"The AI service failed: {str(e)[:200]}")
    return {"session_id": session_id, **result}


@app.post("/api/chat/stream")
def chat_stream(request: Request, message: str = Form(""), session_id: str = Form(""), file: UploadFile | None = File(None)):
    """Same as /api/chat, but streams newline-delimited JSON: step events while the agent works, then 'done'."""
    if not message.strip() and file is None:
        raise HTTPException(400, "Send a message or an image")
    _limit(request)
    session_id, session = _session(session_id)
    path = _save(file) if file is not None else None
    events: queue.Queue = queue.Queue()
    session.on_event = events.put

    def work():
        try:
            events.put({"type": "done", "session_id": session_id, **session.send(message[:500], path)})
        except Exception as e:
            events.put({"type": "error", "detail": f"The AI service failed: {str(e)[:200]}"})
        finally:
            session.on_event = None
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (event := events.get()) is not None:
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


WEIGHTS = Path(__file__).resolve().parent.parent / "weights"


@app.get("/insights")
def insights_page():
    return FileResponse(STATIC / "insights.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/insights")
def insights():
    def load(name):
        f = WEIGHTS / name
        return json.loads(f.read_text()) if f.exists() else None
    local = load("metrics.json") or {}
    return {"local": {"test_acc": local.get("test_acc"), "best_val_acc": local.get("best_val_acc"), "split": local.get("split"),
                      "per_class": {c: [v["precision"], v["recall"], v["support"]] for c, v in (local.get("per_class") or {}).items()
                                    if c in ("cardboard", "glass", "metal", "paper", "plastic", "trash")},
                      "history": local.get("history", [])},
            "colab": load("colab_results.json"), "real_world": load("real_world_summary.json")}


@app.get("/api/confusion-matrix.png")
def confusion_matrix():
    return FileResponse(WEIGHTS / "confusion_matrix.png")
