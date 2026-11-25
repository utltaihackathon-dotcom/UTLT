import logging
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from uuid import UUID
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utlt_brain import Assistant, parse_readings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("utlt-ai-api")

AI_API_URL = os.getenv(
    "UTLT_AI_API_URL",
    "https://raujzsawwpmixwlcgcgs.supabase.co/functions/v1/public-ai-api",
).strip()
AI_API_KEY = os.getenv("UTLT_AI_API_KEY", "").strip()
AI_TIMEOUT = float(os.getenv("UTLT_AI_TIMEOUT_SECONDS", "45"))
WATCHDOG_INTERVAL = max(60, int(os.getenv("UTLT_WATCHDOG_INTERVAL_SECONDS", "300")))

app = FastAPI(title="UTLT.ai Assistant API", version="2.0.0")

allowed_origins = [
    item.strip()
    for item in os.getenv("FRONTEND_ORIGINS", "*").split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

MAX_SESSIONS = 1000
sessions: OrderedDict[str, Assistant] = OrderedDict()
histories: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
sessions_lock = threading.RLock()
watchdog_lock = threading.Lock()
watchdog_state: dict[str, Any] = {
    "enabled": os.getenv("UTLT_WATCHDOG_ENABLED", "true").lower() not in {"0", "false", "no"},
    "last_checked": None,
    "last_ok": None,
    "last_http_status": None,
    "last_latency_ms": None,
    "last_error": None,
    "checks": 0,
}


class ChatRequest(BaseModel):
    session_id: UUID
    message: str = Field(min_length=1, max_length=2000)
    readings: list[dict[str, Any]] = Field(default_factory=list, max_length=5000)
    goal: float = Field(default=350, gt=0, le=100000)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_reply(payload: Any) -> str:
    """Accept common OpenAI-compatible and Supabase edge-function response shapes."""
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            return "\n".join(x for x in parts if x).strip()
    for key in ("reply", "response", "output", "content", "message"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict) and isinstance(value.get("content"), str):
            return value["content"].strip()
    return ""


def _call_utlt_ai(messages: list[dict[str, str]]) -> tuple[str, int, float]:
    if not AI_API_KEY:
        raise RuntimeError("UTLT_AI_API_KEY is not configured on the server.")
    started = time.perf_counter()
    try:
        response = httpx.post(
            AI_API_URL,
            headers={
                "Authorization": f"Bearer {AI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"messages": messages},
            timeout=AI_TIMEOUT,
        )
    except httpx.TimeoutException as exc:
        raise RuntimeError("The UTLT AI API timed out.") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError("Could not connect to the UTLT AI API.") from exc

    elapsed = round((time.perf_counter() - started) * 1000, 1)
    if response.status_code < 200 or response.status_code >= 300:
        # Do not return/log the upstream response body: it may contain sensitive data.
        raise UpstreamStatusError(response.status_code)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("The UTLT AI API returned invalid JSON.") from exc
    reply = _extract_reply(payload)
    if not reply:
        raise RuntimeError("The UTLT AI API response did not contain a text reply.")
    return reply, response.status_code, elapsed


class UpstreamStatusError(RuntimeError):
    def __init__(self, status_code: int):
        super().__init__(f"UTLT AI API returned HTTP {status_code}.")
        self.status_code = status_code


def _watchdog_check() -> None:
    checked_at = _utc_now()
    started = time.perf_counter()
    try:
        _reply, status_code, latency = _call_utlt_ai([
            {"role": "user", "content": "Reply with exactly: UTLT_AI_OK"}
        ])
        state = {
            "last_checked": checked_at,
            "last_ok": True,
            "last_http_status": status_code,
            "last_latency_ms": latency,
            "last_error": None,
        }
    except UpstreamStatusError as exc:
        state = {
            "last_checked": checked_at, "last_ok": False,
            "last_http_status": exc.status_code,
            "last_latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "last_error": str(exc),
        }
    except Exception as exc:
        state = {
            "last_checked": checked_at, "last_ok": False,
            "last_http_status": None,
            "last_latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "last_error": str(exc)[:180],
        }
    with watchdog_lock:
        watchdog_state.update(state)
        watchdog_state["checks"] += 1
    if not state["last_ok"]:
        logger.warning("Watchdog test failed: %s", state["last_error"])


def _watchdog_loop() -> None:
    # First test immediately, then repeat while this Render process is running.
    while True:
        _watchdog_check()
        time.sleep(WATCHDOG_INTERVAL)


@app.on_event("startup")
def start_watchdog() -> None:
    if watchdog_state["enabled"]:
        thread = threading.Thread(target=_watchdog_loop, name="utlt-ai-watchdog", daemon=True)
        thread.start()
        logger.info("UTLT AI watchdog started; interval=%ss", WATCHDOG_INTERVAL)


@app.get("/")
def root():
    return {"status": "online", "service": "UTLT.ai Assistant API", "version": "2.0.0"}


@app.get("/health")
def health():
    with watchdog_lock:
        last_ok = watchdog_state["last_ok"]
    configured = bool(AI_API_KEY)
    return {
        "status": "healthy" if configured and last_ok is not False else "degraded" if not configured or last_ok is False else "healthy",
        "service": "UTLT.ai",
        "ai_api_configured": configured,
        "watchdog_enabled": watchdog_state["enabled"],
        "watchdog_last_ok": last_ok,
    }


@app.get("/diagnostics")
def diagnostics():
    with watchdog_lock:
        state = dict(watchdog_state)
    return {
        "service": "UTLT.ai",
        "checked_at": _utc_now(),
        "backend": "online",
        "ai_api": {
            "configured": bool(AI_API_KEY),
            "url_configured": bool(AI_API_URL),
            "last_ok": state["last_ok"],
            "last_checked": state["last_checked"],
            "last_http_status": state["last_http_status"],
            "last_latency_ms": state["last_latency_ms"],
            "last_error": state["last_error"],
        },
        "watchdog": {
            "enabled": state["enabled"],
            "interval_seconds": WATCHDOG_INTERVAL,
            "checks_completed": state["checks"],
        },
    }


@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    if not AI_API_KEY:
        raise HTTPException(status_code=503, detail="UTLT AI is not configured yet. Add UTLT_AI_API_KEY in Render.")

    try:
        parsed_readings = parse_readings(request.readings)
        session_key = str(request.session_id)
        with sessions_lock:
            assistant = sessions.get(session_key)
            if assistant is None:
                assistant = Assistant(readings=parsed_readings or None, goal=request.goal)
                sessions[session_key] = assistant
            else:
                sessions.move_to_end(session_key)
                if parsed_readings:
                    assistant.readings = parsed_readings
                    assistant.source = "file"
                elif assistant.source == "file":
                    assistant = Assistant(readings=None, goal=request.goal)
                    sessions[session_key] = assistant
            assistant.goal = request.goal
            assistant.refresh()
            domain_context = assistant.reply(message)
            source = assistant.source

            history = histories.get(session_key, [])
            histories.move_to_end(session_key) if session_key in histories else None
            if domain_context.startswith("I didn't quite get that."):
                domain_context = "No matching water-analysis insight was found for this question."
            system_prompt = (
                "You are UTLT.ai, a helpful, natural, accurate household water-saving assistant. "
                "Answer the user's actual question directly and conversationally. "
                "Use the supplied UTLT water-analysis context when relevant. Never invent readings, "
                "measurements, device connectivity, leak detections, or predictions. If the context "
                "says demo data, clearly call it demo data. If there is not enough data, say so. "
                "Do not claim to control devices, book LPG, or perform actions that are not connected. "
                "Keep answers concise, use litres (L), and use India/IST context.\n\n"
                f"Current UTLT data source: {source}.\n"
                f"Existing UTLT water-brain analysis for this message:\n{domain_context}"
            )
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(history[-10:])
            messages.append({"role": "user", "content": message})

        reply, status_code, latency = _call_utlt_ai(messages)

        with sessions_lock:
            history = histories.get(session_key, [])
            history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
            histories[session_key] = history[-20:]
            histories.move_to_end(session_key)
            while len(sessions) > MAX_SESSIONS:
                old_key, _ = sessions.popitem(last=False)
                histories.pop(old_key, None)
            while len(histories) > MAX_SESSIONS:
                histories.popitem(last=False)

        return {"reply": reply, "source": source, "provider": "utlt-ai-api", "latency_ms": latency}
    except HTTPException:
        raise
    except UpstreamStatusError as error:
        with watchdog_lock:
            watchdog_state.update({"last_checked": _utc_now(), "last_ok": False,
                                   "last_http_status": error.status_code, "last_error": str(error)})
        raise HTTPException(status_code=502, detail="Your UTLT AI API returned an error. Open problem.html for diagnostics.") from error
    except RuntimeError as error:
        with watchdog_lock:
            watchdog_state.update({"last_checked": _utc_now(), "last_ok": False, "last_error": str(error)[:180]})
        raise HTTPException(status_code=502, detail="UTLT.ai could not reach its AI API. Open problem.html for diagnostics.") from error
    except Exception as error:
        logger.exception("UTLT.ai request failed")
        raise HTTPException(status_code=500, detail="UTLT.ai could not process that message.") from error
