"""
One FastAPI app, two surfaces:
  1. WebSocket /ws  -- the voice pipeline (unchanged logic from the original
     raw-websockets version, only the transport calls changed).
  2. REST routes    -- call log lookup, flagged-call listing, LLM-based
     triage, and a metrics summary for the dashboard.

Why one app instead of two processes: a separate REST process reading the
same SQLite file a separate pipeline process writes to risks "database is
locked" under concurrency. One process, one event loop, no cross-process
file contention.

WebSocket protocol (unchanged):
  - Client sends BINARY frames: raw PCM16LE mono audio @ 16kHz, any chunk size.
  - Server buffers + VADs it internally. When it detects end-of-turn:
      1. sends back one TEXT frame: JSON with transcript, reply, and
         per-stage latencies
      2. sends back one BINARY frame: the response audio (WAV bytes)
  - Then keeps listening for the next turn, same connection.
"""
import asyncio
import json
import time
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

import db
from vad import EndpointDetector, WINDOW_SIZE
from stt import transcribe
from llm import respond, triage_call
from tts import synthesize
from latency import log_row

BYTES_PER_SAMPLE = 2  # int16

app = FastAPI(title="voice-agent")
db.init_db()


# ---------------------------------------------------------------- pipeline --

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("client connected")
    detector = EndpointDetector(min_silence_ms=600)
    utterance_buf = []   # float32 chunks collected while the user is speaking
    pcm_leftover = b""   # bytes that don't yet fill a full VAD window
    history = []          # running conversation, for multi-turn context

    try:
        while True:
            message = await websocket.receive_bytes()

            pcm_leftover += message
            window_bytes = WINDOW_SIZE * BYTES_PER_SAMPLE

            while len(pcm_leftover) >= window_bytes:
                frame_bytes = pcm_leftover[:window_bytes]
                pcm_leftover = pcm_leftover[window_bytes:]

                int16 = np.frombuffer(frame_bytes, dtype=np.int16)
                float32 = int16.astype(np.float32) / 32768.0

                vad_start = time.perf_counter()
                event = detector.process_chunk(float32)
                vad_ms = (time.perf_counter() - vad_start) * 1000

                if detector.speaking or event == "end":
                    utterance_buf.append(float32)

                if event == "end" and utterance_buf:
                    audio = np.concatenate(utterance_buf)
                    utterance_buf = []
                    asyncio.create_task(run_pipeline(websocket, audio, history, vad_ms))
    except WebSocketDisconnect:
        print("client disconnected")


async def run_pipeline(websocket: WebSocket, audio: np.ndarray, history: list, vad_ms: float):
    t0 = time.perf_counter()

    t = time.perf_counter()
    transcript = transcribe(audio)
    stt_ms = (time.perf_counter() - t) * 1000

    if not transcript:
        return  # noise / silence picked up as a false endpoint -- skip it

    t = time.perf_counter()
    reply_text, ttft = respond(transcript, history)
    llm_total_ms = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    audio_bytes = synthesize(reply_text)
    tts_ms = (time.perf_counter() - t) * 1000

    total_ms = (time.perf_counter() - t0) * 1000

    latency = {
        "vad": round(vad_ms, 1),
        "stt": round(stt_ms, 1),
        "llm_ttft": round(ttft * 1000, 1),
        "llm_total": round(llm_total_ms, 1),
        "tts": round(tts_ms, 1),
        "total": round(total_ms, 1),
    }
    payload = {"transcript": transcript, "reply": reply_text, "latency_ms": latency}
    print(payload)

    await websocket.send_text(json.dumps(payload))
    await websocket.send_bytes(audio_bytes)

    log_row({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "vad_ms": latency["vad"],
        "stt_ms": latency["stt"],
        "llm_ttft_ms": latency["llm_ttft"],
        "llm_total_ms": latency["llm_total"],
        "tts_ms": latency["tts"],
        "total_ms": latency["total"],
    })

    # Additive: structured log for the REST/triage/dashboard surface.
    # latency.py's CSV write above is untouched -- this is a second,
    # independent record, not a replacement.
    db.insert_call(
        call_id=str(uuid.uuid4()),
        transcript=transcript,
        reply=reply_text,
        latency=latency,
    )


# -------------------------------------------------------------------- REST --

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/calls/flagged")
async def flagged_calls(since: float | None = None):
    return db.get_flagged_calls(since=since)


@app.get("/calls/{call_id}")
async def get_call(call_id: str):
    call = db.get_call(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="call not found")
    return call


@app.get("/metrics")
async def metrics():
    return db.get_metrics()


@app.post("/triage/{call_id}")
async def triage(call_id: str):
    call = db.get_call(call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="call not found")
    with db._connect() as conn:  # noqa: SLF001 -- internal helper, same module family
        threshold = db._flag_threshold(conn)
    try:
        summary = triage_call(call, threshold_ms=threshold)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"call_id": call_id, "threshold_ms": threshold, "summary": summary}


@app.get("/dashboard")
async def dashboard():
    return FileResponse("static/dashboard.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
