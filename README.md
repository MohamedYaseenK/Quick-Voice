# QuickVoice - VA

A real-time voice agent you can hold a spoken conversation with over a live WebSocket connection — you talk, it detects when you've actually stopped (not just paused), transcribes your turn, generates a reply, and speaks it back, entirely on CPU with no GPU involved.

It also logs every call as a structured, queryable record, flags abnormally slow calls, and can auto-generate a root-cause summary for a flagged call via an LLM — the operational layer a support/FDE team would actually run on top of a voice platform, not just the agent itself.

## Demo

<!-- Add your terminal output / screenshot here -->
<!-- Example: ![Demo output](./assets/demo_output.png) -->

## How the voice pipeline works

A client streams raw microphone audio to the server continuously, in small chunks, over a WebSocket. Each turn goes through five stages:

1. **Streaming audio in** — PCM16 audio sent over the WebSocket as it's recorded.
2. **Turn detection (VAD)** — Silero VAD scores each audio window for speech; a small state machine declares a turn "over" after ~600ms of continuous silence.
3. **Speech-to-text** — the buffered utterance is transcribed in one shot with `faster-whisper` (`base` model, int8).
4. **Reply generation** — the transcript goes to Gemini, streamed so time-to-first-token (TTFT) can be measured separately from total generation time.
5. **Text-to-speech** — the reply is converted to audio with `pyttsx3`, offline and CPU-only.

The reply audio goes back over the same connection. Every stage's timing is logged two ways: a flat CSV (`logs/latency_log.csv`) and a structured SQLite record (`logs/calls.db`) used by the observability layer below.

## The observability / triage layer

This is the part that makes the project relevant beyond "a working voice agent" — it's what you'd build to actually *operate* one:

- **Structured call log (SQLite).** Every call is stored with its transcript, reply, and full per-stage latency breakdown.
- **Automatic anomaly flagging.** A call is flagged if its total latency exceeds the 95th percentile of recent calls. Below 20 logged calls (not enough history for a percentile to mean anything), it falls back to a fixed 6-second threshold instead.
- **LLM-based triage (`POST /triage/{call_id}`).** Feeds a flagged call's latency breakdown to Gemini and gets back a plain-English root-cause summary — e.g. *"LLM time-to-first-token was 3.6s, roughly double the recent baseline; VAD and STT were both normal."* The prompt explicitly allows "no anomaly found" as a valid answer, rather than forcing a diagnosis.
- **Built-in dashboard (`GET /dashboard`).** Call count, p50/p95/p99 latency, flagged rate, and a table of flagged calls — no Grafana dependency (see "What's optional and unverified" below for why).
- **n8n automation (optional, Tier 2).** A workflow that polls `/calls/flagged` every 60s and runs triage automatically on anything new.

## Architecture

```
voice-agent/
├── server.py                 # ONE FastAPI app: /ws (the pipeline) + REST (calls/triage/metrics/dashboard)
├── client.py                  # Test harness — streams mic audio, plays back replies
├── db.py                       # SQLite call log: insert, lookup, percentile-based flagging, metrics
├── vad.py                       # Silero VAD wrapper — turn/endpoint detection
├── stt.py                        # faster-whisper wrapper — audio → text
├── llm.py                         # Gemini wrapper — chat reply (streamed, for TTFT) + triage_call()
├── tts.py                          # pyttsx3 wrapper — text → audio
├── latency.py                      # CSV logging helper (unchanged, still runs alongside db.py)
├── static/
│   └── dashboard.html               # Self-contained dashboard, polls /metrics + /calls/flagged
├── requirements.txt                  # Full dev environment (includes client.py's mic/playback deps)
├── requirements-server.txt            # Server-only deps, used by the Docker image
├── Dockerfile                          # UNVERIFIED — see below
├── docker-compose.yml                   # UNVERIFIED — see below
├── n8n/
│   └── triage_workflow.json              # UNVERIFIED — see below
├── template.sh
├── .env                                    # GEMINI_API_KEY (not committed)
└── logs/
    ├── latency_log.csv                       # Flat per-turn latency table
    └── calls.db                               # Structured call log (SQLite, WAL mode)
```

## Setup

```bash
chmod +x template.sh
./template.sh
```

Creates a virtual environment, installs `requirements.txt`, and generates `.env` for your Gemini key (https://aistudio.google.com/apikey).

## Running it

Two terminals, both with the virtual environment activated:

```bash
# Terminal 1 — start the server (now FastAPI/uvicorn on port 8000, not the old raw-websockets port 8765)
python server.py

# Terminal 2 — start talking
python client.py
```

Then open `http://localhost:8000/dashboard` to see call volume and latency live, and `http://localhost:8000/docs` for FastAPI's interactive API explorer.

## What's verified, and what isn't — read this before demoing

This matters for an honest project writeup, so it's stated plainly rather than left for you to discover:

**Tested and passing** (unit tests against `db.py`, and FastAPI `TestClient` integration tests against `server.py`'s full REST surface, run with the heavy ML stages stubbed out since this was built in a sandbox with no audio hardware and no GPU):
- Cold-start fallback threshold and the switch to real p95 after 20+ calls
- Flagged-call filtering, metrics aggregation, 404 handling on unknown call IDs
- `/triage/{id}` failing cleanly (HTTP 503, not a crash) when `GEMINI_API_KEY` is missing

**Two real bugs this testing caught before you'd have hit them live:**
1. Route ordering — `/calls/{call_id}` was shadowing `/calls/flagged`, so the flagged-list endpoint always 404'd. Fixed by declaring the static route first.
2. `llm.py` crashed at **import time**, not call time, if the API key was missing — meaning the whole server would fail to even start. Fixed with lazy model initialization.

**Not runnable in the environment this was built in, so genuinely untested — verify locally before relying on them:**
- The full audio pipeline end to end (no microphone, no PortAudio in that sandbox)
- The Docker build and `docker-compose up` (no Docker daemon available there)
- The n8n workflow import (`n8n/triage_workflow.json` is valid JSON and matches n8n's documented node schema, but was never imported into a running n8n instance)

**Deliberately not included:** Grafana. It can't read SQLite without an unsigned community plugin (`frser-sqlite-datasource`), which is a reliability risk that couldn't be tested without Docker. The built-in `/dashboard` page covers the same metrics without that dependency — Grafana is left commented out in `docker-compose.yml` for anyone who wants to test and enable it later.

**Known technical debt:** `llm.py` uses `google.generativeai`, which Google has fully deprecated in favor of `google.genai`. It still works, but a future pass should migrate it.

## Latency results (from live runs)

| Stage | Typical time |
|---|---|
| VAD | < 1 ms |
| STT (faster-whisper, int8) | ~0.5–1.8 s |
| LLM TTFT (Gemini) | ~2–3.6 s |
| LLM total | ~2–3.6 s |
| TTS (pyttsx3) | ~0.1–1.1 s |
| **Total round-trip** | **~4–5 s** |

VAD and TTS stayed well under a second combined across every test run. Gemini's time-to-first-token was consistently the largest chunk of the total — pointing to starting TTS on partial replies instead of waiting for the full response as the next optimization.

## Tech stack

- **Transport:** FastAPI WebSocket (`/ws`), raw PCM16 audio frames
- **VAD / turn detection:** Silero VAD
- **STT:** `faster-whisper` (CTranslate2 backend, int8 quantization)
- **LLM:** Gemini API (streaming for chat; single-shot for triage)
- **TTS:** `pyttsx3` (offline, CPU-based)
- **Structured logging:** SQLite (WAL mode)
- **Ops automation (optional):** n8n
- **Deployment (optional):** Docker / Docker Compose

## Possible next steps

- Stream TTS on partial LLM output instead of waiting for the full reply, to cut perceived latency
- Swap batch STT for a streaming/incremental transcription approach
- Migrate `llm.py` off the deprecated `google.generativeai` package to `google.genai`
- Replace `pyttsx3` with a higher-quality, still-CPU, streaming-capable TTS model (e.g. Kokoro-82M via ONNX)
- Verify and enable the Docker/n8n/Grafana stack locally, then fold the verified result back into this README
