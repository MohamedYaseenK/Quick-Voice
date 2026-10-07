"""
Turns a transcript into a spoken-style response, using Gemini.
Streams the completion so we can measure time-to-first-token (TTFT) --
for a voice agent, TTFT matters more to perceived latency than total
generation time, since TTS could start as soon as the first sentence
is out (a further optimization this v1 doesn't implement yet).
"""
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"

SYSTEM_PROMPT = (
    "You are a voice assistant. Keep replies to 1-2 short sentences -- "
    "they will be spoken aloud, not read."
)

TRIAGE_SYSTEM_PROMPT = (
    "You are a support engineer's triage assistant for a voice agent platform. "
    "You will be given one call's per-stage latency breakdown (vad, stt, "
    "llm_ttft, llm_total, tts, total, all in milliseconds), its flag threshold, "
    "and its transcript. Identify which stage, if any, is the anomaly and state "
    "it in one or two plain sentences. If nothing is clearly abnormal relative "
    "to the threshold, say so explicitly -- do not invent a cause. Never guess "
    "at causes outside the data given."
)

_chat_model = None
_triage_model = None


def _require_api_key():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file -- "
            "see https://aistudio.google.com/apikey"
        )
    return api_key


def _get_chat_model():
    global _chat_model
    if _chat_model is None:
        genai.configure(api_key=_require_api_key())
        _chat_model = genai.GenerativeModel(MODEL_NAME, system_instruction=SYSTEM_PROMPT)
    return _chat_model


def _get_triage_model():
    global _triage_model
    if _triage_model is None:
        genai.configure(api_key=_require_api_key())
        _triage_model = genai.GenerativeModel(MODEL_NAME, system_instruction=TRIAGE_SYSTEM_PROMPT)
    return _triage_model


def respond(user_text: str, history: list) -> tuple[str, float]:
    """
    Returns (full_response_text, ttft_seconds).
    `history` is a list of {"role": "user"/"model", "parts": [text]} dicts
    (Gemini's chat format), mutated in place so the conversation carries
    context across turns.
    """
    history.append({"role": "user", "parts": [user_text]})
    start = time.perf_counter()
    ttft = None
    chunks = []

    stream = _get_chat_model().generate_content(history, stream=True)
    for chunk in stream:
        if chunk.text:
            if ttft is None:
                ttft = time.perf_counter() - start
            chunks.append(chunk.text)

    full_text = "".join(chunks)
    history.append({"role": "model", "parts": [full_text]})
    return full_text, (ttft or 0.0)


def triage_call(call: dict, threshold_ms: float) -> str:
    """
    call: a row dict from db.get_call() -- has transcript, reply, and the
    vad_ms/stt_ms/llm_ttft_ms/llm_total_ms/tts_ms/total_ms fields.
    Returns a plain-English root-cause summary. Raises RuntimeError if
    GEMINI_API_KEY is missing -- the caller (the REST endpoint) turns that
    into a clean HTTP error rather than letting it crash the process.
    """
    prompt = (
        f"Flag threshold for this period: {threshold_ms:.1f} ms total.\n"
        f"Call latency breakdown (ms): vad={call['vad_ms']}, stt={call['stt_ms']}, "
        f"llm_ttft={call['llm_ttft_ms']}, llm_total={call['llm_total_ms']}, "
        f"tts={call['tts_ms']}, total={call['total_ms']}.\n"
        f"Transcript: {call['transcript']!r}\n"
        f"Reply: {call['reply']!r}"
    )
    response = _get_triage_model().generate_content(prompt)
    return response.text.strip()
