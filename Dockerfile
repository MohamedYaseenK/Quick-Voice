# UNVERIFIED IN THIS BUILD: the sandbox this was written in has no Docker
# daemon available (confirmed: `docker` not found). Build and run this
# locally before trusting it -- `docker build -t voice-agent .`

FROM python:3.11-slim

# espeak is pyttsx3's actual TTS engine on Linux -- without it,
# pyttsx3.init() raises at runtime, not at pip install time, so this is
# easy to miss until the first real call comes in.
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY db.py vad.py stt.py llm.py tts.py latency.py server.py ./
COPY static ./static

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
