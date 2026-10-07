"""
Reference client: streams your mic to the server, plays back whatever
comes back. This is a TEST HARNESS, not part of the "product" -- in
a real deployment a browser or Plivo's telephony stack would be the
thing sending audio over the WebSocket, not this script.
"""
import asyncio
import io
import json

import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # matches the server's VAD window size
SERVER_URL = "ws://localhost:8000/ws"


async def mic_sender(ws):
    loop = asyncio.get_event_loop()
    q = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        loop.call_soon_threadsafe(q.put_nowait, pcm16)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=CHUNK_SAMPLES, callback=callback,
    )
    with stream:
        while True:
            chunk = await q.get()
            await ws.send(chunk)


async def receiver(ws):
    async for message in ws:
        if isinstance(message, str):
            data = json.loads(message)
            print("\n--- turn ---")
            print("you said:", data["transcript"])
            print("agent:   ", data["reply"])
            print("latency (ms):", data["latency_ms"])
        else:
            audio, sr = sf.read(io.BytesIO(message), dtype="float32")
            sd.play(audio, sr)
            sd.wait()


async def main():
    async with websockets.connect(SERVER_URL, max_size=None) as ws:
        print("connected. start talking...")
        await asyncio.gather(mic_sender(ws), receiver(ws))


if __name__ == "__main__":
    asyncio.run(main())
