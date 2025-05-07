"""
realtime_demo.py – Conversazione vocale con GPT‑4o‑realtime (Windows‑friendly)

Prerequisiti
------------
python -m venv .venv && .venv\Scripts\activate
pip install --upgrade "openai>=1.77.0" sounddevice numpy python-dotenv
# Se sounddevice/PyAudio dà errore su Windows:
pip install pipwin && pipwin install pyaudio

Ricorda di impostare la variabile OPENAI_API_KEY (o usa un file .env).
"""

import asyncio, base64, numpy as np, sounddevice as sd
from dotenv import load_dotenv
from openai import AsyncOpenAI

# ---------- Parametri audio ----------
SAMPLE_RATE = 24_000           # Hz, formato nativo dell’API
CHUNK_MS    = 100              # durata di ogni blocco che inviamo
BLOCK = SAMPLE_RATE * CHUNK_MS // 1000

load_dotenv()                  # legge OPENAI_API_KEY
client = AsyncOpenAI()

async def main() -> None:
    # 1️⃣ WebSocket al modello realtime
    async with client.beta.realtime.connect(
        model="gpt-4o-realtime-preview"
    ) as conn:

        # 2️⃣ Configura la sessione (solo audio in/out, VAD server)
        await conn.session.update(session={
            "modalities": ["audio"],
            "voice": "alloy",
            "input_audio_format":  "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "silence_duration_ms": 300}
        })

        print("Parla pure (Ctrl‑C per uscire)…")

        loop = asyncio.get_running_loop()

        # --- Callback microfono: invia chunk base64 ----------
        def mic_callback(indata, frames, t, status):
            if status:
                print(status, flush=True)

            pcm16 = (indata[:, 0] * 32768).astype(np.int16).tobytes()
            b64   = base64.b64encode(pcm16).decode("ascii")

            loop.call_soon_threadsafe(
                asyncio.create_task,
                conn.input_audio_buffer.append(audio=b64)  # ⬅️ PARAMETRO GIUSTO
            )

        # --- Stream microfono + event loop -------------------
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                            blocksize=BLOCK, dtype='float32',
                            callback=mic_callback):

            async for event in conn:
                if event.type == "response.audio.delta":
                    raw = base64.b64decode(event.delta)
                    audio = np.frombuffer(raw, dtype=np.int16
                              ).astype(np.float32) / 32768
                    sd.play(audio, SAMPLE_RATE)

                elif event.type == "response.done":
                    sd.wait()
                    print("\n🌟 Fine risposta, puoi parlare di nuovo.")

if __name__ == "__main__":
    asyncio.run(main())
