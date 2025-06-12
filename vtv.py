#!/usr/bin/env python3
"""
app.py – Multilingual Voice Translator (2025 Ultra-Modern Edition)

Upgrades & Features:
  • Async transcription, translation & TTS in parallel with retries & backoff
  • Local Whisper fallback if AssemblyAI unavailable
  • Disk + LRU + SQLite caching of transcripts, translations & audio
  • Zip-on-the-fly for “Download All”
  • Prometheus metrics & health endpoint
  • Sentry integration for error tracking
  • Graceful shutdown, clean exit, TTL cleanup of stale files
  • Fully typed, Pydantic settings, contextual logging
Features & Upgrades:
  • Async transcription with AssemblyAI + local Whisper fallback
  • Async translation & TTS in parallel with retry/backoff
  • Disk + LRU + SQLite caching of transcripts, translations & MP3s
  • “Download All” ZIP bundle on the fly
  • TTL-driven cleanup of stale files
  • Prometheus metrics & /health endpoint
  • Sentry integration for error tracking
  • Pydantic settings, contextual logging, graceful shutdown
"""

import os
import uuid
import asyncio
import logging
import sqlite3
import tempfile
import zipfile
import atexit
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import gradio as gr
import assemblyai as aai
import backoff
import prometheus_client
import sentry_sdk
import whisper
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pydantic import BaseSettings, Field
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from dotenv import load_dotenv
from dotenv import load_dotenv
from pydantic import BaseSettings, Field
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

# ─── Settings ────────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    ASSEMBLYAI_API_KEY: str
    ELEVENLABS_API_KEY: str
    VOICE_ID: str
    TARGET_LANGS: list[str] = Field(default_factory=lambda: ["ru", "tr", "sv", "de", "es", "ja"])
    CACHE_DIR: Path = Path("cache")
    DATA_DIR: Path = Path("data")
    DB_PATH: Path = Path("data/translations.db")
    AUDIO_TTL_SECONDS: int = 3600
    AUDIO_TTL_SECONDS: int = 3600  # 1 hour
    SENTRY_DSN: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
load_dotenv()
if settings.SENTRY_DSN:
    sentry_sdk.init(dsn=settings.SENTRY_DSN)

# ─── Logging & Metrics ───────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
REQUEST_COUNT = prometheus_client.Counter("requests_total", "Total translation requests")
REQUEST_LATENCY = prometheus_client.Histogram("request_latency_seconds", "Translation request latency")

# ─── Prepare dirs & DB ──────────────────────────────────────────────────────
for d in (settings.CACHE_DIR, settings.DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

_conn: sqlite3.Connection | None = None
    sentry_sdk.init(dsn=settings.SENTRY_DSN, traces_sample_rate=1.0)

# ─── Logging & Metrics ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
REQUEST_COUNT = prometheus_client.Counter("requests_total", "Total translation requests")
REQUEST_LATENCY = prometheus_client.Histogram("request_latency_seconds", "Translation latency")

# ─── Prepare Directories & Database ─────────────────────────────────────────
for folder in (settings.CACHE_DIR, settings.DATA_DIR):
    folder.mkdir(parents=True, exist_ok=True)

_conn: sqlite3.Connection | None = None
def init_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
        cur = _conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id TEXT, lang TEXT, 
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id TEXT, lang TEXT,
                transcript TEXT, translation TEXT,
                audio_path TEXT, timestamp TEXT,
                PRIMARY KEY (id, lang)
            )
        """)
        _conn.commit()
    return _conn

atexit.register(lambda: _conn and _conn.close())

# ─── Cleanup stale files ─────────────────────────────────────────────────────
async def cleanup_cache():
    now = datetime.utcnow()
    for mp3 in settings.CACHE_DIR.glob("*.mp3"):
        if now - datetime.fromtimestamp(mp3.stat().st_mtime) > timedelta(seconds=settings.AUDIO_TTL_SECONDS):
            try: mp3.unlink()
            except: pass
# ─── TTL Cleanup ──────────────────────────────────────────────────────────────
async def cleanup_cache():
    """Remove audio files older than TTL."""
    now = datetime.utcnow()
    for mp3 in settings.CACHE_DIR.glob("*.mp3"):
        if now - datetime.fromtimestamp(mp3.stat().st_mtime) > timedelta(seconds=settings.AUDIO_TTL_SECONDS):
            try:
                mp3.unlink()
                logging.debug(f"Deleted stale audio: {mp3.name}")
            except Exception as e:
                logging.warning(f"Could not delete {mp3}: {e}")

async def periodic_cleanup():
    while True:
        await cleanup_cache()
        await asyncio.sleep(settings.AUDIO_TTL_SECONDS)
asyncio.create_task(periodic_cleanup())

# ─── Whisper fallback loader ───────────────────────────────────────────────────
_whisper_model = whisper.load_model("small")

# ─── Core functions ──────────────────────────────────────────────────────────

@backoff.on_exception(backoff.expo, Exception, max_tries=4, jitter=backoff.full_jitter)
async def transcribe_audio(path: str) -> str:
    """Try AssemblyAI, fallback to Whisper local."""
    try:
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        trans = aai.Transcriber()
        job = await asyncio.to_thread(trans.transcribe, path)

# schedule cleanup task when event loop starts
asyncio.get_event_loop().create_task(periodic_cleanup())

# ─── Whisper Fallback ────────────────────────────────────────────────────────
_whisper_model = whisper.load_model("small")

# ─── Core Functions ──────────────────────────────────────────────────────────
@backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter)
async def transcribe_audio(path: str) -> str:
    """Attempt AssemblyAI transcription, fallback to Whisper."""
    try:
        aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
        job = await asyncio.to_thread(aai.Transcriber().transcribe, path)
        if job.status == aai.TranscriptStatus.error:
            raise RuntimeError(job.error)
        return job.text
    except Exception:

        logging.warning("AssemblyAI failed, falling back to Whisper")

        logging.warning("AssemblyAI failed, using Whisper locally")

        result = _whisper_model.transcribe(path)
        return result["text"].strip()

@lru_cache(maxsize=512)
def translate_text(text: str, target: str) -> str:

    """Translate with LRU cache."""
    return Translator(from_lang="en", to_lang=target).translate(text)

@backoff.on_exception(backoff.expo, Exception, max_tries=4, jitter=backoff.full_jitter)
async def text_to_speech(text: str, lang: str, rec_id: str) -> Path:
    client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
    resp = client.text_to_speech.convert(

    """Cached text translation."""
    logging.info(f"Translating to {target}")
    return Translator(from_lang="en", to_lang=target).translate(text)

@backoff.on_exception(backoff.expo, Exception, max_tries=3, jitter=backoff.full_jitter)
async def text_to_speech(text: str, lang: str, rec_id: str) -> Path:
    """Generate TTS via ElevenLabs and save to cache."""
    client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
    response = client.text_to_speech.convert(

        voice_id=settings.VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_22050_32",
        optimize_streaming_latency="0",

        voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.8, style=0.5, use_speaker_boost=True),
    )
    out = settings.CACHE_DIR / f"{rec_id}-{lang}.mp3"
    with open(out, "wb") as f:
        async for chunk in resp:
            if chunk: f.write(chunk)
    return out

async def voice_to_voice(path: str, langs: list[str]) -> list[str]:
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        rec_id = uuid.uuid4().hex
        transcript = await transcribe_audio(path)
        db = init_db()
        outputs: list[str] = []
        # create temp zip
        zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
        with zipfile.ZipFile(zip_path, "w") as zf:
            for lang in langs:
                trans = translate_text(transcript, lang)
                audio = await text_to_speech(trans, lang, rec_id)
                timestamp = datetime.utcnow().isoformat()
                db.execute("""
                    INSERT OR REPLACE INTO records(id, lang, transcript, translation, audio_path, timestamp)
                    VALUES(?,?,?,?,?,?)
                """, (rec_id, lang, transcript, trans, str(audio), timestamp))
                db.commit()
                zf.write(audio, arcname=audio.name)
                outputs += [str(audio), trans]
            zf.writestr("transcript.txt", transcript)
        outputs.append(zip_path)
        return outputs

# ─── Health & Metrics App ────────────────────────────────────────────────────
fastapp = FastAPI()
@fastapp.get("/health")
async def health(): return {"status": "ok"}
fastapp.mount("/metrics", prometheus_client.make_asgi_app())

# ─── Gradio UI ───────────────────────────────────────────────────────────────
with gr.Blocks(css="styles.css") as demo:
    gr.Markdown("## 🎙️ Speak English → Multilingual Voice Translator")
    langs = gr.CheckboxGroup("Select languages", settings.TARGET_LANGS, settings.TARGET_LANGS)
    audio_in = gr.Audio(source="microphone", type="filepath")
    btn = gr.Button("Translate")

    outputs = []
    for lg in settings.TARGET_LANGS:
        outputs += [gr.Audio(interactive=False, label=f"🔊 {lg.upper()}"), gr.Markdown()]
    outputs += [gr.File(label="📦 Download All Translations (.zip)")]

        voice_settings=VoiceSettings(
            stability=0.5, similarity_boost=0.8, style=0.5, use_speaker_boost=True
        ),
    )
    out_file = settings.CACHE_DIR / f"{rec_id}-{lang}.mp3"
    with open(out_file, "wb") as f:
        async for chunk in response:
            if chunk:
                f.write(chunk)
    logging.info(f"TTS saved: {out_file.name}")
    return out_file

async def voice_to_voice(path: str, langs: list[str]) -> list[str]:
    """Full pipeline: transcribe, translate, TTS, ZIP and metrics."""
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        rec_id = uuid.uuid4().hex
        text = await transcribe_audio(path)
        db = init_db()
        outputs: list[str] = []
        # Create a zip bundle
        zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for lang in langs:
                translated = translate_text(text, lang)
                audio_file = await text_to_speech(translated, lang, rec_id)
                timestamp = datetime.utcnow().isoformat()
                db.execute("""
                    INSERT OR REPLACE INTO records (id, lang, transcript, translation, audio_path, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (rec_id, lang, text, translated, str(audio_file), timestamp))
                db.commit()
                zipf.write(audio_file, arcname=audio_file.name)
                outputs += [str(audio_file), translated]
            zipf.writestr("original_transcript.txt", text)
        outputs.append(zip_path)
        return outputs

# ─── FastAPI for Health & Metrics ────────────────────────────────────────────
fastapi_app = FastAPI(title="Multilingual Voice Translator API")

@fastapi_app.get("/health")
async def health_check():
    return {"status": "ok"}

fastapi_app.mount("/metrics", prometheus_client.make_asgi_app())

# ─── Gradio UI ───────────────────────────────────────────────────────────────
with gr.Blocks(css="styles.css") as gradio_ui:
    gr.Markdown("## 🎙️ Speak English → Multilingual Voice Translator")
    langs = gr.CheckboxGroup(
        label="Select target languages",
        choices=settings.TARGET_LANGS,
        value=settings.TARGET_LANGS,
        interactive=True
    )
    audio_in = gr.Audio(source="microphone", type="filepath")
    btn = gr.Button("Translate", variant="primary")

    outputs = []
    for lg in settings.TARGET_LANGS:
        outputs += [
            gr.Audio(label=f"🔊 {lg.upper()}", interactive=False),
            gr.Markdown(label=f"✏️ {lg.upper()} Translation")
        ]
    # final slot: ZIP bundle for all translations
    outputs.append(gr.File(label="📦 Download All Translations (.zip)"))


    btn.click(fn=voice_to_voice, inputs=[audio_in, langs], outputs=outputs, show_progress=True)

# ─── Mount Gradio into FastAPI & Run ─────────────────────────────────────────

app = fastapp
app.mount("/", WSGIMiddleware(demo.launch(embed=True, prevent_thread_lock=True)))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 7860)), log_level="info")

app = fastapi_app
app.mount("/", WSGIMiddleware(gradio_ui.launch(embed=True, prevent_thread_lock=True)))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), log_level="info")
        # final slot: ZIP bundle for all translations
    outputs.append(gr.File(label="📦 Download All Translations (.zip)"))

    btn.click(
        fn=voice_to_voice,
        inputs=[audio_in, langs],
        outputs=outputs,
        show_progress=True
    )

# ─── Mount Gradio into FastAPI & Run ─────────────────────────────────────────
app = fastapi_app
app.mount(
    "/", 
    WSGIMiddleware(gradio_ui.launch(embed=True, prevent_thread_lock=True))
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level="info"
    )
