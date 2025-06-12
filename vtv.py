#!/usr/bin/env python3
"""
app.py – Multilingual Voice Translator (2025 Edition)

Features:
  • Async transcription, translation & TTS in parallel
  • Disk + LRU caching of translations & audio files
  • Persistent SQLite DB of transcripts & translations
  • Dynamic language selection via ENV or UI
  • Graceful error handling & detailed logging
  • Auto-cleanup of stale audio files
"""

import os
import uuid
import asyncio
import logging
import sqlite3
import atexit
import shutil
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta

import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
from pydantic import BaseSettings, Field

# ─── Configuration ─────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    ASSEMBLYAI_API_KEY: str
    ELEVENLABS_API_KEY: str
    VOICE_ID: str
    TARGET_LANGS: list[str] = Field(
        default_factory=lambda: ["ru", "tr", "sv", "de", "es", "ja"]
    )
    CACHE_DIR: Path = Path("cache")
    DATA_DIR: Path = Path("data")
    DB_PATH: Path = Path("data/translations.db")
    AUDIO_TTL_SECONDS: int = 3600  # keep audio for 1h

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
tts_client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ─── Prepare directories & DB ─────────────────────────────────────────────────
for d in (settings.CACHE_DIR, settings.DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

_conn: sqlite3.Connection | None = None

def init_db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
        cur = _conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                transcript TEXT,
                lang TEXT,
                translation TEXT,
                audio_path TEXT
            )
        """)
        _conn.commit()
    return _conn

def save_record(
    rec_id: str,
    transcript: str,
    lang: str,
    translation: str,
    audio_path: str
) -> None:
    conn = init_db()
    conn.execute("""
        INSERT OR REPLACE INTO records
        (id, timestamp, transcript, lang, translation, audio_path)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        rec_id,
        datetime.utcnow().isoformat(),
        transcript,
        lang,
        translation,
        audio_path
    ))
    conn.commit()

atexit.register(lambda: _conn and _conn.close())

# ─── Cleanup stale files ──────────────────────────────────────────────────────
async def cleanup_cache():
    """Delete audio files older than TTL."""
    now = datetime.utcnow()
    for file in settings.CACHE_DIR.glob("*.mp3"):
        if now - datetime.fromtimestamp(file.stat().st_mtime) > timedelta(seconds=settings.AUDIO_TTL_SECONDS):
            try:
                file.unlink()
                logging.debug(f"Deleted stale audio: {file.name}")
            except Exception as e:
                logging.warning(f"Failed to delete {file}: {e}")

# schedule cleanup every hour
async def schedule_cleanup():
    while True:
        await cleanup_cache()
        await asyncio.sleep(settings.AUDIO_TTL_SECONDS)

# ─── Core Functions ───────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def translate_text(text: str, target: str) -> str:
    """Translate text with LRU cache."""
    logging.info(f"Translating to {target}...")
    return Translator(from_lang="en", to_lang=target).translate(text)

async def transcribe_audio(file_path: str) -> str:
    """Async transcription via AssemblyAI."""
    logging.info("Transcribing audio...")
    transcriber = aai.Transcriber()
    job = await asyncio.to_thread(transcriber.transcribe, file_path)
    if job.status == aai.TranscriptStatus.error:
        logging.error(f"Transcription error: {job.error}")
        raise RuntimeError(job.error)
    logging.info("Transcription complete.")
    return job.text

async def text_to_speech(text: str, lang: str, rec_id: str) -> Path:
    """Async TTS via ElevenLabs, saves file and DB record."""
    logging.info(f"Generating TTS [{lang}] …")
    resp = tts_client.text_to_speech.convert(
        voice_id=settings.VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_22050_32",
        optimize_streaming_latency="0",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True
        ),
    )
    out_path = settings.CACHE_DIR / f"{rec_id}-{lang}.mp3"
    with open(out_path, "wb") as f:
        async for chunk in resp:
            if chunk:
                f.write(chunk)
    logging.info(f"TTS saved: {out_path.name}")

    # Save record in DB
    # transcript is stored separately in orchestrator
    save_record(rec_id, "", lang, text, str(out_path))
    return out_path

# ─── Orchestrator ─────────────────────────────────────────────────────────────

async def voice_to_voice(
    file_path: str,
    langs: list[str]
) -> list[Path | str]:
    """
    1) Transcribe → 2) Translate → 3) TTS in parallel
    Returns list: [audio1, audio2, ..., transcript1, transcript2, ...]
    """
    # schedule cleanup task
    if not hasattr(voice_to_voice, "_cleanup_started"):
        asyncio.create_task(schedule_cleanup())
        setattr(voice_to_voice, "_cleanup_started", True)

    rec_id = uuid.uuid4().hex
    transcript = await transcribe_audio(file_path)

    tasks: list[asyncio.Task] = []
    for lang in langs:
        # translate text (sync cached)
        translated = await asyncio.to_thread(translate_text, transcript, lang)
        # record transcript-text mapping
        save_record(rec_id, transcript, lang, translated, "")
        # schedule TTS
        tasks.append(text_to_speech(translated, lang, rec_id))

    # run TTS in parallel
    audio_paths = await asyncio.gather(*tasks)

    # prepare outputs: audios + captions
    outputs: list[Path | str] = []
    for path, lang in zip(audio_paths, langs):
        outputs += [path, f"({lang}) {translate_text(transcript, lang)}"]

    return outputs

# ─── Gradio UI ───────────────────────────────────────────────────────────────

with gr.Blocks(css=""" 
  body { font-family: 'Montserrat', sans-serif; } 
  .gradio-container { max-width: 900px; margin: auto; }
""") as demo:
    gr.Markdown("## 🎙️ Speak English → Multilingual Voice Translations")
    langs = gr.CheckboxGroup(
        label="Select target languages",
        choices=settings.TARGET_LANGS,
        value=settings.TARGET_LANGS,
        interactive=True
    )
    audio_in = gr.Audio(source="microphone", type="filepath")
    submit = gr.Button("Translate", variant="primary")

    # build output slots: (Audio, Text) x N
    outputs: list[gr.components.Component] = []
    for lg in settings.TARGET_LANGS:
        outputs += [
            gr.Audio(label=f"🔊 {lg.upper()}", interactive=False),
            gr.Markdown(label=f"✏️ {lg.upper()} Transcript")
        ]

    submit.click(
        fn=voice_to_voice,
        inputs=[audio_in, langs],
        outputs=outputs,
        show_progress=True
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=True,
        debug=False
    )
