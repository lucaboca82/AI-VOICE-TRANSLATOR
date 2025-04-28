import os
import numpy as np
import gradio as gr
import assemblyai as aai
from translate import Translator
import uuid
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pathlib import Path
from dotenv import load_dotenv
import time
import threading

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and voice ID from environment variables
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")

# Set up AssemblyAI API key
aai.settings.api_key = ASSEMBLYAI_API_KEY

def delete_file_after_delay(file_path, delay=300):
    """Delete the file after a given delay (default 5 minutes)"""
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    else:
        print(f"File {file_path} already deleted or doesn't exist.")

def voice_to_voice(audio_file):
    # Transcribe speech to text
    transcript = transcribe_audio(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        raise gr.Error(transcript.error)
    else:
        transcript = transcript.text

    # Translate text into different languages
    list_translations = translate_text(transcript)
    generated_audio_paths = []

    # Generate speech for each translated text
    for translation in list_translations:
        translated_audio_file_name = text_to_speech(translation)
        path = Path(translated_audio_file_name)
        generated_audio_paths.append(path)

    return generated_audio_paths[0], generated_audio_paths[1], generated_audio_paths[2], generated_audio_paths[3], generated_audio_paths[4], generated_audio_paths[5], list_translations[0], list_translations[1], list_translations[2], list_translations[3], list_translations[4], list_translations[5]

# Function to transcribe audio using AssemblyAI
def transcribe_audio(audio_file):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)
    return transcript

# Function to translate text
def translate_text(text: str) -> list:
    languages = ["ru", "tr", "sv", "de", "es", "ja"]
    list_translations = []

    for lan in languages:
        translator = Translator(from_lang="en", to_lang=lan)
        translation = translator.translate(text)
        list_translations.append(translation)

    return list_translations

# Function to generate speech using ElevenLabs
def text_to_speech(text: str) -> str:
    client = ElevenLabs(
        api_key=ELEVENLABS_API_KEY,
    )

    # Calling the text_to_speech conversion API with detailed parameters
    response = client.text_to_speech.convert(
        voice_id=VOICE_ID,  # Use the voice ID from environment variable
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",  # Use the turbo model for low latency
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
        ),
    )

    # Generate a unique file name and save the audio
    save_file_path = f"{uuid.uuid4()}.mp3"
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    # Start a background thread to delete the file after 5 minutes
    threading.Thread(target=delete_file_after_delay, args=(save_file_path, 300), daemon=True).start()

    return save_file_path

# Gradio interface for recording and displaying results
input_audio = gr.Audio(
    sources=["microphone"],
    type="filepath",
    show_download_button=True,
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
)

with gr.Blocks() as demo:
    gr.Markdown("## Record yourself in English and immediately receive voice translations.")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["microphone"],
                                type="filepath",
                                show_download_button=True,
                                waveform_options=gr.WaveformOptions(
                                    waveform_color="#01C6FF",
                                    waveform_progress_color="#0066B4",
                                    skip_length=2,
                                    show_controls=False,
                                ),)
            with gr.Row():
                submit = gr.Button("Submit", variant="primary")
                btn = gr.ClearButton(audio_input)

    with gr.Row():
        with gr.Group() as turkish:
            tr_output = gr.Audio(label="Turkish", interactive=False)
            tr_text = gr.Markdown()

        with gr.Group() as swedish:
            sv_output = gr.Audio(label="Swedish", interactive=False)
            sv_text = gr.Markdown()

        with gr.Group() as russian:
            ru_output = gr.Audio(label="Russian", interactive=False)
            ru_text = gr.Markdown()

    with gr.Row():
        with gr.Group():
            de_output = gr.Audio(label="German", interactive=False)
            de_text = gr.Markdown()

        with gr.Group():
            es_output = gr.Audio(label="Spanish", interactive=False)
            es_text = gr.Markdown()

        with gr.Group():
            jp_output = gr.Audio(label="Japanese", interactive=False)
            jp_text = gr.Markdown()

    output_components = [ru_output, tr_output, sv_output, de_output, es_output, jp_output, ru_text, tr_text, sv_text, de_text, es_text, jp_text]
    submit.click(fn=voice_to_voice, inputs=audio_input, outputs=output_components, show_progress=True)

if __name__ == "__main__":
    demo.launch()
