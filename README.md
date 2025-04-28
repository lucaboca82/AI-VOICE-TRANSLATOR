# AI VOICE TRANSLATOR

AI Voice Translator is an advanced speech-to-speech translation application built using Python, Gradio, AssemblyAI, ElevenLabs, and Google Translate. It enables real-time voice translation from English into multiple languages, providing both translated text and audio output.

## Features

- **Speech-to-Text**: Converts spoken English into text using AssemblyAI's transcription API.
- **Text Translation**: Translates the transcribed text into six languages: Russian, Turkish, Swedish, German, Spanish, and Japanese.
- **Voice Synthesis**: Generates translated speech using ElevenLabs API for text-to-speech (TTS).
- **Real-Time Translation**: Translates and speaks the translated text instantly after user input.
- **Interactive Interface**: Built using Gradio, allowing easy microphone input and audio output.

## Technologies Used

- **Python**: Main programming language used for the backend.
- **Gradio**: A library for building user-friendly interfaces for machine learning models.
- **AssemblyAI**: Used for speech-to-text transcription of audio files.
- **ElevenLabs**: Text-to-speech synthesis, generating voice output for translated text.
- **Google Translate**: For translating English text into multiple target languages (Russian, Turkish, Swedish, German, Spanish, and Japanese).
- **UUID**: Generates unique file names for audio files.
- **Threading**: Ensures temporary files are automatically deleted after a set time (5 minutes) to save storage.
- **Dotenv**: For securely loading API keys and configurations from environment variables.

## How It Works

1. **Record Audio**: Speak in English using the microphone.
2. **Transcription**: The audio is transcribed into text using AssemblyAI.
3. **Translation**: The transcribed text is translated into multiple languages (Russian, Turkish, Swedish, German, Spanish, Japanese).
4. **Voice Synthesis**: The translated text is then converted back into speech using ElevenLabs TTS, and the audio is returned to the user.
5. **Real-Time Output**: The application provides both the translated text and the corresponding audio for each language.

## Setup Instructions

### Prerequisites

To run this project locally, you need the following Python libraries:

- gradio
- assemblyai
- translate
- elevenlabs
- python-dotenv
- threading
- uuid
- os

You can install all dependencies by running the following command:

```bash
pip install -r requirements.txt
