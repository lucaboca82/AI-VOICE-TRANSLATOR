# AI VOICE TRANSLATOR

AI Voice Translator is a Python-based application that converts audio input in English into multiple languages (Turkish, Swedish, Russian, German, Spanish, Japanese). It uses Gradio for the frontend, AssemblyAI for speech-to-text transcription, and ElevenLabs for text-to-speech translation in different languages.

## Features

- **Speech-to-Text**: Converts spoken English audio to text using AssemblyAI.
- **Text Translation**: Translates the transcribed text into Turkish, Swedish, Russian, German, Spanish, and Japanese.
- **Text-to-Speech**: Converts the translated text into audio using ElevenLabs API.
- **Multi-language Support**: Translates and generates speech in six languages.

## Tech Stack

- **Python**: Main programming language.
- **Gradio**: User interface for audio recording and displaying translated results.
- **AssemblyAI**: Transcription service for converting speech to text.
- **ElevenLabs**: Text-to-speech API for generating audio from translated text.
- **dotenv**: Environment variables management.
- **uuid**: For generating unique file names for audio files.

## Demo

You can try out the AI Voice Translator live on Hugging Face Spaces. Record your speech in English, and instantly receive translations in multiple languages. Experience it here:
[Try the Demo](https://huggingface.co/spaces/aridepai17/aivoicetranslator)


## Prerequisites

- Python 3.7 or later
- Install required Python packages:

```bash
pip install -r requirements.txt
```

## LICENSE
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

