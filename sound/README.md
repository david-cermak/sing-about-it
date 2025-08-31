# 🎵 PDF to MP3 Audio Converter

Convert PDF documents into high-quality MP3 audiobooks using advanced text-to-speech technology.

## 🚀 Quick Start

Convert any PDF to MP3 in one command:

```bash
python main.py document.pdf
```

That's it! Your PDF will be converted to a natural-sounding MP3 audiobook.

## 📋 What It Does

This pipeline transforms PDF documents into audio files through three stages:

1. **PDF → Markdown**: Extracts and formats text from PDF
2. **Markdown → WAV**: Converts text to speech using Kokoro TTS
3. **WAV → MP3**: Compresses audio to MP3 format

## 🛠️ Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements

- **Python 3.8+**
- **FFmpeg** (for MP3 conversion) - [Download here](https://ffmpeg.org/download.html)
- **Dependencies**: PyMuPDF, Kokoro TTS, pydub, soundfile, numpy

## 📖 Usage Examples

### Full Pipeline (PDF → MP3)

```bash
# Basic conversion
python main.py document.pdf

# Custom voice and quality
python main.py document.pdf --voice af_nova --speed 1.2 --bitrate 320k

# Keep intermediate files for inspection
python main.py document.pdf --keep-intermediate

# Custom output filename
python main.py document.pdf --output my_audiobook.mp3
```

### Partial Pipelines

```bash
# PDF to Markdown only
python main.py document.pdf --markdown-only

# PDF to WAV only
python main.py document.pdf --wav-only --voice af_heart --speed 1.1
```

### Individual Scripts

```bash
# Step-by-step processing
python convert_pdf2md.py document.pdf --output document.md
python convert_md2wav.py document.md --voice af_nova --speed 1.2
python convert_wav2mp3.py document.wav --bitrate 320k
```

## 🎙️ Voice Options

### Available Voices

| Voice Code | Description | Gender |
|------------|-------------|---------|
| `af_heart` | Warm, natural (default) | Female |
| `af_nova` | Clear, professional | Female |
| `af_sky` | Bright, energetic | Female |
| `af_sarah` | Calm, soothing | Female |
| `am_michael` | Deep, authoritative | Male |
| `am_adam` | Friendly, conversational | Male |
| `bm_lewis` | British accent | Male |
| `bf_emma` | British accent | Female |

### Language Support

| Code | Language | Flag |
|------|----------|------|
| `a` | American English | 🇺🇸 |
| `b` | British English | 🇬🇧 |
| `e` | Spanish | 🇪🇸 |
| `f` | French | 🇫🇷 |
| `h` | Hindi | 🇮🇳 |
| `i` | Italian | 🇮🇹 |
| `j` | Japanese | 🇯🇵 |
| `p` | Brazilian Portuguese | 🇧🇷 |
| `z` | Mandarin Chinese | 🇨🇳 |

## ⚙️ Advanced Options

### Main Pipeline Options

| Option | Description | Default |
|--------|-------------|---------|
| `--voice` | TTS voice selection | `af_heart` |
| `--speed` | Speaking speed multiplier | `1.0` |
| `--bitrate` | MP3 bitrate quality | `192k` |
| `--output` | Output filename | `{pdf_name}.mp3` |
| `--keep-intermediate` | Keep .md and .wav files | `false` |

### Quality Settings

| Bitrate | Quality | File Size | Use Case |
|---------|---------|-----------|-----------|
| `128k` | Good | Smallest | Mobile, streaming |
| `192k` | High | Medium | General use (default) |
| `256k` | Very High | Large | Archival quality |
| `320k` | Maximum | Largest | Professional use |

## 🔧 Individual Script Usage

### PDF to Markdown (`convert_pdf2md.py`)

```bash
# Basic conversion
python convert_pdf2md.py document.pdf

# Custom output and options
python convert_pdf2md.py document.pdf --output custom.md --no-page-numbers

# Print to console
python convert_pdf2md.py document.pdf --stdout
```

### Markdown to WAV (`convert_md2wav.py`)

```bash
# Basic conversion
python convert_md2wav.py document.md

# Custom voice and speed
python convert_md2wav.py document.md --voice af_nova --speed 1.3

# Different language
python convert_md2wav.py document.md --lang-code f --voice bf_emma
```

### WAV to MP3 (`convert_wav2mp3.py`)

```bash
# Basic conversion
python convert_wav2mp3.py audio.wav

# High quality
python convert_wav2mp3.py audio.wav --bitrate 320k

# Alternative method (no FFmpeg)
python convert_wav2mp3.py audio.wav --alternative
```

## 🎯 Use Cases

### 📚 **Educational Content**
- Convert textbooks and research papers to audio
- Create study materials for auditory learners
- Make academic content accessible while commuting

### 📰 **Professional Documents**
- Turn reports and documentation into podcasts
- Convert meeting notes to audio summaries
- Create audio versions of written presentations

### ♿ **Accessibility**
- Make documents accessible for visually impaired users
- Convert text for people with reading difficulties
- Create audio alternatives for any written content

## 🔍 Troubleshooting

### Common Issues

**FFmpeg Not Found**
```bash
# Install FFmpeg from https://ffmpeg.org/download.html
# Or use alternative conversion method:
python convert_wav2mp3.py audio.wav --alternative
```

**Out of Memory**
- Use smaller PDF files or split large documents
- Reduce TTS speed: `--speed 0.8`
- Lower MP3 bitrate: `--bitrate 128k`

**Poor Audio Quality**
- Increase bitrate: `--bitrate 320k`
- Try different voices: `--voice af_nova`
- Adjust speed: `--speed 0.9` (slower = clearer)

**No Audio Generated**
- Check if PDF contains readable text
- Verify Kokoro TTS installation: `pip install kokoro`
- Test with shorter documents first

### Getting Help

```bash
# Show all options
python main.py --help
python convert_pdf2md.py --help
python convert_md2wav.py --help
python convert_wav2mp3.py --help
```

## 📁 Output Files

| Extension | Description | When Created |
|-----------|-------------|--------------|
| `.mp3` | Final audio file | Always (default) |
| `.md` | Markdown text | With `--keep-intermediate` |
| `.wav` | Uncompressed audio | With `--keep-intermediate` |

## 🏗️ Architecture

The pipeline is designed as modular, reusable components:

```
📄 PDF Document
    ↓ (convert_pdf2md.py)
📝 Markdown Text
    ↓ (convert_md2wav.py)
🎧 WAV Audio
    ↓ (convert_wav2mp3.py)
🎵 MP3 File
```

Each step can be run independently or as part of the complete pipeline through `main.py`.

## 🤝 Contributing

Each script includes comprehensive error handling, progress reporting, and can be used both as libraries and standalone tools. The modular design makes it easy to extend or modify individual components.

---

**Made with ❤️ using Kokoro TTS and PyMuPDF**
