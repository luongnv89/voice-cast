<p align="center">
  <img src="./logo.svg" alt="VoiceCast Logo" width="400">
</p>

<p align="center">
  <a href="https://github.com/luongnv89/voice-cast/stargazers"><img src="https://img.shields.io/github/stars/luongnv89/voice-cast?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://github.com/luongnv89/voice-cast/blob/main/LICENSE"><img src="https://img.shields.io/github/license/luongnv89/voice-cast?style=flat-square" alt="MIT License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue?style=flat-square" alt="Python 3.10+"></a>
  <img src="https://img.shields.io/badge/version-0.2.0-green?style=flat-square" alt="Version">
  <a href="https://github.com/luongnv89/voice-cast/issues"><img src="https://img.shields.io/github/issues/luongnv89/voice-cast?style=flat-square" alt="Issues"></a>
</p>

<h1 align="center">Clone Any Voice from a 5-Second Clip</h1>

<p align="center"><strong>Your words, any voice.</strong></p>

<p align="center">
  VoiceCast turns a short audio sample into a voice you can use for text-to-speech — in 16 languages,<br>
  with expressive emotions, through a desktop app, command line, or Python API.
</p>

<p align="center">
  <a href="#get-started-in-60-seconds"><strong>Get Started in 60 Seconds →</strong></a>
</p>

---

![VoiceCast GUI](./voicecast-app.png)

## Ever Needed a Specific Voice on Demand?

- You're building an audiobook, a game, or a prototype — and the voice matters, but hiring voice talent for every iteration is slow and expensive.
- You need multilingual narration but can't find a single voice that sounds natural across languages.
- Existing TTS tools produce robotic, flat output that doesn't match the expressiveness you need — no laughs, no sighs, no personality.

VoiceCast solves all three. Record 5–30 seconds of any voice, and generate natural, expressive speech in that voice — instantly, locally, for free.

## What VoiceCast Gives You

- **Any voice, cloned in seconds** — Feed in a 5–30 second WAV sample and VoiceCast learns the voice. No training, no cloud upload, no waiting.
- **16 languages, one tool** — English, Spanish, French, German, Chinese, Japanese, and 10 more. Switch languages without switching voices.
- **Expressive speech that sounds human** — Add `[laugh]`, `[sigh]`, `[gasp]`, and more with Chatterbox Turbo. Your cloned voice doesn't just talk — it *performs*.
- **Three ways to use it** — A polished desktop GUI for quick tasks, a CLI for automation, and a Python API for integration into your own projects.
- **Runs on your machine** — No API keys, no cloud dependencies, no per-word billing. Your voice data stays local.

<p align="center">
  <a href="#get-started-in-60-seconds"><strong>Start Cloning Voices Now →</strong></a>
</p>

## How It Works

1. **Install** — Clone the repo, create a virtual environment, and `pip install -e .`.
2. **Download a model intentionally** — VoiceCast does not download models during install or first generation; use the CLI or GUI Model Manager to fetch only what you need.
3. **Pick a voice sample** — Any clean 5–30 second audio clip of the voice you want to clone.
4. **Choose your engine** — Coqui XTTS v2 for multilingual quality, or Chatterbox for speed and expressiveness.
5. **Generate speech** — Type your text, hit generate, and get a WAV file in the cloned voice.

| Engine | Languages | Speed | Best For |
|--------|-----------|-------|----------|
| **Coqui XTTS v2** | 16 | Medium | Multilingual narration, production quality |
| **Chatterbox Turbo** | English | Fast | Rapid iteration, expressive speech with emotion tags |
| **Chatterbox Standard** | English | Medium | High-fidelity English output |

<p align="center">
  <a href="#get-started-in-60-seconds"><strong>Try It Yourself →</strong></a>
</p>

## Get Started in 60 Seconds

```bash
# Clone the repository
git clone https://github.com/luongnv89/voice-cast.git
cd voicecast

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install code and dependencies only; models are downloaded separately
pip install -e .

# See model cache status, then download only what you need
python vcloner.py --list-models
python vcloner.py --download-models coqui-xtts-v2
```

**Launch the GUI:**
```bash
python voice_cloning_app.py
```

Use the **Model Manager** tab to download models before generating.

**Or use the CLI:**
```bash
python vcloner.py -i voice.wav -t "Hello world" -o output.wav
```

**Or call the Python API:**
```python
from voice_cloner import VoiceCloner

# Explicit model download; normal generation never downloads implicitly.
VoiceCloner.download_model("coqui-xtts-v2")

cloner = VoiceCloner(speaker_wav="./voice-samples/speaker.wav")
cloner.say("Hello, this is my cloned voice!", save_audio=True, output_file="output.wav")
```

**Add expressive speech with Chatterbox Turbo:**
```python
cloner.say("That's hilarious [laugh]! I can't believe it [gasp]!")
```

Supported tags: `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`, `[gasp]`, `[yawn]`

## FAQ

**Is VoiceCast free?**
Yes. VoiceCast is MIT licensed — free for personal and commercial use, forever. See [LICENSE](LICENSE).

**Does it download models automatically?**
No. Install and initialization do not download model files. Run `python vcloner.py --list-models`, `python vcloner.py --download-models <model-id>`, or use the GUI Model Manager to manage local models.

**Does it need a GPU?**
No. VoiceCast runs on CPU. An NVIDIA GPU with CUDA speeds up generation significantly, and Apple Silicon users can install the optional MLX backend for hardware acceleration.

**What are the system requirements?**
Python 3.10+, 8GB RAM (16GB recommended). Optional: NVIDIA GPU with CUDA or Apple Silicon with MLX.

**How does Coqui compare to Chatterbox?**
Coqui XTTS v2 supports 16 languages and produces high-quality multilingual output. Chatterbox is English-only but faster and supports expressive emotion tags. Use both — VoiceCast makes switching engines seamless.

**Is my voice data sent to the cloud?**
No. Everything runs locally on your machine. No API keys, no cloud uploads, no telemetry.

**Can I use this in production?**
Yes. VoiceCast provides a Python API designed for integration. See the [API Reference](docs/api-reference.md) for details.

**How long does the voice sample need to be?**
5–30 seconds of clean speech. Longer samples can improve quality, but even 5 seconds produces usable results.

## Start Building with VoiceCast

VoiceCast puts voice cloning in your hands — no cloud, no cost, no restrictions. Clone voices for audiobooks, games, accessibility tools, creative projects, or anything else you can imagine.

MIT licensed. Runs locally. Works on Linux, macOS, and Windows.

[**Get Started in 60 Seconds →**](#get-started-in-60-seconds)

---

<details>
<summary>Documentation</summary>

| Document | Description |
|----------|-------------|
| [API Reference](docs/api-reference.md) | Complete Python API documentation |
| [CLI Reference](docs/cli-reference.md) | Command-line interface guide |
| [GUI Guide](docs/gui-guide.md) | Desktop application user manual |
| [Engines Guide](docs/engines.md) | TTS engine comparison and parameters |
| [Model Management](docs/model-management.md) | Explicit model download, cache, CLI, GUI, and API workflow |
| [Architecture](docs/architecture.md) | System design and patterns |
| [Development](docs/development.md) | Contributing and setup guide |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |

</details>

<details>
<summary>System Requirements</summary>

- Python 3.10+
- 8GB RAM (16GB recommended)
- NVIDIA GPU with CUDA (optional, for faster processing)
- Apple Silicon with MLX (optional, for hardware acceleration on Mac)

</details>

<details>
<summary>Optional: Install Chatterbox Engine</summary>

```bash
pip install -e ".[chatterbox]"
```

</details>

<details>
<summary>Optional: Install MLX Backend (Apple Silicon)</summary>

```bash
pip install -e ".[mlx]"
```

</details>

## Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) — XTTS v2 model
- [Chatterbox](https://github.com/resemble-ai/chatterbox) — Fast TTS by Resemble AI
- [PyTorch](https://pytorch.org/) — Deep learning framework
