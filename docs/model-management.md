# Model Management

VoiceCast separates Python package installation from model downloads. Installing the
project with `pip install -e .` installs code and dependencies only; model files are
not downloaded until you explicitly request them.

## Workflow

1. Install VoiceCast:

   ```bash
   pip install -e .
   ```

2. List models and local cache status:

   ```bash
   python vcloner.py --list-models
   ```

3. Download only the model(s) you need:

   ```bash
   python vcloner.py --download-models coqui-xtts-v2
   python vcloner.py --download-models chatterbox-turbo chatterbox-standard
   python vcloner.py --download-models --engine chatterbox
   ```

4. Generate audio with an installed model:

   ```bash
   python vcloner.py -i voice.wav -t "Hello world" -o output.wav --engine coqui
   ```

If a selected model is missing, VoiceCast raises a clear `ModelNotInstalledError`
with the command needed to download it. It does not silently download models during
normal generation.

## Available Models

| Model ID | Engine | Description |
|----------|--------|-------------|
| `coqui-xtts-v2` | `coqui` | Multilingual XTTS v2 voice cloning |
| `chatterbox-turbo` | `chatterbox` | Fast English voice cloning with expression tags |
| `chatterbox-standard` | `chatterbox` | Higher-quality English voice cloning |
| `mlx-kokoro` | `mlx-audio` | Apple Silicon preset voice TTS |
| `mlx-csm` | `mlx-audio` | Apple Silicon voice cloning |

MLX model downloads use the Hugging Face cache and require `huggingface_hub`.
Generation still requires the optional MLX backend on Apple Silicon.

## Python API

```python
from voice_cloner import VoiceCloner

# Discover models
for model in VoiceCloner.list_models():
    print(model.id, model.is_installed)

# Explicitly download a model
VoiceCloner.download_model("coqui-xtts-v2")

# Generate only after the model is installed
cloner = VoiceCloner(speaker_wav="voice.wav", engine="coqui")
cloner.say("Hello", save_audio=True, output_file="output.wav")

# Switch models/engines explicitly
cloner.switch_engine("chatterbox-turbo")
```

You can keep multiple `VoiceCloner` instances alive at the same time, each with a
different engine/model selection. Cache state is tracked per model ID.

See `examples/model_management.py` for a complete script.

## Cache Location

VoiceCast detects models in the backend-native caches used by each provider. The
registry, explicit downloaders, and generation backends all use these same default
locations so a downloaded model is the model generation sees:

- Coqui: platform-specific TTS cache (`~/.local/share/tts` on Linux,
  `~/Library/Application Support/tts` on macOS)
- Chatterbox and MLX: Hugging Face hub cache (`$HF_HOME/hub` or
  `~/.cache/huggingface/hub`)

## GUI

The desktop app includes a **Model Manager** tab. Use it to view installed status,
download selected models, and keep downloads off the main generation flow. Downloads
run in a background thread so the GUI stays responsive.
