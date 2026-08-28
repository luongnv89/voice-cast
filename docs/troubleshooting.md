# Troubleshooting

This guide covers common issues and their solutions when using VoiceCast.

## Installation Issues

### CUDA Not Available

**Symptoms:**
```
UserWarning: CUDA not available, using CPU
```

**Solutions:**

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   ```

2. **Install CUDA-enabled PyTorch:**
   ```bash
   pip uninstall torch torchaudio
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify CUDA in Python:**
   ```python
   import torch

   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Shows GPU name
   ```

### Missing Dependencies

**Symptoms:**
```
ModuleNotFoundError: No module named 'TTS'
```

**Solutions:**

```bash
# Shared dependencies
pip install -e .

# Install the extra for the engine you use (one per environment):
pip install -e ".[coqui]"
# or: pip install -e ".[chatterbox]"
# or: pip install -e ".[audio8]"
# or: pip install -e ".[mlx]"  # Apple Silicon only
```

### sounddevice Issues on Linux

**Symptoms:**
```
OSError: PortAudio library not found
```

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libportaudio2 portaudio19-dev

# Fedora
sudo dnf install portaudio portaudio-devel

# Then reinstall
pip install --force-reinstall sounddevice
```

### PySide6 Issues

**Symptoms:**
```
ImportError: cannot import name 'QtCore' from 'PySide6'
```

**Solution:**
```bash
# On Linux, install Qt dependencies
sudo apt-get install libxcb-cursor0 libxcb-xinerama0

# Reinstall PySide6
pip install --force-reinstall PySide6
```

---

## Runtime Issues

### Model Not Installed or Download Timeout

**Symptoms:**
```
ModelNotInstalledError: Model 'coqui-xtts-v2' for engine 'coqui' is not installed.
TimeoutError: Connection timed out while downloading model
```

**Solutions:**

1. **List model status:**
   ```bash
   python vcloner.py --list-models
   ```
2. **Download explicitly:**
   ```bash
   python vcloner.py --download-models coqui-xtts-v2
   python vcloner.py --download-models --engine chatterbox
   ```
3. **Check internet connection**
4. **Set longer timeout for Hugging Face downloads:**
   ```bash
   HF_HUB_DOWNLOAD_TIMEOUT=300 python vcloner.py --download-models chatterbox-turbo
   ```
5. **Check cache location**: Chatterbox and MLX models are stored in `~/.cache/huggingface/hub` by default.

### Out of Memory (GPU)

**Symptoms:**
```
CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**

1. **Force CPU usage:**
   ```python
   cloner = VoiceCloner(speaker_wav="speaker.wav", device="cpu")
   ```

2. **Clear GPU memory:**
   ```python
   import torch

   torch.cuda.empty_cache()
   ```

3. **Close other GPU applications**

4. **Use smaller model:**
   ```python
   # Chatterbox Turbo uses less memory
   cloner = VoiceCloner(speaker_wav="speaker.wav", engine="chatterbox-turbo")
   ```

### Out of Memory (CPU/RAM)

**Symptoms:**
- Process killed
- System becomes unresponsive

**Solutions:**

1. **Close other applications**
2. **Process shorter texts**
3. **Add swap space (Linux):**
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

## Audio Issues

### No Audio Playback

**Symptoms:**
- Audio generates but doesn't play
- No sound from speakers

**Solutions:**

1. **Check system audio:**
   - Ensure speakers/headphones connected
   - Volume not muted
   - Correct output device selected

2. **Test sounddevice:**
   ```python
   import sounddevice as sd
   import numpy as np

   # Generate test tone
   fs = 44100
   duration = 1
   t = np.linspace(0, duration, int(fs * duration))
   tone = np.sin(2 * np.pi * 440 * t)
   sd.play(tone, fs)
   sd.wait()
   ```

3. **Check audio device:**
   ```python
   import sounddevice as sd

   print(sd.query_devices())
   ```

4. **Set specific device:**
   ```python
   import sounddevice as sd

   sd.default.device = "Your Device Name"
   ```

### Audio Quality Issues

**Problem: Robotic/Distorted Audio**

Solutions:
- Increase temperature (Coqui): `temperature=0.8`
- Adjust exaggeration (Chatterbox): `exaggeration=0.6`
- Use higher quality reference audio

**Problem: Wrong Speed/Pitch**

Solutions:
- Adjust speed parameter: `cloner.say(text, speed=1.0)`
- Ensure reference audio sample rate is standard (22050 or 44100 Hz)

**Problem: Inconsistent Voice**

Solutions:
- Use longer reference audio (15-30 seconds)
- Lower temperature for consistency: `temperature=0.5`
- Use same language for reference and output

---

## File Issues

### Speaker File Not Found

**Symptoms:**
```
FileNotFoundError: Speaker reference file not found: ./speaker.wav
```

**Solutions:**

1. **Check file exists:**
   ```bash
   ls -la ./voice-samples/
   ```

2. **Use absolute path:**
   ```python
   import os

   speaker_path = os.path.abspath("./voice-samples/speaker.wav")
   cloner = VoiceCloner(speaker_wav=speaker_path)
   ```

3. **Check file permissions:**
   ```bash
   chmod 644 ./voice-samples/speaker.wav
   ```

### Invalid Audio Format

**Symptoms:**
```
RuntimeError: Error loading audio file
```

**Solutions:**

1. **Convert to WAV:**
   ```bash
   ffmpeg -i input.mp3 -ar 22050 output.wav
   ```

2. **Check audio file:**
   ```python
   import soundfile as sf

   data, sr = sf.read("speaker.wav")
   print(f"Duration: {len(data) / sr:.2f}s, Sample rate: {sr}")
   ```

3. **Ensure mono or stereo (not multichannel)**

### Output Directory Doesn't Exist

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: './outputs/audio.wav'
```

**Solution:**
Directories are created automatically, but verify parent exists:
```python
import os

os.makedirs("./outputs", exist_ok=True)
cloner.say(text, save_audio=True, output_file="./outputs/audio.wav")
```

---

## Engine-Specific Issues

### Coqui: Unsupported Language

**Symptoms:**
```
ValueError: Language 'xx' not supported
```

**Solution:**
Check supported languages:
```python
cloner = VoiceCloner(speaker_wav="speaker.wav", engine="coqui")
print(cloner.get_supported_languages())
# ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh', 'ja', 'hu', 'ko']
```

### Chatterbox: Tags Not Working

**Symptoms:**
- `[laugh]` appears in output as spoken text
- No emotional expression

**Solutions:**

1. **Ensure using Turbo variant:**
   ```python
   cloner = VoiceCloner(speaker_wav="speaker.wav", engine="chatterbox-turbo")
   # NOT chatterbox-standard - tags only work in Turbo
   ```

2. **Check tag spelling:**
   Valid: `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`, `[gasp]`, `[yawn]`

### Chatterbox: Not Installed

**Symptoms:**
```
ImportError: chatterbox-tts package required
```

**Solution:**
```bash
pip install -e ".[chatterbox]"
```

---

## GUI Issues

### GUI Won't Start

**Symptoms:**
- Window doesn't appear
- Immediate crash

**Solutions:**

1. **Check Qt dependencies (Linux):**
   ```bash
   sudo apt-get install libxcb-xinerama0 libxcb-cursor0
   ```

2. **Run from terminal to see errors:**
   ```bash
   python voice_cloning_app.py
   ```

3. **Reset Qt cache:**
   ```bash
   rm -rf ~/.cache/qt*
   ```

### GUI Freezes During Generation

**Symptoms:**
- Interface becomes unresponsive
- "Not Responding" in window title

**Note:** This shouldn't happen as generation runs in a background thread. If it occurs:

1. **Wait** - First-time model loading takes 30-60 seconds
2. **Check console** for error messages
3. **Verify system resources** (RAM, GPU memory)

### Controls Don't Appear

**Symptoms:**
- Engine dropdown changes but no parameters show

**Solution:**
This is a bug - report it with:
- Python version
- PySide6 version
- Operating system

---

## CLI Issues

### Arguments Not Recognized

**Symptoms:**
```
error: unrecognized arguments: --some-option
```

**Solution:**
Check available arguments:
```bash
python vcloner.py --help
```

### Rich Output Garbled

**Symptoms:**
- Strange characters in terminal
- Colors not displaying

**Solution:**
```bash
# Disable rich formatting
NO_COLOR=1 python vcloner.py ...
```

---

## Performance Issues

### Slow First Run

**Expected behavior**: Installation and generation do not download models automatically. Explicit model downloads can be large (500 MB-2 GB), and cached models make later generation faster.

**Speed up:**
- Use wired internet connection for explicit downloads
- Pre-download models with `python vcloner.py --download-models <model-id>`

### Slow Generation

**Solutions:**

1. **Use GPU:**
   ```python
   cloner = VoiceCloner(speaker_wav="speaker.wav", device="cuda")
   ```

2. **Use Chatterbox Turbo** for faster generation:
   ```python
   cloner = VoiceCloner(speaker_wav="speaker.wav", engine="chatterbox-turbo")
   ```

3. **Process shorter texts** - break long content into sentences

### High Memory Usage

**Solutions:**

1. **Clear Python garbage:**
   ```python
   import gc

   gc.collect()
   ```

2. **Use single engine instance** - don't create multiple VoiceCloner objects

3. **Reduce batch size** - process one text at a time

---

## Getting Help

If your issue isn't covered here:

1. **Check existing issues:**
   https://github.com/luongnv89/voice-cast/issues

2. **Create new issue** with:
   - Python version: `python --version`
   - Package versions: `pip list | grep -E "torch|TTS|chatterbox"`
   - Operating system
   - Complete error message
   - Steps to reproduce

3. **Include system info:**
   ```python
   import torch
   import platform

   print(f"Python: {platform.python_version()}")
   print(f"PyTorch: {torch.__version__}")
   print(f"CUDA: {torch.cuda.is_available()}")
   print(f"OS: {platform.system()} {platform.release()}")
   ```

## See Also

- [Development Guide](development.md) - For debugging and contributing
- [Engines Guide](engines.md) - Engine-specific details
- [Architecture](architecture.md) - System design for understanding errors
