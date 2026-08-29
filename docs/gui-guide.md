# GUI User Guide

VoiceCast includes a desktop GUI application for easy voice cloning without coding.

## Launching the Application

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Launch GUI
python voice_cloning_app.py
```

## Interface Overview

![VoiceCast GUI](../voicecast-app.png)

The interface includes a **Voice Cloning** tab for generation and a **Model Manager** tab for explicit model downloads. VoiceCast does not download models during installation or first generation.

The voice cloning tab is organized into sections:

```
+--------------------------------------------------+
|  VoiceCast                                       |
+--------------------------------------------------+
|  Reference Voice                                 |
|  [Browse...] /path/to/voice.wav                  |
+--------------------------------------------------+
|  TTS Engine                                      |
|  [Coqui XTTS v2     v]                          |
|                                                  |
|  Engine Parameters (varies by engine)            |
|  Language: [en v]  Temperature: [===|====]       |
|  Silence between chunks: [200 ms]                |
+--------------------------------------------------+
|  Text Input                                      |
|  +----------------------------------------------+|
|  | Enter text to convert to speech...           ||
|  |                                              ||
|  +----------------------------------------------+|
+--------------------------------------------------+
|  [Generate & Play]  [Save Audio]                 |
+--------------------------------------------------+
|  Status: Ready                                   |
+--------------------------------------------------+
```

## Step-by-Step Usage

### 1. Select Reference Voice

Click **Browse** to select your reference audio file:

- **Supported formats**: WAV, MP3
- **Recommended duration**: 5-30 seconds of clear speech
- **Quality tips**:
  - Use clean audio without background noise
  - Single speaker only
  - Clear pronunciation

### 2. Choose TTS Engine

Select from the dropdown:

| Engine | Best For |
|--------|----------|
| **Coqui XTTS v2** | Multilingual, high quality |
| **Chatterbox Turbo** | Fast generation, English |
| **Chatterbox Standard** | Higher quality, English |

### 3. Configure Engine Parameters

Parameters change based on selected engine:

**Coqui XTTS v2:**
- **Language**: Select target language (16 languages available)
- **Temperature**: Controls variation (0.1=consistent, 1.0=varied)

**Chatterbox Turbo/Standard:**
- **CFG Weight**: Text adherence (lower for fast speakers)
- **Exaggeration**: Expressiveness (higher for dramatic speech)

**All engines:**
- **Silence between chunks**: Pause inserted between long-text chunks in
  milliseconds (default: 200 ms; set to 0 for no pause)

### 4. Enter Text

Type or paste the text you want to convert:

```
Hello! This is a test of the voice cloning system.
It can convert any text to speech using your voice.
```

**For Chatterbox Turbo**, you can use paralinguistic tags:
```
That's amazing [laugh]! I can't believe it [gasp]!
```

Available tags: `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`, `[gasp]`, `[yawn]`

### 5. Download Required Model

Open the **Model Manager** tab before generation to:

1. View every supported model and whether it is installed
2. Download selected models with progress feedback
3. Keep downloads off the voice generation path

If you try to generate with a missing model, VoiceCast prompts you to open the Model Manager instead of downloading automatically.

### 6. Generate Speech

Click **Generate & Play** to:
1. Generate the audio (progress shown in status bar)
2. Automatically play the result

The generation runs in a background thread, so the UI remains responsive.

### 7. Save Audio

After generation, click **Save Audio** to:
1. Choose save location
2. Save as WAV file

## Engine-Specific Tips

### Coqui XTTS v2

**Pros:**
- 16 languages supported
- High voice similarity
- Good for long-form content

**Settings Guide:**
- **Temperature 0.3-0.5**: Consistent, professional output
- **Temperature 0.7-0.9**: More natural variation
- **Temperature 1.0**: Maximum creativity (may be unstable)

**Supported Languages:**
English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh), Japanese (ja), Hungarian (hu), Korean (ko)

### Chatterbox Turbo

**Pros:**
- Faster generation
- Paralinguistic tags for emotion
- Good pacing control

**Settings Guide:**
- **CFG Weight 0.2-0.4**: For fast speakers
- **CFG Weight 0.5**: Balanced (default)
- **CFG Weight 0.7-1.0**: Strong text adherence

- **Exaggeration 0.3**: Subtle expression
- **Exaggeration 0.5**: Natural (default)
- **Exaggeration 0.8+**: Dramatic, theatrical

### Chatterbox Standard

**Pros:**
- Higher quality than Turbo
- Better for professional use
- More consistent output

**When to Use:**
- Final production audio
- When quality matters more than speed
- Professional voice-over work

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Generate & Play |
| `Ctrl+S` | Save Audio |
| `Ctrl+O` | Browse for voice file |

## Workflow Tips

### Quick Testing

1. Use Chatterbox Turbo for rapid iteration
2. Once satisfied with text, switch to Coqui or Chatterbox Standard
3. Save final version

### Long Content

VoiceCast automatically uses the selected engine's chunk-size limit for long
text. Adjust **Silence between chunks** to control pacing at chunk boundaries;
short text is still synthesized in one call.

### Voice Matching

To get the best voice match:
1. Use 10-30 seconds of reference audio
2. Ensure reference matches your target style
3. Adjust temperature/exaggeration for fine-tuning

## Troubleshooting

### "No audio output"

1. Check system audio settings
2. Ensure speakers/headphones are connected
3. Try saving to file and playing externally

### "Generation failed"

1. Check reference voice file is valid audio
2. Ensure sufficient disk space for temp files
3. Check console output for detailed errors

### "Model not installed"

1. Open the **Model Manager** tab
2. Download the model for your selected engine
3. Return to **Voice Cloning** and generate again

### "Slow generation"

1. First model load can still take time after download
2. Use Chatterbox Turbo for speed
3. GPU (CUDA) significantly speeds up generation

### "Engine not available"

Install the required package:

```bash
# For Chatterbox engines
pip install -e ".[chatterbox]"
```

### GUI Not Responding

- Generation runs in background thread
- Wait for status bar to show "Complete"
- If frozen for >2 minutes, check console for errors

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Storage | 5 GB | 10 GB |
| GPU | - | NVIDIA with CUDA |
| Python | 3.10+ | 3.11 |

## See Also

- [API Reference](api-reference.md) - Programmatic usage
- [CLI Reference](cli-reference.md) - Command-line usage
- [Engines Guide](engines.md) - Engine comparison
- [Troubleshooting](troubleshooting.md) - Common issues
