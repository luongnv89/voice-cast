# API Reference

This document provides the complete API reference for using VoiceCloner programmatically.

## VoiceCloner Class

The main class for voice cloning operations.

### Import

```python
from voice_cloner import VoiceCloner
```

### Constructor

```python
VoiceCloner(
    speaker_wav: str,
    engine: str | TTSEngineBase | None = None,
    device: str | None = None,
    auto_download: bool = False,
    **engine_kwargs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speaker_wav` | `str` | Required | Path to speaker reference audio file (WAV/MP3) |
| `engine` | `str \| TTSEngineBase` | `"coqui"` | Engine name or custom engine instance |
| `device` | `str` | Auto-detect | `"cuda"` for GPU or `"cpu"` |
| `auto_download` | `bool` | `False` | Explicit opt-in for backend downloads during generation. Leave `False` to require pre-downloaded models. |
| `**engine_kwargs` | `dict` | `{}` | Engine-specific constructor parameters |

**Available Engines:**
- `"coqui"` - Coqui XTTS v2 (default, multilingual)
- `"chatterbox-turbo"` - Chatterbox Turbo (fast, paralinguistic tags)
- `"chatterbox-standard"` - Chatterbox Standard (higher quality)

**Example:**

```python
# Basic initialization with default Coqui engine
cloner = VoiceCloner(speaker_wav="./voice-samples/speaker.wav")

# Using Chatterbox Turbo
cloner = VoiceCloner(speaker_wav="./voice-samples/speaker.wav", engine="chatterbox-turbo")

# Force CPU usage
cloner = VoiceCloner(speaker_wav="./voice-samples/speaker.wav", device="cpu")
```

**Model download behavior:** VoiceCast does not download model files during normal initialization or generation. Use `VoiceCloner.download_model()` or `python vcloner.py --download-models <model-id>` first. Missing models raise `ModelNotInstalledError`.

**Raises:**
- `ValueError` - If the engine requires reference audio but `speaker_wav` is empty or not provided (fails fast, before any engine is created)
- `FileNotFoundError` - If speaker reference file doesn't exist
- `ValueError` - If unknown engine name is provided
- `ModelNotInstalledError` - If generation needs a model that has not been downloaded

---

### Factory Methods

#### `from_coqui()`

```python
@classmethod
VoiceCloner.from_coqui(
    speaker_wav: str,
    device: str | None = None,
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
) -> VoiceCloner
```

Create a VoiceCloner with Coqui TTS engine.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speaker_wav` | `str` | Required | Path to speaker reference audio |
| `device` | `str` | Auto-detect | Device to use |
| `model_name` | `str` | XTTS v2 | Coqui model identifier |

**Example:**

```python
cloner = VoiceCloner.from_coqui(speaker_wav="./voice-samples/speaker.wav", device="cuda")
```

---

#### `from_chatterbox()`

```python
@classmethod
VoiceCloner.from_chatterbox(
    speaker_wav: str,
    variant: str = "turbo",
    device: str | None = None
) -> VoiceCloner
```

Create a VoiceCloner with Chatterbox TTS engine.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `speaker_wav` | `str` | Required | Path to speaker reference (~10 seconds recommended) |
| `variant` | `str` | `"turbo"` | `"turbo"` (350M, fast) or `"standard"` (500M, quality) |
| `device` | `str` | Auto-detect | Device to use |

**Example:**

```python
# Fast Turbo variant
cloner = VoiceCloner.from_chatterbox(speaker_wav="./voice-samples/speaker.wav", variant="turbo")

# Higher quality Standard variant
cloner = VoiceCloner.from_chatterbox(speaker_wav="./voice-samples/speaker.wav", variant="standard")
```

---

### Methods

#### `generate()`

```python
generate(
    text: str,
    language: str = "en",
    chunk_size: int | None = None,
    silence_duration: int = 200,
    output_file: str | None = None,
    **kwargs
) -> str
```

Synthesize text, save the resulting WAV file, and return its path. This method
never plays audio. When `chunk_size` is set and the text is longer than the
limit, sentences are synthesized separately and joined with `silence_duration`
milliseconds of zero-filled audio between chunks. When it is omitted, the
configured engine's `MAX_CHUNK_CHARS` value is used; an engine that inherits
the base value of `0` has no automatic limit. The default padding is 200 ms,
and padding is never added before the first or after the last chunk.

`chunk_size` must be `None` or a positive integer. An explicit value overrides
the engine default for that call. `silence_duration` must be a
non-negative integer in milliseconds. Text at or below the effective limit is
sent to the engine once without modification.

---

#### `say()`

```python
say(
    text_to_voice: str,
    language: str = "en",
    play_audio: bool = True,
    save_audio: bool = False,
    output_file: str | None = None,
    speed: float = 1.0,
    chunk_size: int | None = None,
    silence_duration: int = 200,
    **kwargs
) -> str | None
```

Convert text to speech using the configured engine.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_to_voice` | `str` | Required | Text to synthesize |
| `language` | `str` | `"en"` | Language code (see [Engines](engines.md)) |
| `play_audio` | `bool` | `True` | Play audio after generation |
| `save_audio` | `bool` | `False` | Save audio to file |
| `output_file` | `str` | Auto-generated | Output file path |
| `speed` | `float` | `1.0` | Playback speed multiplier |
| `chunk_size` | `int \| None` | `None` | Maximum characters per synthesis chunk. `None` uses the engine's `MAX_CHUNK_CHARS`; a positive value overrides it for this call |
| `silence_duration` | `int` | `200` | Silence between consecutive chunks, in **milliseconds**. Chunked path only; `0` inserts no silence; negative values are rejected |
| `**kwargs` | `dict` | `{}` | Engine-specific parameters |

**Returns:** the path to the written WAV file when `save_audio=True`, otherwise `None`.

**Chunked synthesis:** when an effective `chunk_size` is set and the text exceeds
it, the text is split on sentence boundaries (`utils.split_into_chunks`), each
chunk is synthesized separately, and the results are concatenated with
`silence_duration` ms of silence between consecutive chunks — no leading or
trailing padding. Negative `silence_duration` values are rejected. Text at or
below the effective limit is synthesized in a single engine call with the
original unmodified string, identical to the behavior without chunking. The
following conservative defaults are declared by the built-in engines:

| Engine | `MAX_CHUNK_CHARS` |
|--------|-------------------:|
| Coqui XTTS v2 | 240 |
| Chatterbox Turbo/Standard | 100 |
| MLX Kokoro/CSM | 200 |
| Audio8 | 200 |

A `RuntimeError` is raised if the engine returns
different sample rates for different chunks.

**Engine-specific kwargs:**

*For Coqui:*
- `temperature` (float, 0.1-1.0): Sampling temperature. Default: 0.7
- `gpt_cond_len` (int, 32-256): GPT conditioning length. Default: 128

*For Chatterbox:*
- `cfg_weight` (float, 0.0-1.0): Text adherence. Lower for fast speakers. Default: 0.5
- `exaggeration` (float, 0.0-1.5): Expressiveness. Higher = more dramatic. Default: 0.5

**Examples:**

```python
# Basic usage - generate and play
cloner.say("Hello world!")

# Save to file
cloner.say("This is a test.", save_audio=True, output_file="output.wav")

# Play at 1.5x speed without saving
cloner.say("Speed up the playback.", speed=1.5, play_audio=True, save_audio=False)

# French with Coqui engine
cloner.say("Bonjour le monde!", language="fr", temperature=0.8)

# Chatterbox with paralinguistic tag
chatter_cloner.say("That's hilarious [laugh]!", cfg_weight=0.3, exaggeration=0.7)
```

---

#### `get_engine_parameters()`

```python
get_engine_parameters() -> dict[str, dict]
```

Get supported parameters for the current engine.

**Returns:**

Dictionary mapping parameter names to their metadata:

```python
{
    "param_name": {
        "type": <type>,
        "default": <value>,
        "description": "...",
        "min": <value>,  # optional
        "max": <value>,  # optional
        "options": [...]  # optional, for enums
    }
}
```

**Example:**

```python
params = cloner.get_engine_parameters()
for name, info in params.items():
    print(f"{name}: {info['description']} (default: {info['default']})")
```

---

#### `get_supported_languages()`

```python
get_supported_languages() -> list[str]
```

Get supported languages for the current engine.

**Returns:** List of language codes (e.g., `["en", "es", "fr"]`)

**Example:**

```python
languages = cloner.get_supported_languages()
print(f"Supported: {', '.join(languages)}")
# Output: Supported: en, es, fr, de, it, pt, ...
```

---

#### `switch_engine()`

```python
switch_engine(engine: str, **engine_kwargs) -> None
```

Switch an existing `VoiceCloner` instance to another engine/model without implicit downloads.

**Example:**

```python
cloner.switch_engine("chatterbox-turbo")
```

---

#### `list_models()` (static)

```python
@staticmethod
list_models() -> list[ModelInfo]
```

List registered models with current cache status.

---

#### `download_model()` (static)

```python
@staticmethod
download_model(model_id: str, progress_callback: ProgressCallback | None = None) -> Path
```

Explicitly download a model and return the local cache path.

---

#### `is_model_installed()` (static)

```python
@staticmethod
is_model_installed(model_id: str) -> bool
```

Return whether a model is available in the local cache.

---

#### `get_model_id_for_engine()` (static)

```python
@staticmethod
get_model_id_for_engine(engine_name: str) -> str
```

Map a public engine name such as `chatterbox-turbo` to its model ID.

---

#### `available_engines()` (static)

```python
@staticmethod
available_engines() -> dict[str, str]
```

Get dictionary of available TTS engines.

**Returns:** Dictionary mapping engine IDs to display names

**Example:**

```python
engines = VoiceCloner.available_engines()
for engine_id, display_name in engines.items():
    print(f"{engine_id}: {display_name}")
# Output:
# coqui: Coqui XTTS v2
# chatterbox-turbo: Chatterbox Turbo (350M)
# chatterbox-standard: Chatterbox Standard (500M)
```

---

## TTSFactory Class

Factory for creating and managing TTS engine instances.

### Import

```python
from tts_factory import TTSFactory
```

### Methods

#### `create()`

```python
@classmethod
TTSFactory.create(
    engine_name: str,
    speaker_wav: str,
    device: str | None = None,
    **engine_kwargs
) -> TTSEngineBase
```

Create an engine instance directly.

**Example:**

```python
engine = TTSFactory.create(engine_name="chatterbox-turbo", speaker_wav="./speaker.wav", device="cuda")
```

---

#### `available_engines()`

```python
@classmethod
TTSFactory.available_engines() -> list[str]
```

Get list of registered engine names.

---

#### `is_available()`

```python
@classmethod
TTSFactory.is_available(engine_name: str) -> bool
```

Check if an engine's dependencies are installed.

**Example:**

```python
if TTSFactory.is_available("chatterbox-turbo"):
    cloner = VoiceCloner(speaker_wav="./speaker.wav", engine="chatterbox-turbo")
else:
    print('Install the Chatterbox extra: pip install -e ".[chatterbox]"')
```

---

## TTSEngineBase Class

Abstract base class for TTS engines. Extend this to create custom engines.

### Import

```python
from tts_engine_base import TTSEngineBase
```

### Abstract Methods

```python
@abstractmethod
def generate(self, text: str, language: str = "en", **kwargs) -> tuple[np.ndarray, int]:
    """Generate audio from text. Returns (audio_data, sample_rate)."""


@abstractmethod
def get_supported_parameters(self) -> dict[str, dict]:
    """Return supported parameters with metadata."""


@property
@abstractmethod
def name(self) -> str:
    """Human-readable engine name."""


@property
@abstractmethod
def supports_languages(self) -> list[str]:
    """List of supported language codes."""
```

### Creating Custom Engines

```python
import numpy as np
from tts_engine_base import TTSEngineBase


class MyCustomEngine(TTSEngineBase):
    def generate(self, text, language="en", my_param=0.5, **kwargs):
        # Your TTS implementation
        audio_data = np.zeros(16000)  # 1 second of silence
        sample_rate = 16000
        return audio_data.astype(np.float32), sample_rate

    def get_supported_parameters(self):
        return {
            "my_param": {"type": float, "default": 0.5, "description": "My custom parameter", "min": 0.0, "max": 1.0}
        }

    @property
    def name(self):
        return "My Custom Engine"

    @property
    def supports_languages(self):
        return ["en"]


# Register the engine
from tts_factory import TTSFactory

TTSFactory.register(name="my-engine", engine_class=MyCustomEngine, display_name="My Custom Engine")

# Use it
cloner = VoiceCloner(speaker_wav="./speaker.wav", engine="my-engine")
```

---

## Complete Examples

### Basic Text-to-Speech

```python
from voice_cloner import VoiceCloner

# Initialize with reference voice
cloner = VoiceCloner(speaker_wav="./voice-samples/speaker.wav")

# Generate and play speech
cloner.say("Hello! This is a test of the voice cloning system.")
```

### Batch Processing

```python
from voice_cloner import VoiceCloner

cloner = VoiceCloner(speaker_wav="./voice-samples/speaker.wav")

texts = ["First sentence to convert.", "Second sentence to convert.", "Third sentence to convert."]

for i, text in enumerate(texts):
    cloner.say(text, play_audio=False, save_audio=True, output_file=f"output_{i + 1}.wav")
    print(f"Generated: output_{i + 1}.wav")
```

### Multilingual with Coqui

```python
from voice_cloner import VoiceCloner

cloner = VoiceCloner.from_coqui(speaker_wav="./speaker.wav")

# English
cloner.say("Hello, how are you?", language="en")

# Spanish
cloner.say("Hola, cmo ests?", language="es")

# French
cloner.say("Bonjour, comment allez-vous?", language="fr")

# German
cloner.say("Hallo, wie geht es Ihnen?", language="de")
```

### Expressive Speech with Chatterbox

```python
from voice_cloner import VoiceCloner

cloner = VoiceCloner.from_chatterbox(speaker_wav="./speaker.wav", variant="turbo")

# Use paralinguistic tags for expression
cloner.say("That's the funniest thing I've ever heard [laugh]!", cfg_weight=0.3, exaggeration=0.8)

cloner.say("I'm so exhausted [sigh]... it's been a long day.", cfg_weight=0.5, exaggeration=0.6)
```

### Engine Comparison

```python
from voice_cloner import VoiceCloner

text = "This is a comparison of different TTS engines."
speaker = "./voice-samples/speaker.wav"

# Coqui XTTS v2
coqui = VoiceCloner(speaker_wav=speaker, engine="coqui")
coqui.say(text, save_audio=True, output_file="coqui_output.wav", play_audio=False)

# Chatterbox Turbo (fast)
turbo = VoiceCloner(speaker_wav=speaker, engine="chatterbox-turbo")
turbo.say(text, save_audio=True, output_file="turbo_output.wav", play_audio=False)

# Chatterbox Standard (quality)
standard = VoiceCloner(speaker_wav=speaker, engine="chatterbox-standard")
standard.say(text, save_audio=True, output_file="standard_output.wav", play_audio=False)
```

---

## Error Handling

```python
from voice_cloner import VoiceCloner

try:
    cloner = VoiceCloner(speaker_wav="./nonexistent.wav")
except FileNotFoundError as e:
    print(f"Speaker file not found: {e}")

try:
    cloner = VoiceCloner(speaker_wav="./speaker.wav", engine="unknown-engine")
except ValueError as e:
    print(f"Invalid engine: {e}")

try:
    cloner.say("Test", language="invalid")
except Exception as e:
    print(f"Generation error: {e}")
```

---

## See Also

- [Engines Guide](engines.md) - Detailed comparison of TTS engines
- [CLI Reference](cli-reference.md) - Command-line interface
- [Architecture](architecture.md) - System architecture and design
