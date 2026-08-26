# Architecture

This document describes the architecture and design patterns used in VoiceCast.

## Overview

VoiceCast follows a layered architecture with clear separation of concerns:

```mermaid
graph TB
    subgraph "User Interfaces"
        GUI[GUI Application<br/>voice_cloning_app.py]
        CLI[CLI Tool<br/>vcloner.py]
        API[Python API<br/>voice_cloner.py]
    end

    subgraph "Core Layer"
        VC[VoiceCloner Class]
        TF[TTSFactory]
        MR[ModelRegistry]
        MD[ModelDownloader]
    end

    subgraph "Engine Layer"
        BASE[TTSEngineBase<br/>Abstract Interface]
        COQUI[CoquiEngine<br/>XTTS v2]
        CHAT[ChatterboxEngine<br/>Turbo/Standard]
    end

    subgraph "External Libraries"
        TTS[Coqui TTS]
        CB[Chatterbox TTS]
        SD[sounddevice]
        SF[soundfile]
    end

    GUI --> VC
    CLI --> VC
    API --> VC
    VC --> TF
    GUI --> MR
    CLI --> MR
    CLI --> MD
    TF --> BASE
    BASE --> COQUI
    BASE --> CHAT
    COQUI --> TTS
    CHAT --> CB
    VC --> SD
    VC --> SF
```

## Components

### User Interface Layer

#### GUI Application (`voice_cloning_app.py`)
- Built with PySide6 (Qt bindings)
- Non-blocking TTS generation using `QThread`
- Dynamic engine-specific controls via `gui/engine_controls.py`
- Features: voice file selection, text input, playback, and save

#### CLI Tool (`vcloner.py`)
- Argument parsing with `argparse`
- Rich console output for enhanced terminal UX
- Supports all engine-specific parameters
- Engine listing with availability status

#### Python API (`voice_cloner.py`)
- Primary programmatic interface
- Factory methods: `from_coqui()`, `from_chatterbox()`
- Model-management helpers: `list_models()`, `download_model()`, `is_model_installed()`, `switch_engine()`
- Audio playback and file saving built-in

### Core Layer

#### VoiceCloner Class

The main entry point for all TTS operations:

```mermaid
classDiagram
    class VoiceCloner {
        +speaker_wav: str
        +engine: TTSEngineBase
        +engine_name: str
        +__init__(speaker_wav, engine, device, **kwargs)
        +say(text, language, play_audio, save_audio, speed, **kwargs)
        +get_engine_parameters(): dict
        +get_supported_languages(): list
        +available_engines(): dict
        +from_coqui(speaker_wav, device, model_name): VoiceCloner
        +from_chatterbox(speaker_wav, variant, device): VoiceCloner
    }
```

#### ModelRegistry and ModelDownloader

Model management is explicit and separate from generation:

- `ModelRegistry` stores model metadata, provider/engine mapping, cache locations, and installation status.
- `ModelDownloader` routes explicit downloads to provider-specific adapters.
- Engines check model availability before lazy loading and raise `ModelNotInstalledError` when a model is missing.
- Normal install, API initialization, CLI generation, and GUI generation do not download model files implicitly.

#### TTSFactory

Factory pattern for engine instantiation:

```mermaid
classDiagram
    class TTSFactory {
        -_registry: dict
        -_display_names: dict
        +register(name, engine_class, display_name, **kwargs)
        +create(engine_name, speaker_wav, device, **kwargs): TTSEngineBase
        +available_engines(): list
        +get_display_name(engine_name): str
        +get_engine_info(): dict
        +is_available(engine_name): bool
    }
```

### Engine Layer

#### TTSEngineBase (Abstract)

Defines the interface all TTS engines must implement:

```mermaid
classDiagram
    class TTSEngineBase {
        <<abstract>>
        +speaker_wav: str
        +device: str
        +generate(text, language, **kwargs): tuple[ndarray, int]*
        +get_supported_parameters(): dict*
        +name: str*
        +supports_languages: list*
    }

    class CoquiEngine {
        +SUPPORTED_LANGUAGES: list
        +model_name: str
        +generate(text, language, temperature, gpt_cond_len)
    }

    class ChatterboxEngine {
        +PARALINGUISTIC_TAGS: list
        +variant: str
        +generate(text, language, cfg_weight, exaggeration)
        +validate_text(text): tuple
    }

    TTSEngineBase <|-- CoquiEngine
    TTSEngineBase <|-- ChatterboxEngine
```

## Design Patterns

### Factory Pattern

The `TTSFactory` class implements the Factory pattern for engine creation:

- **Registration**: Engines self-register on module import
- **Creation**: Factory creates appropriate engine based on name
- **Extensibility**: New engines can be added without modifying core code

### Template Method Pattern

`TTSEngineBase` defines the template for TTS operations:

- Abstract methods define what subclasses must implement
- Common functionality (device detection) in base class
- Subclasses customize generation logic

### Explicit Model Management + Lazy Loading

Models are downloaded only through explicit model-management commands/API calls.
Engines still use lazy loading to avoid loading large model weights until first use:

```python
@property
def model(self):
    if self._model is None:
        # Load heavy model only when first needed
        self._model = load_model()
    return self._model
```

This reduces startup time and memory usage when engines aren't used while keeping network downloads user-controlled.

## Data Flow

### Text-to-Speech Generation

```mermaid
sequenceDiagram
    participant User
    participant VoiceCloner
    participant TTSFactory
    participant Engine
    participant Model

    User->>VoiceCloner: say("Hello world")
    VoiceCloner->>Engine: generate(text, language, **kwargs)
    Engine->>Model: Lazy load if needed
    Model-->>Engine: Model ready
    Engine->>Model: Generate audio
    Model-->>Engine: audio_data, sample_rate
    Engine-->>VoiceCloner: audio_data, sample_rate
    VoiceCloner->>VoiceCloner: Save file (optional)
    VoiceCloner->>VoiceCloner: Play audio (optional)
    VoiceCloner-->>User: Complete
```

## Threading Model

The GUI application uses Qt's threading model:

```mermaid
graph LR
    subgraph "Main Thread"
        UI[UI Controls]
        SIG[Qt Signals]
    end

    subgraph "Worker Thread"
        GEN[TTS Generation]
        PROC[Audio Processing]
    end

    UI -->|Start Generation| GEN
    GEN -->|finished signal| SIG
    SIG -->|Update UI| UI
```

- **Main Thread**: UI updates, user interaction
- **Worker Thread**: Heavy TTS generation
- **Signals**: Thread-safe communication via Qt signals

## File Organization

```
voicecast/
├── voice_cloner.py          # Core VoiceCloner class
├── tts_engine_base.py       # Abstract engine interface
├── tts_factory.py           # Engine factory
├── vcloner.py               # CLI tool
├── voice_cloning_app.py     # GUI application
├── voice_cloning_app.py    # Main application entry point
├── engines/
│   ├── __init__.py
│   ├── coqui_engine.py      # Coqui XTTS v2 engine
│   └── chatterbox_engine.py # Chatterbox engine
├── gui/
│   ├── __init__.py
│   ├── engine_controls.py   # Dynamic UI controls
│   └── model_manager_widget.py # GUI model management
├── models/
│   ├── model_registry.py    # Model metadata and cache detection
│   ├── model_downloader.py  # Explicit download orchestration
│   └── downloaders/         # Provider-specific download adapters
├── tests/
│   └── test_voice_cloner.py # Unit tests
├── voice-samples/           # Reference voice files
└── docs/                    # Documentation
```

## Extending the Architecture

### Adding a New TTS Engine

1. Create engine class extending `TTSEngineBase`:

```python
from tts_engine_base import TTSEngineBase


class MyEngine(TTSEngineBase):
    def generate(self, text, language="en", **kwargs):
        # Implementation
        return audio_data, sample_rate

    def get_supported_parameters(self):
        return {"my_param": {"type": float, "default": 0.5}}

    @property
    def name(self):
        return "My Custom Engine"

    @property
    def supports_languages(self):
        return ["en", "es"]
```

2. Register in `tts_factory.py`:

```python
from engines.my_engine import MyEngine

TTSFactory.register(name="my-engine", engine_class=MyEngine, display_name="My Custom Engine")
```

3. Add GUI controls in `gui/engine_controls.py` if needed.

## Performance Considerations

| Aspect | Impact | Mitigation |
|--------|--------|------------|
| Model Loading | 10-30s on first use | Lazy loading |
| Generation Time | 1-5s per sentence | Background threading (GUI) |
| Memory Usage | 2-8GB depending on model | Single engine instance |
| GPU Memory | 2-4GB VRAM | CPU fallback supported |
