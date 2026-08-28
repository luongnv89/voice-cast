# Development Guide

This guide covers setting up a development environment and contributing to VoiceCast.

## Prerequisites

- Python 3.10–3.12
- Git
- (Optional) NVIDIA GPU with CUDA for faster processing

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/luongnv89/voice-cast.git
cd voicecast
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
# Shared GUI/CLI dependencies and development tools
pip install -e ".[dev]"

# Install exactly one engine per virtual environment:
# Coqui XTTS v2 (Coqui TTS fork, Transformers 5.16)
pip install -e ".[coqui]"
# Or Chatterbox (Python 3.10–3.12, Transformers 5.2)
# pip install -e ".[chatterbox]"
```

The supported Coqui TTS fork and Chatterbox 0.1.7 require incompatible
Transformers versions, so do not combine these extras. Use separate virtual environments
when developing both engines. `requirements.txt` is the universal lock for the
shared core plus `dev`; it intentionally excludes the mutually exclusive
engine extras. Regenerate it with:

```bash
uv pip compile pyproject.toml --universal --extra dev -o requirements.txt
```

### 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This sets up automatic code quality checks before each commit.

## Project Structure

```
voicecast/
├── voice_cloner.py          # Core VoiceCloner class
├── tts_engine_base.py       # Abstract engine interface
├── tts_factory.py           # Engine factory pattern
├── vcloner.py               # CLI tool
├── voice_cloning_app.py     # GUI application
├── voice_cloning_app.py    # Main application entry point
├── engines/
│   ├── __init__.py
│   ├── coqui_engine.py      # Coqui XTTS v2 implementation
│   └── chatterbox_engine.py # Chatterbox implementation
├── gui/
│   ├── __init__.py
│   └── engine_controls.py   # Dynamic GUI controls
├── tests/
│   └── test_voice_cloner.py # Unit tests
├── docs/                    # Documentation
├── voice-samples/           # Sample reference voices
├── .github/workflows/       # CI/CD configuration
├── pyproject.toml          # Project configuration
└── .pre-commit-config.yaml # Pre-commit hooks
```

## Code Style

VoiceCast uses:

- **Ruff** for linting and formatting (replaces Black, isort, Flake8)
- **PEP 8** style guidelines
- **Type hints** for all public APIs

### Running Linters

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Style Guidelines

```python
# Type hints for function signatures
def generate(self, text: str, language: str = "en", **kwargs) -> tuple[np.ndarray, int]:
    """
    Generate audio from text.

    Args:
        text: The text to convert to speech.
        language: Language code (e.g., "en", "fr").
        **kwargs: Engine-specific parameters.

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    pass


# Use descriptive variable names
speaker_reference_path = "./voice-samples/speaker.wav"
audio_sample_rate = 22050

# Constants in UPPER_CASE
SUPPORTED_LANGUAGES = ["en", "es", "fr"]
DEFAULT_TEMPERATURE = 0.7
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_voice_cloner.py

# Run specific test
pytest tests/test_voice_cloner.py::test_initialization
```

### Coverage Baseline

Current project coverage (recorded 2025-01-XX): **34%** total. CI emits a per-run
report via `--cov-report=term-missing`. Target: increase incrementally with each
new test addition.

### Writing Tests

```python
# tests/test_my_feature.py
import pytest
from voice_cloner import VoiceCloner


class TestMyFeature:
    def test_basic_functionality(self, tmp_path):
        """Test basic feature behavior."""
        # Setup
        speaker_wav = "./voice-samples/test.wav"

        # Execute
        cloner = VoiceCloner(speaker_wav=speaker_wav)
        result = cloner.some_method()

        # Assert
        assert result is not None

    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(FileNotFoundError):
            VoiceCloner(speaker_wav="./nonexistent.wav")
```

### Test Fixtures

Common fixtures are in `tests/conftest.py`:

```python
import pytest


@pytest.fixture
def sample_voice_path():
    return "./voice-samples/test.wav"


@pytest.fixture
def cloner(sample_voice_path):
    return VoiceCloner(speaker_wav=sample_voice_path)
```

## Adding a New TTS Engine

### 1. Create Engine Class

```python
# engines/my_engine.py
import numpy as np
from tts_engine_base import TTSEngineBase


class MyEngine(TTSEngineBase):
    """TTS engine using My TTS Library."""

    SUPPORTED_LANGUAGES = ["en", "es"]

    def __init__(self, speaker_wav: str, device: str | None = None, my_param: float = 0.5):
        super().__init__(speaker_wav, device)
        self.my_param = my_param
        self._model = None  # Lazy loading

    @property
    def model(self):
        """Lazy load model on first use."""
        if self._model is None:
            from my_tts_library import Model

            self._model = Model.load(device=self.device)
        return self._model

    def generate(self, text: str, language: str = "en", my_param: float = None, **kwargs) -> tuple[np.ndarray, int]:
        param = my_param or self.my_param

        audio = self.model.synthesize(text=text, reference=self.speaker_wav, param=param)

        return audio.numpy().astype(np.float32), 22050

    def get_supported_parameters(self) -> dict:
        return {
            "my_param": {"type": float, "default": 0.5, "description": "My custom parameter", "min": 0.0, "max": 1.0}
        }

    @property
    def name(self) -> str:
        return "My TTS Engine"

    @property
    def supports_languages(self) -> list[str]:
        return self.SUPPORTED_LANGUAGES
```

### 2. Register Engine

Add to `tts_factory.py`:

```python
def _register_default_engines():
    # ... existing registrations ...

    # Register My Engine
    try:
        from engines.my_engine import MyEngine

        TTSFactory.register(name="my-engine", engine_class=MyEngine, display_name="My TTS Engine")
    except ImportError as e:
        logger.warning(f"My engine not available: {e}")
```

### 3. Add GUI Controls (Optional)

In `gui/engine_controls.py`:

```python
def _create_my_engine_controls(self):
    """Create controls for My Engine."""
    layout = QVBoxLayout()

    # My param slider
    param_layout = QHBoxLayout()
    param_layout.addWidget(QLabel("My Param:"))
    self.my_param_slider = QSlider(Qt.Horizontal)
    self.my_param_slider.setRange(0, 100)
    self.my_param_slider.setValue(50)
    param_layout.addWidget(self.my_param_slider)
    layout.addLayout(param_layout)

    return layout
```

### 4. Add Tests

```python
# tests/test_my_engine.py
import pytest
from engines.my_engine import MyEngine


class TestMyEngine:
    @pytest.fixture
    def engine(self):
        return MyEngine(speaker_wav="./voice-samples/test.wav")

    def test_generate(self, engine):
        audio, sr = engine.generate("Test text")
        assert audio is not None
        assert sr == 22050

    def test_supported_parameters(self, engine):
        params = engine.get_supported_parameters()
        assert "my_param" in params
```

### 5. Update Documentation

Add to `docs/engines.md` and update other relevant docs.

## CI/CD Pipeline

### GitHub Actions Workflow

The CI pipeline runs on every push and pull request:

1. **Lint & Format**: Ruff checks
2. **Security**: Bandit security scan
3. **Tests**: pytest on Python 3.10, 3.11, 3.12
4. **Dependency audit**: pip-audit against the pinned shared lockfile
4. **Pre-commit**: Validates all hooks pass

### Running CI Checks Locally

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific hooks
pre-commit run ruff --all-files
pre-commit run bandit --all-files
```

## Git Workflow

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring

### Commit Messages

Follow conventional commit format:

```
type(scope): description

feat(engine): add support for new TTS engine
fix(cli): handle missing output directory
docs(api): add examples for batch processing
refactor(factory): simplify engine registration
```

### Pull Request Process

1. Create feature branch from `main`
2. Make changes with tests
3. Ensure CI passes
4. Request review
5. Squash and merge

## Security Scanning

Bandit runs security checks:

```bash
# Run security scan
bandit -r . -x tests,.venv

# Check specific file
bandit voice_cloner.py
```

## Common Development Tasks

### Adding a Dependency

1. Add to `pyproject.toml`:
   ```toml
   dependencies = [
       "new-package>=1.0.0",
   ]
   ```
2. Reinstall: `pip install -e .`

### Updating Documentation

1. Edit files in `docs/`
2. Update cross-references
3. Test links work

### Releasing a New Version

1. Update version in `pyproject.toml`
2. Update CHANGELOG (if exists)
3. Create git tag: `git tag v0.3.0`
4. Push tag: `git push origin v0.3.0`

## Troubleshooting Development Issues

### Import Errors

```bash
# Reinstall in editable mode
pip install -e .
```

### Pre-commit Failures

```bash
# Update hooks
pre-commit autoupdate

# Clear cache
pre-commit clean
```

### Test Failures

```bash
# Run with verbose output
pytest -vvs

# Show locals on failure
pytest --showlocals
```

## See Also

- [Architecture](architecture.md) - System design
- [API Reference](api-reference.md) - API documentation
- [Contributing Guidelines](../CONTRIBUTING.md) - Contribution process
