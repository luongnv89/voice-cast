# VoiceCast Documentation

Welcome to the VoiceCast documentation. This guide covers everything from basic usage to extending the system with custom TTS engines.

## Getting Started

If you're new to VoiceCast, start here:

1. **[Quick Start](../README.md)** - Installation and basic usage
2. **[GUI Guide](gui-guide.md)** - Desktop application walkthrough
3. **[Engines Guide](engines.md)** - Choose the right TTS engine

## User Guides

| Guide | Description |
|-------|-------------|
| [GUI Guide](gui-guide.md) | Step-by-step desktop application usage |
| [CLI Reference](cli-reference.md) | Command-line interface documentation |
| [API Reference](api-reference.md) | Complete Python API reference |
| [Engines Guide](engines.md) | TTS engine comparison and parameters |
| [Model Management](model-management.md) | Explicit model download, cache, CLI, GUI, and API workflow |

## Technical Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System design with diagrams |
| [Model Management](model-management.md) | Model registry, downloader, and cache behavior |
| [Development](development.md) | Contributing and extending |
| [Troubleshooting](troubleshooting.md) | Common issues and fixes |

## Quick Links

### By Use Case

- **I want to clone a voice quickly** → [GUI Guide](gui-guide.md)
- **I need to batch process files** → [CLI Reference](cli-reference.md)
- **I'm integrating into my app** → [API Reference](api-reference.md)
- **I need to install or switch models** → [Model Management](model-management.md)
- **I want multilingual support** → [Engines Guide](engines.md#coqui-xtts-v2)
- **I need expressive speech** → [Engines Guide](engines.md#chatterbox-turbo)

### By Role

- **End User** → [GUI Guide](gui-guide.md), [Troubleshooting](troubleshooting.md)
- **Developer** → [API Reference](api-reference.md), [Architecture](architecture.md)
- **Contributor** → [Development](development.md), [Architecture](architecture.md)

## Document Map

```mermaid
graph LR
    README[README.md] --> INDEX[docs/index.md]
    INDEX --> GUI[GUI Guide]
    INDEX --> CLI[CLI Reference]
    INDEX --> API[API Reference]
    INDEX --> ENG[Engines Guide]
    INDEX --> ARCH[Architecture]
    INDEX --> DEV[Development]
    INDEX --> TROUBLE[Troubleshooting]

    GUI --> ENG
    CLI --> ENG
    API --> ENG
    API --> ARCH
    DEV --> ARCH
```

## Version

This documentation is for VoiceCast v0.2.0.
