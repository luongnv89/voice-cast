# Convert to Standard WAV

Utility script for converting audio files to standard WAV format (16-bit PCM,
22050 Hz, mono). Useful for preparing voice samples for TTS engines.

## Usage

```bash
bash docs/convert-to-standard-wav.sh input_audio.wav
```

Output: `standard_<input_audio.wav>`

## Requirements

- `ffmpeg` installed on the system

## Notes

This script is provided as-is. It is not integrated into the main application
and must be run manually.
