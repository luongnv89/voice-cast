import os

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
VoiceCloner = pytest.importorskip("voice_cloner").VoiceCloner


# Fixture for the VoiceCloner instance
@pytest.fixture
def voice_cloner():
    # Use a sample speaker reference file for testing
    speaker_wav = "./original_voice.wav"
    if not os.path.exists(speaker_wav):
        pytest.skip("Sample speaker file not found. Skipping tests.")
    return VoiceCloner(speaker_wav=speaker_wav)


# Test initialization of VoiceCloner
def test_voice_cloner_initialization(voice_cloner):
    assert voice_cloner is not None
    assert voice_cloner.device in ["cuda", "cpu"]
    assert os.path.exists(voice_cloner.speaker_wav)


# Test text-to-speech generation
def test_say_method(voice_cloner):
    text_to_voice = "This is a test."
    voice_cloner.say(text_to_voice, language="en", play_audio=False, save_audio=True, speed=1.0)

    # Check if the audio file was saved
    assert os.path.exists("audio_*.wav")  # Replace with actual filename logic if needed


# Test audio file saving
def test_save_audio(voice_cloner):
    text_to_voice = "This is a test for saving audio."
    save_path = "tests/output_audio.wav"
    voice_cloner.say(text_to_voice, language="en", play_audio=False, save_audio=True, speed=1.0)

    # Verify the saved audio file
    assert os.path.exists(save_path)
    audio_data, samplerate = sf.read(save_path)
    assert isinstance(audio_data, np.ndarray)
    assert samplerate > 0


# Test playback speed adjustment
def test_playback_speed(voice_cloner):
    text_to_voice = "This is a test for playback speed."
    voice_cloner.say(text_to_voice, language="en", play_audio=False, save_audio=True, speed=1.5)

    # Verify the saved audio file
    save_path = "tests/output_audio_fast.wav"
    assert os.path.exists(save_path)
    audio_data, samplerate = sf.read(save_path)
    assert isinstance(audio_data, np.ndarray)
    assert samplerate > 0


# Test invalid speaker file
def test_invalid_speaker_file():
    with pytest.raises(FileNotFoundError):
        VoiceCloner(speaker_wav="nonexistent_file.wav")


# Test unsupported language
def test_unsupported_language(voice_cloner):
    with pytest.raises((ValueError, RuntimeError)):
        voice_cloner.say("This is a test.", language="xx", play_audio=False)
