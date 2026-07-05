"""Programmatic model management example for VoiceCast.

This example lists available models, downloads one explicitly if needed, and then
creates a VoiceCloner. Normal VoiceCloner generation does not download models
implicitly.
"""

from voice_cloner import VoiceCloner

MODEL_ID = "coqui-xtts-v2"
ENGINE = "coqui"
VOICE_SAMPLE = "./voice-samples/speaker.wav"


def main() -> None:
    print("Available models:")
    for model in VoiceCloner.list_models():
        status = "installed" if model.is_installed else "missing"
        print(f"- {model.id:22} {model.engine:12} {status:10} ~{model.size_mb} MB")

    if not VoiceCloner.is_model_installed(MODEL_ID):
        print(f"\nDownloading {MODEL_ID} explicitly...")
        VoiceCloner.download_model(MODEL_ID)

    cloner = VoiceCloner(speaker_wav=VOICE_SAMPLE, engine=ENGINE)
    cloner.say(
        "Hello from the explicit model management example.",
        save_audio=True,
        output_file="model_management_example.wav",
        play_audio=False,
    )

    cloner.switch_engine("chatterbox-turbo")
    print("Switched to chatterbox-turbo. Download it first before generating with it.")


if __name__ == "__main__":
    main()
