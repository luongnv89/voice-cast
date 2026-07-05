import argparse
import logging
import os

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn, TransferSpeedColumn
from rich.table import Table

from models import DownloadProgress, ModelDownloader, ModelRegistry
from models.exceptions import ModelNotInstalledError
from tts_factory import TTSFactory, bootstrap_engines
from voice_cloner import VoiceCloner

# Configure logging with Rich for a better terminal experience
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)])
logger = logging.getLogger("voice_cloner")

console = Console()


def list_models():
    """Display a table of all available models and their status."""
    registry = ModelRegistry()
    models = registry.list_models()

    table = Table(title="Available TTS Models")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Engine", style="magenta")
    table.add_column("Name", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Description")

    for model in models:
        status = "[green]Installed[/green]" if model.is_installed else "[yellow]Not installed[/yellow]"
        size_str = f"{model.size_mb} MB"
        description = model.description if len(model.description) <= 50 else model.description[:47] + "..."
        table.add_row(model.id, model.engine, model.name, size_str, status, description)

    console.print()
    console.print(table)
    console.print()


def download_models(model_ids: list[str], engine: str | None = None):
    """Download specified models with progress display."""
    registry = ModelRegistry()
    downloader = ModelDownloader()

    # Determine which models to download
    if engine:
        engine_models = registry.get_models_for_engine(engine)
        if not engine_models:
            console.print(f"[red]No models found for engine:[/red] {engine}")
            return
        models_to_download = [m.id for m in engine_models]
        console.print(f"[bold]Downloading all models for engine: {engine}[/bold]")
    else:
        models_to_download = model_ids

    if not models_to_download:
        console.print("[yellow]No models specified to download.[/yellow]")
        console.print("Use --list-models to see available models.")
        return

    console.print()

    for model_id in models_to_download:
        try:
            model = registry.get_model(model_id)

            if model.is_installed:
                console.print(f"[green]{model_id}[/green]: Already installed at {model.install_path}")
                continue

            console.print(f"[bold]Downloading {model_id}[/bold] (~{model.size_mb} MB)...")

            # Create progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Downloading {model_id}", total=model.size_mb * 1024 * 1024)

                def update_progress(p: DownloadProgress, task_id=task):
                    progress.update(task_id, completed=p.downloaded_bytes, total=p.total_bytes)

                downloader.download(model_id, progress_callback=update_progress)

            console.print(f"[green]{model_id}[/green]: Downloaded successfully!")

        except Exception as e:
            console.print(f"[red]Error downloading {model_id}:[/red] {e}")

    console.print()


def main():
    bootstrap_engines()

    # Get available engines for help text
    available_engines = TTSFactory.available_engines()
    model_engine_groups = ["chatterbox", "mlx-audio"]
    engine_choices = sorted(set(available_engines + model_engine_groups))
    engines_help = ", ".join(available_engines)

    parser = argparse.ArgumentParser(
        description="Clone a voice and generate speech using multiple TTS engines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Coqui (default)
  python vcloner.py -i voice.wav -t "Hello world" -o output.wav

  # Using Chatterbox Turbo (fast)
  python vcloner.py -i voice.wav -t "Hello world" -o output.wav --engine chatterbox-turbo

  # Using Chatterbox with custom parameters
  python vcloner.py -i voice.wav -t "That's funny [laugh]!" -o output.wav \\
      --engine chatterbox-turbo --cfg-weight 0.3 --exaggeration 0.7

  # List available engines
  python vcloner.py --list-engines

  # List available models and their status
  python vcloner.py --list-models

  # Download a specific model
  python vcloner.py --download-models coqui-xtts-v2

  # Download all models for an engine
  python vcloner.py --download-models --engine chatterbox
        """,
    )

    parser.add_argument(
        "-i", "--input_voice", help="Path to the original voice WAV/MP3 file (REQUIRED for generation)."
    )
    parser.add_argument("-t", "--text", help="Text to be converted to speech (REQUIRED for generation).")
    parser.add_argument("-o", "--output_file", help="Path and name of the output audio file (REQUIRED for generation).")
    parser.add_argument(
        "-e",
        "--engine",
        default=None,
        choices=engine_choices,
        help=(
            f"TTS engine to use for generation. Available generation engines: {engines_help}. "
            "With --download-models, group names like chatterbox are also accepted. Default: coqui"
        ),
    )
    parser.add_argument(
        "-l", "--language", default="en", help="Language code (default: en). For Coqui: en, es, fr, de, etc."
    )

    # Coqui-specific arguments
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Coqui: Sampling temperature (0.1-1.0). Default: 0.7"
    )

    # Chatterbox-specific arguments
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="Chatterbox: CFG weight for text adherence (0.0-1.0). Lower for fast speakers. Default: 0.5",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Chatterbox: Expressiveness level (0.0-1.5). Higher = more dramatic. Default: 0.5",
    )

    # Utility arguments
    parser.add_argument("--list-engines", action="store_true", help="List available TTS engines and exit.")
    parser.add_argument("--no-play", action="store_true", help="Don't play audio after generation.")

    # Model management arguments
    parser.add_argument("--list-models", action="store_true", help="List available TTS models and their status.")
    parser.add_argument(
        "--download-models",
        nargs="*",
        metavar="MODEL_ID",
        help="Download specified model(s). Use with --engine to download all models for an engine.",
    )

    args = parser.parse_args()

    # Handle --list-models
    if args.list_models:
        list_models()
        return

    # Handle --download-models
    if args.download_models is not None:
        # If --download-models is used without model IDs, require an explicit --engine.
        if not args.download_models and not args.engine:
            console.print("[red]Error:[/red] Specify model IDs or use --engine to download all models for an engine.")
            console.print("Example: [cyan]python vcloner.py --download-models --engine chatterbox[/cyan]")
            return

        engine_filter = args.engine if args.download_models == [] else None
        download_models(args.download_models, engine=engine_filter)
        return

    # Handle --list-engines
    if args.list_engines:
        console.print("\n[bold]Available TTS Engines:[/bold]\n")
        engine_info = TTSFactory.get_engine_info()
        for engine_id, display_name in engine_info.items():
            available = TTSFactory.is_available(engine_id)
            status = "[green]installed[/green]" if available else "[red]not installed[/red]"
            console.print(f"  {engine_id}: {display_name} ({status})")
        console.print("")
        return

    # Validate required arguments for generation
    if not args.input_voice or not args.text or not args.output_file:
        parser.print_help()
        console.print("\n[red]Error:[/red] -i, -t, and -o are required for audio generation.")
        return

    # Ensure the directory for the output file exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        logger.info("[bold cyan]Initializing VoiceCloner[/bold cyan]")
        selected_engine = args.engine or "coqui"
        if selected_engine not in available_engines:
            console.print(f"[red]Error:[/red] '{selected_engine}' is a model group, not a generation engine.")
            console.print(f"Choose one of: [cyan]{engines_help}[/cyan]")
            return

        logger.info(f"  Engine: {selected_engine}")
        logger.info(f"  Reference voice: {args.input_voice}")

        # Build engine-specific kwargs
        engine_kwargs = {"language": args.language}

        if selected_engine == "coqui":
            engine_kwargs["temperature"] = args.temperature
        elif "chatterbox" in selected_engine:
            engine_kwargs["cfg_weight"] = args.cfg_weight
            engine_kwargs["exaggeration"] = args.exaggeration

        # Create cloner. Model downloads are explicit; generation fails fast if missing.
        cloner = VoiceCloner(speaker_wav=args.input_voice, engine=selected_engine, auto_download=False)

        logger.info("[bold green]Generating speech...[/bold green]")
        cloner.say(
            args.text, play_audio=not args.no_play, save_audio=True, output_file=args.output_file, **engine_kwargs
        )

        logger.info(f"[bold green]Speech saved to:[/bold green] {args.output_file}")

    except ModelNotInstalledError as e:
        console.print(f"\n[red]Model not installed:[/red] {e.model_id}")
        console.print(f"Download it with: [cyan]{e.install_command}[/cyan]")
        console.print("Or use [cyan]--list-models[/cyan] to see all available models.")
    except FileNotFoundError:
        logger.error(f"[bold red]Error:[/bold red] Input voice file not found: {args.input_voice}")
    except ImportError as e:
        logger.error(f"[bold red]Missing dependency:[/bold red] {e}")
        logger.info("Install required package with: pip install <package-name>")
    except Exception as e:
        logger.error(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    main()
