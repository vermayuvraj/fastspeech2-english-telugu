"""Telugu entry point for the shared, tested FastSpeech 2 pipeline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fs2.train import main

if __name__ == "__main__":
    main("Telugu")
