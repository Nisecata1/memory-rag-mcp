"""Compatibility entry point for the project_v1 worker."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from versions.project_v1.worker import main

if __name__ == "__main__":
    main()
