"""Run this experiment independently from any working directory."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchlib.versions import version_main

if __name__ == "__main__":
    version_main(Path(__file__).resolve().parent)
