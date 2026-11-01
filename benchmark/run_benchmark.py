"""Optional dispatcher for independently runnable benchmark versions."""
import sys
import subprocess
sys.dont_write_bytecode = True
from benchlib.runner import (
    PROJECT_SOURCE_FILES, ROOT, apply_run_config, execute, fingerprint,
    hash_file, main, source_snapshot, validate_prediction,
)

if __name__ == "__main__":
    main()
