import os
import sys

# Ensure this directory (with a space in its name) is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.cli import main  # noqa: E402


if __name__ == "__main__":
    main() 