#!/usr/bin/env python
"""Einstiegspunkt der Auswertungs-Toolbox.

    python eval/mlflow_toolbox.py --help

Gleichwertig zu ``python -m eval.toolbox`` aus dem Projektwurzelverzeichnis.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toolbox.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
