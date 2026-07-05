"""``python -m honest_research`` entry point — delegates to the CLI.

Keeps a single source of truth for the CLI (``cli.py`` at the repo root) so both
``python cli.py ...`` and ``python -m honest_research ...`` run the exact same code.
Imported lazily inside ``main`` so importing the package never pulls argparse wiring.
"""
from __future__ import annotations

import sys


def main() -> int:
    """Import and run the root CLI, returning its process exit code."""
    from cli import main as cli_main  # cli.py lives at the repo root, on sys.path
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
