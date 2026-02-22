#!/usr/bin/env python3
"""
Mistral Vibe — Codex-style TUI for Mistral AI.

Usage:
    python run.py                          # uses MISTRAL_API_KEY env var
    python run.py --model codestral-latest
    python run.py --help

Environment Variables:
    MISTRAL_API_KEY    Your Mistral API key (required)
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mistral-vibe",
        description="Codex-style TUI for Mistral AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models available:
  mistral-large-latest    Most capable (default)
  mistral-medium-latest   Balanced speed/quality
  mistral-small-latest    Fast and lightweight
  codestral-latest        Optimised for code
  open-mistral-7b         Open-source 7B
  open-mixtral-8x7b       Open-source MoE

Environment:
  MISTRAL_API_KEY         Your API key from https://console.mistral.ai

Keyboard shortcuts inside the TUI:
  Enter        Send message
  Ctrl+L       Clear conversation
  Ctrl+N       New session
  Ctrl+R       Cycle models
  Ctrl+H       Toggle help panel
  Ctrl+C       Exit
        """,
    )
    parser.add_argument(
        "--model",
        default="mistral-large-latest",
        metavar="MODEL",
        help="Mistral model to use (default: mistral-large-latest)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="Mistral API key (overrides MISTRAL_API_KEY env var)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="mistral-vibe 0.1.0",
    )

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print(
            "Error: No API key found.\n"
            "Set the MISTRAL_API_KEY environment variable or use --api-key.\n\n"
            "  export MISTRAL_API_KEY='your-key-here'\n"
            "  python run.py\n\n"
            "Get your key at: https://console.mistral.ai",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from mistral_vibe.app import run
    except ImportError as exc:
        print(
            f"Import error: {exc}\n"
            "Install dependencies with:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    run(api_key=api_key, model=args.model)


if __name__ == "__main__":
    main()
