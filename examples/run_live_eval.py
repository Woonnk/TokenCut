"""Prompt for a model connection locally, then run the paired demo evaluation."""

import argparse
import getpass
import os
import sys
import warnings

from tokencut.cli import main


def run() -> int:
    parser = argparse.ArgumentParser(description="Run TokenCut's live agent comparison")
    parser.add_argument("--model", help="Model name; prompted when omitted")
    parser.add_argument("--output", default="live-evaluation.json")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    model = args.model or os.getenv("OPENAI_MODEL") or input("Model name: ").strip()
    if not model.strip():
        parser.error("A model name is required")
    if not 1 <= args.repeats <= 20:
        parser.error("repeats must be between 1 and 20")
    original_key = os.environ.get("OPENAI_API_KEY")
    try:
        if not original_key:
            if not sys.stdin.isatty():
                parser.error(
                    "Use an interactive terminal for hidden key entry or set OPENAI_API_KEY"
                )
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                try:
                    key = getpass.getpass("OpenAI API key (hidden): ").strip()
                except getpass.GetPassWarning:
                    parser.error(
                        "Hidden key entry is unavailable; configure OPENAI_API_KEY locally"
                    )
            os.environ["OPENAI_API_KEY"] = key
        command = [
            "eval",
            "--model",
            model,
            "--repeats",
            str(args.repeats),
            "--format",
            "json",
            "-o",
            args.output,
        ]
        if args.force:
            command.append("--force")
        result = main(command)
        if result in {0, 1}:
            print(f"Evaluation report: {args.output}")
            print(
                "Target gate passed."
                if result == 0
                else "Target gate did not pass; inspect the report."
            )
        return result
    finally:
        if original_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = original_key


if __name__ == "__main__":
    raise SystemExit(run())
