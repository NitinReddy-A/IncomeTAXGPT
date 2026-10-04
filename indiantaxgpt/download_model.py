"""Download a quantized Llama 2 chat model from Hugging Face into ./models.

Usage:
    python -m indiantaxgpt.download_model                       # Q4_K_M, ~4.1 GB, 8 GB RAM
    python -m indiantaxgpt.download_model --file llama-2-7b-chat.Q8_0.gguf   # ~7.2 GB, 16 GB RAM
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

DEFAULT_REPO = "TheBloke/Llama-2-7B-Chat-GGUF"
DEFAULT_FILE = "llama-2-7b-chat.Q4_K_M.gguf"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="indiantaxgpt-download-model",
        description="Download a GGUF model for IndianTaxGPT.",
    )
    parser.add_argument(
        "--repo", default=DEFAULT_REPO, help=f"Hugging Face repo (default: {DEFAULT_REPO})"
    )
    parser.add_argument(
        "--file", default=DEFAULT_FILE, help=f"File in the repo (default: {DEFAULT_FILE})"
    )
    parser.add_argument(
        "--dest", type=Path, default=Path("models"), help="Target folder (default: models)"
    )
    args = parser.parse_args(argv)

    from huggingface_hub import hf_hub_download

    args.dest.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=args.repo, filename=args.file, local_dir=args.dest)
    print(f"Saved to {path}")
    if args.file != DEFAULT_FILE:
        print(f"Set LLM_MODEL_PATH={Path(path).as_posix()} in your .env to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
