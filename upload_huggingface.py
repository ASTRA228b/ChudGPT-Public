"""Create and upload the exported ChudGPT-Public folder to Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", help="Hugging Face repo, e.g. ASTRA228b/ChudGPT-Public")
    parser.add_argument("--folder", default="release")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    folder = Path(args.folder)
    required = {"model.safetensors", "config.json", "tokenizer.json", "model.py", "inference.py", "README.md"}
    missing = sorted(name for name in required if not (folder / name).is_file())
    if missing:
        raise FileNotFoundError(f"Release is incomplete; missing: {', '.join(missing)}")
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=args.repo_id, repo_type="model")
    print(f"Uploaded: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
