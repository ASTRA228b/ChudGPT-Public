"""Create a private Hugging Face Space that exposes ChudGPT-Public by API."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("space_id", help="Space ID, e.g. ASTRA228b/ChudGPT-Public-API")
    parser.add_argument("--model-repo", required=True, help="Uploaded model ID")
    parser.add_argument("--public", action="store_true", help="Make API public instead of token-protected")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent / "space"
    api = HfApi()
    api.create_repo(
        args.space_id, repo_type="space", space_sdk="gradio",
        private=not args.public, exist_ok=True,
    )
    api.add_space_variable(args.space_id, "MODEL_REPO_ID", args.model_repo)
    api.upload_folder(folder_path=folder, repo_id=args.space_id, repo_type="space")
    print(f"Space deployed: https://huggingface.co/spaces/{args.space_id}")
    print("API instructions appear under the Space's 'Use via API' link after it builds.")


if __name__ == "__main__":
    main()
