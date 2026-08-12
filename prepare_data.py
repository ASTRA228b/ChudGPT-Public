from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm

from chudlm.config import load_yaml
from chudlm.data import clean_text, iter_documents

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>", "<system>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean, split, tokenize, and pack language-model data.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--force-tokenizer", action="store_true")
    return parser.parse_args()


def train_tokenizer(documents: list[str], path: Path, vocab_size: int) -> Tokenizer:
    print(f"Training a byte-level BPE tokenizer with vocabulary size {vocab_size:,}...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(documents, trainer=trainer, length=len(documents))
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path))
    return tokenizer


def encode_split(documents: list[str], tokenizer: Tokenizer, output_path: Path) -> int:
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer is missing the required <eos> token")
    encoded = [tokenizer.encode(document).ids + [eos_id] for document in tqdm(documents, desc=output_path.stem)]
    tokens = np.asarray([token for document in encoded for token in document], dtype=np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens.tofile(output_path)
    return int(tokens.size)


def main() -> None:
    args = parse_args()
    data_config = load_yaml(args.config)
    model_config = load_yaml(args.model_config)
    input_path = Path(data_config["input_path"])
    documents = [
        cleaned
        for raw in iter_documents(input_path, data_config["text_field"], data_config["dialogue_field"])
        if len(cleaned := clean_text(raw)) >= int(data_config["min_characters"])
    ]
    if len(documents) < 2:
        raise ValueError("Need at least two usable documents after cleaning")
    random.Random(int(data_config["seed"])).shuffle(documents)
    validation_count = max(1, round(len(documents) * float(data_config["validation_fraction"])))
    validation_documents, training_documents = documents[:validation_count], documents[validation_count:]

    tokenizer_path = Path(data_config["tokenizer_path"])
    tokenizer = (
        train_tokenizer(training_documents, tokenizer_path, int(model_config["vocab_size"]))
        if args.force_tokenizer or not tokenizer_path.is_file()
        else Tokenizer.from_file(str(tokenizer_path))
    )
    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != int(model_config["vocab_size"]):
        raise ValueError(
            f"Tokenizer has {actual_vocab} tokens but model expects {model_config['vocab_size']}. "
            "Use more data, lower vocab_size, or pass --force-tokenizer."
        )
    output_dir = Path(data_config["output_dir"])
    train_tokens = encode_split(training_documents, tokenizer, output_dir / "train.bin")
    validation_tokens = encode_split(validation_documents, tokenizer, output_dir / "validation.bin")
    metadata = {"train_tokens": train_tokens, "validation_tokens": validation_tokens, "documents": len(documents)}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {train_tokens:,} training and {validation_tokens:,} validation tokens in {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError) as error:
        print(f"Data preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error

