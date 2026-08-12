from __future__ import annotations

import argparse

from chudlm.config import dataclass_from_dict, load_yaml
from chudlm.model import ModelConfig, TransformerLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the model's exact trainable parameter count.")
    parser.add_argument("--config", default="configs/model.yaml")
    args = parser.parse_args()
    model = TransformerLM(dataclass_from_dict(ModelConfig, load_yaml(args.config)))
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")


if __name__ == "__main__":
    main()

