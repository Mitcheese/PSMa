"""Minimal smoke test: parse config and build model without running data loaders."""

import os
import argparse

from psma_pretrain.utils.config import cfg_from_yaml_file
from psma_pretrain.tools.builder import model_builder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain_base.yaml")
    args = parser.parse_args()

    cfg = cfg_from_yaml_file(args.config)
    _ = model_builder(cfg.model)
    print("SMOKE_OK")


if __name__ == "__main__":
    os.environ.setdefault("PSMA_DATA_ROOT", "data")
    main()
