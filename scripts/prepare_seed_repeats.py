"""Create paired AUTSL 10-class screening configs with distinct output names.

Only the experiment name, random seed and output paths differ from the two
already reviewed seed-42 configs. Existing differing files are never replaced.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.config import load_config


CONFIG_DIR = ROOT / "configs" / "experiment"
SOURCES = {
    "base16": CONFIG_DIR / "autsl_10_base16_screen.yaml",
    "centerrepair": CONFIG_DIR / "autsl_10_centerrepair_screen.yaml",
}


def create_configs(seeds: list[int]) -> list[Path]:
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")

    originals = {role: load_config(path) for role, path in SOURCES.items()}
    for field in ("model", "augmentation", "training"):
        left = asdict(getattr(originals["base16"], field))
        right = asdict(getattr(originals["centerrepair"], field))
        if field == "training":
            left.pop("checkpoint_dir")
            right.pop("checkpoint_dir")
        if left != right:
            raise ValueError(f"Source configs differ in {field}")

    created = []
    for seed in seeds:
        for role, source in SOURCES.items():
            config = yaml.safe_load(source.read_text(encoding="utf-8"))
            name = f"autsl_10_{role}_seed{seed}"
            config["name"] = name
            config["seed"] = seed
            config["output_dir"] = f"outputs/{name}/"
            config["training"]["checkpoint_dir"] = f"outputs/{name}/checkpoints/"
            target = CONFIG_DIR / f"{name}.yaml"
            content = (
                "# Paired repeat of the 20-epoch seed-42 preprocessing screen.\n"
                f"# Source: {source.name}; no checkpoint resume.\n"
                + yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
            )
            if target.exists() and target.read_text(encoding="utf-8") != content:
                raise FileExistsError(f"Existing config differs: {target}")
            if not target.exists():
                target.write_text(content, encoding="utf-8")
            created.append(target)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 2026])
    args = parser.parse_args()
    for path in create_configs(args.seeds):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
