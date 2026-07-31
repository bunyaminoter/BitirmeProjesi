"""
ONNX export script.

Exports a trained model to ONNX format for deployment.

Usage:
    python scripts/export_onnx.py --config configs/experiment/asl_citizen_100.yaml --checkpoint checkpoints/best_model.pt --output model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import load_config
from src.utils.logging import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="exports/model.onnx")
    return parser.parse_args()


def main() -> None:
    """Main ONNX export entry point."""
    args = parse_args()
    config = load_config(args.config)

    logger = setup_logging(level="INFO")
    log = get_logger("export")

    log.info(f"Exporting model: {config.name}")
    log.info(f"Output: {args.output}")

    # TODO: Load model, create sample input, export via ONNXExporter

    log.info("Export pipeline ready. Implement to start.")


if __name__ == "__main__":
    main()
