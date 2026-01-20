import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train coordinate classifier")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--model", required=True, help="Path to output model .joblib")
    parser.add_argument("--config", default=None, help="Path to train config YAML")

    args = parser.parse_args()
    outputs = train_model(args.data, args.model, args.config)
    print(f"Model saved: {outputs.model_path}")
    print(f"Metrics saved: {outputs.metrics_path}")
    print(f"Metadata saved: {outputs.meta_path}")
    print(f"Log saved: {outputs.log_path}")


if __name__ == "__main__":
    main()
