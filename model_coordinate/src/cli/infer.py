import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.infer import classify_coordinate, parse_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer friend/foe label for target coordinate")
    parser.add_argument("--model", required=True, help="Path to model .joblib")
    parser.add_argument("--friends", required=True, help="JSON array or path to .json")
    parser.add_argument("--foes", required=True, help="JSON array or path to .json")
    parser.add_argument("--target", required=True, help="JSON array or path to .json")

    args = parser.parse_args()
    friend_list, foe_list, target_coord = parse_inputs(args.friends, args.foes, args.target)
    result = classify_coordinate(args.model, friend_list, foe_list, target_coord)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
