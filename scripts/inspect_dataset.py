import argparse
import json
from pathlib import Path

from multimodal_perception.dataset import audit_dataset, open_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--output", default="reports/dataset_audit.json")
    args = parser.parse_args()
    result = audit_dataset(open_dataset(args.root))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))
