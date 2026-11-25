import hashlib
import json
from pathlib import Path
import urllib.request


if __name__ == "__main__":
    root = Path("models/weights")
    root.mkdir(parents=True, exist_ok=True)
    source = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
    target = root / "yolox_nano.onnx"
    if not target.exists():
        temporary = target.with_suffix(".partial")
        urllib.request.urlretrieve(source, temporary)
        temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    expected_file = Path("reports/model_provenance.json")
    if expected_file.exists():
        expected = json.loads(expected_file.read_text())["sha256"]
        if digest != expected:
            raise ValueError("Model checksum differs from the validated baseline")
    print(json.dumps({"source": source, "sha256": digest, "bytes": target.stat().st_size}, indent=2))
