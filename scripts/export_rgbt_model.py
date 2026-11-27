"""Export the pinned upstream LCAFNet checkpoint in an isolated CPU environment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


SOURCE_COMMIT = "491f21019f15abb5b94c3dc832307f567747a461"


def validate_repository(repository):
    """Bind exported provenance to the source actually used for unpickling."""
    repository = Path(repository).resolve()
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError("LCAFNet source must be checked out at the pinned commit")
    dirty = subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise ValueError("LCAFNet tracked source has local modifications; use a clean pinned checkout")
    return commit


def export_model(repository, checkpoint, output):
    source_commit = validate_repository(repository)
    sys.path.insert(0, str(Path(repository).resolve()))
    import torch
    import onnx

    torch.set_num_threads(4)
    checkpoint, output = Path(checkpoint), Path(output)
    expected = "40c041aa51d974f7c7b7f398b45060d420785312e39c33879fd00f1fba49469b"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected:
        raise ValueError("Checkpoint checksum differs from the audited upstream download")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = saved["ema"] if saved.get("ema") is not None else saved["model"]
    model = model.float().eval()
    inputs = (torch.zeros(1, 3, 640, 640), torch.zeros(1, 3, 640, 640))
    with torch.no_grad():
        prediction = model(*inputs)[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, inputs, str(output), opset_version=17, input_names=["rgb", "thermal"],
                      output_names=["predictions", "head4", "head8", "head16", "head32"],
                      do_constant_folding=True, dynamo=False)
    graph = onnx.load(str(output))
    del graph.graph.output[1:]
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    metadata = {"name": "LCAFNet M3FD", "repository": "https://github.com/WenCongWu/LCAFNet",
                "source_commit": source_commit,
                "checkpoint_url": "https://drive.google.com/file/d/14HxymFlpwtq4eVr8-gn5QoivRs7AYieJ/view",
                "checkpoint_sha256": expected, "model_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "classes": list(model.names), "architecture": model.yaml, "torch": torch.__version__,
                "onnx": onnx.__version__, "opset": 17,
                "inputs": {"rgb": [1, 3, 640, 640], "thermal": [1, 3, 640, 640]},
                "outputs": {"predictions": list(prediction.shape)},
                "preprocessing": "aligned RGB and 3-channel repeated thermal grayscale; centered letterbox 640; RGB order; float32 0..1",
                "output_format": "decoded center xy, width height, objectness, six class probabilities",
                "training_dataset": "M3FD; no TartanRGBT fine-tuning or accuracy certification"}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ["name", "classes", "inputs", "outputs", "model_sha256"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default="results/thermal_research/LCAFNet")
    parser.add_argument("--checkpoint", default="models/weights/LCAFNet_M3FD.pt")
    parser.add_argument("--output", default="models/weights/lcafnet_m3fd.onnx")
    args = parser.parse_args()
    export_model(args.repository, args.checkpoint, args.output)
