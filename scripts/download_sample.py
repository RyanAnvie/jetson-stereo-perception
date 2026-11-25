"""Select one trajectory from the official manifest; never download all data."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from huggingface_hub import HfApi, hf_hub_download


def download_sample(root, sequence=None, revision="main", inspect_only=False):
    api = HfApi()
    info = api.dataset_info("theairlabcmu/TartanRGBT", revision=revision, files_metadata=True)
    revision = info.sha
    required = {"zed_left_rect.zip", "thermal_left_rect_8.zip", "stereo_depth.zip", "target_timestamps.txt"}
    sequences = {}
    shared = []
    for item in info.siblings:
        path = item.rfilename
        if path.startswith(("calibration/", "configs/")) or path in ("README.md", "LICENSE", "LICENSE.txt"):
            shared.append(item)
        if path.rsplit("/", 1)[-1] in required:
            sequences.setdefault(path.rsplit("/", 1)[0], []).append(item)
    candidates = []
    for name, files in sequences.items():
        if {f.rfilename.rsplit("/", 1)[-1] for f in files} == required:
            candidates.append((sum(f.size for f in files), name))
    if sequence is None:
        sequence = min(candidates)[1]
    if sequence not in dict((name, size) for size, name in candidates):
        raise ValueError("Sequence does not contain all required modalities")
    selected = sequences[sequence] + shared
    manifest = {"repository": info.id, "revision": revision, "sequence": sequence,
                "license": info.card_data.get("license"), "files": [],
                "total_bytes": sum(f.size for f in selected)}
    for item in selected:
        manifest["files"].append({"path": item.rfilename, "size": item.size})
    print(json.dumps(manifest, indent=2), flush=True)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if inspect_only:
        return
    if manifest["total_bytes"] > 5_000_000_000:
        raise ValueError("Selected download exceeds the 5 GB budget; choose a smaller trajectory")
    root = Path(root)
    for item, entry in zip(selected, manifest["files"]):
        print("Downloading", item.rfilename, flush=True)
        path = Path(hf_hub_download(info.id, item.rfilename, repo_type="dataset", revision=revision, local_dir=root))
        if path.stat().st_size != item.size:
            raise ValueError(f"Size mismatch: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        entry["sha256"] = digest.hexdigest()
        if path.suffix == ".zip":
            print("Extracting", path.name, flush=True)
            with zipfile.ZipFile(path) as archive:
                destination = path.parent.resolve()
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    if not target.is_relative_to(destination):
                        raise ValueError(f"Unsafe ZIP member: {member.filename}")
                archive.extractall(destination)
    (root / "sample_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path("reports/download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Ready:", root / sequence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--sequence")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    download_sample(args.root, args.sequence, args.revision, args.inspect_only)
