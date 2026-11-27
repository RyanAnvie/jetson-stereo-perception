"""Fetch only the same-sequence right RGB reference used by depth diagnostics."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


def download(root="data/tartan"):
    root = Path(root)
    manifest = json.loads((root/"sample_manifest.json").read_text())
    revision, sequence = manifest["revision"], manifest["sequence"]
    repository = "theairlabcmu/TartanRGBT"
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repository}/tree/{revision}/{sequence}", timeout=40) as response:
        items = json.load(response)
    item = next(x for x in items if x["path"].endswith("/zed_right_rect.zip"))
    if item["size"] > 500_000_000:
        raise ValueError("Reference archive exceeds this audit's 500 MB bound")
    path = root/item["path"]
    url = f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{item['path']}"
    if not path.is_file():
        partial = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as target:
            while True:
                block = response.read(1024*1024)
                if not block: break
                target.write(block)
        partial.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != item["size"] or digest != item["lfs"]["oid"]:
        raise ValueError("Reference archive does not match the pinned source")
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not (path.parent/member.filename).resolve().is_relative_to(path.parent.resolve()):
                raise ValueError("Unsafe reference ZIP member")
        archive.extractall(path.parent)
    record = {"revision": revision, "url": url, "path": str(path), "bytes": path.stat().st_size,
              "sha256": digest, "reason": "Independent stereo-depth consistency check"}
    output = Path("reports/jetson/tartan_fixed/stereo_reference_download.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    download()
