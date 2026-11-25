"""Download one KITTI raw sequence from the public official object store."""

import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile


ROOT = Path("data/kitti")
BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
FILES = [
    "raw_data/2011_09_26_calib.zip",
    "raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip",
    "raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_tracklets.zip",
]


def download_kitti():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "sample_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    manifest = {"dataset": "KITTI raw", "sequence": "2011_09_26_drive_0005_sync",
                "date": "2011_09_26", "license": "CC-BY-NC-SA-3.0", "files": [],
                "source": "https://www.cvlibs.net/datasets/kitti/raw_data.php",
                "download_source": BASE + "raw_data_downloader.zip"}
    for name in FILES:
        url = BASE + name
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            entry = {"url": url, "bytes": int(response.headers["Content-Length"]),
                     "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}
        manifest["files"].append(entry)
    total = sum(entry["bytes"] for entry in manifest["files"])
    print(json.dumps(manifest, indent=2), flush=True)
    if total > 5_000_000_000:
        raise ValueError("Selected files exceed the 5 GB download budget")
    Path("reports/kitti_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for entry in manifest["files"]:
        target = ROOT / entry["url"].rsplit("/", 1)[-1]
        partial = target.with_suffix(".partial")
        old = None
        if previous:
            old = next(item for item in previous["files"] if item["url"] == entry["url"])
            if entry["etag"] != old["etag"]:
                raise ValueError("Remote archive changed since the recorded run; inspect before updating")
        if not target.exists():
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-", "If-Match": entry["etag"]} if offset else {}
            request = urllib.request.Request(entry["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("Server did not honor the resume range")
                with partial.open("ab" if offset else "wb") as handle:
                    while True:
                        block = response.read(1024*1024)
                        if not block:
                            break
                        handle.write(block)
            partial.replace(target)
        if target.stat().st_size != entry["bytes"]:
            raise ValueError(f"Size mismatch: {target}")
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            while True:
                block = handle.read(1024*1024)
                if not block:
                    break
                digest.update(block)
        entry["sha256"] = digest.hexdigest()
        if old and entry["sha256"] != old["sha256"]:
            raise ValueError(f"Archive hash changed: {target}")
        print("Extracting", target.name, flush=True)
        with zipfile.ZipFile(target) as archive:
            for member in archive.infolist():
                destination = (ROOT / member.filename).resolve()
                if not destination.is_relative_to(ROOT.resolve()):
                    raise ValueError("Unsafe archive member")
                # Keep the full calibrated measurements; images and intensity stay unchanged.
                archive.extract(member, ROOT)
    manifest["total_download_bytes"] = total
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path("reports/kitti_download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Ready:", ROOT / manifest["date"] / manifest["sequence"], flush=True)


if __name__ == "__main__":
    download_kitti()
