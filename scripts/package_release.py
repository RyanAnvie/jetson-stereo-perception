"""Create a verified source archive including the demonstration videos."""
import hashlib
import json
from pathlib import Path
import zipfile

from check_release import ROOT, check, digest


def package():
    paths = check()
    folder = ROOT/"results/releases"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder/"jetson-stereo-perception-source.zip"
    files = [{"path": str(p), "bytes": (ROOT/p).stat().st_size, "sha256": digest(ROOT/p)} for p in paths]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in paths:
            handle.write(ROOT/path, str(Path("jetson-stereo-perception")/path))
    with zipfile.ZipFile(archive) as handle:
        for row in files:
            content = handle.read("jetson-stereo-perception/"+row["path"])
            assert hashlib.sha256(content).hexdigest() == row["sha256"], row["path"]
    result = {"archive": archive.name, "bytes": archive.stat().st_size, "sha256": digest(archive),
              "file_count": len(files), "all_members_verified": True,
              "scope": "Public working-tree files only; no Git history, data, models, environments or local backups", "files": files}
    archive.with_suffix(".manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    archive.with_suffix(".sha256").write_text(result["sha256"]+"  "+archive.name+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    package()
