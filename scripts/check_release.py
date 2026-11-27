"""Validate the public working tree, measurements and media without GPU dependencies."""
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {"README.md", "README.zh-CN.md", "LICENSE", "THIRD_PARTY.md", "JETSON_AGX_HANDOFF.md", "pyproject.toml", ".gitignore", ".gitattributes"}
PUBLIC_DIRS = {"src", "scripts", "tests", "configs", "docs", "requirements", "reports"}
MEDIA = {"docs/media/tartan-rgbt-balanced.mp4", "docs/media/tartan-rgbt-responsive.mp4"}
REPORT_FILES = {"reports/README.md", "reports/model_provenance.json",
                "reports/jetson/tartan_fixed/fixed_corrected.json",
                "reports/jetson/tartan_fixed/fixed_corrected.outputs.json"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_files(root=ROOT):
    """Use Git's candidate set when available; also work in a standalone source ZIP."""
    try:
        top = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        top = ""
    if top and Path(top).resolve() == root.resolve():
        raw = subprocess.check_output(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"])
        paths = {Path(name.decode()) for name in raw.split(b"\0") if name}
    else:
        paths = {p.relative_to(root) for p in root.rglob("*") if p.is_file()
                 and (p.relative_to(root).parts[0] in PUBLIC_DIRS or str(p.relative_to(root)) in ROOT_FILES)
                 and not any(part.startswith(".") or part.endswith(".egg-info") or part == "__pycache__" for part in p.relative_to(root).parts[:-1])}
        paths = {p for p in paths if p.parts[0] != "reports" or str(p).startswith("reports/published/") or str(p) in REPORT_FILES}
    return sorted(p for p in paths if (root/p).is_file() and not (root/p).is_symlink())


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered)-1)*q/100
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] + (ordered[upper]-ordered[lower])*(index-lower)


def check_measurements(root):
    folder = root/"reports/published/tartan_latency"
    summary = json.loads((folder/"benchmark_summary.json").read_text())
    total = 0
    for profile, expected in summary["profiles"].items():
        selected = [r for r in summary["runs"] if r["profile"] == profile]
        samples = []
        for run in selected:
            report_path = root/run["report"]
            report = json.loads(report_path.read_text())
            with report_path.with_suffix(".csv").open() as handle:
                rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
            assert len(rows) == run["frames"] == report["timed_frames"] == 296
            assert math.isclose(run["elapsed_wall_s"], report["elapsed_wall_s"], abs_tol=1e-10)
            assert math.isclose(run["fps"], len(rows)/run["elapsed_wall_s"], abs_tol=1e-9)
            for row in rows:
                assert all(math.isfinite(v) for v in row.values())
                assert 0 <= row["input_to_detection_ms"] <= row["input_to_delivery_ms"] <= row["input_to_encoder_ms"] <= row["input_to_output_ms"]
            samples.extend(rows)
        assert len(samples) == expected["frames"]
        fps = sum(r["frames"] for r in selected)/sum(r["elapsed_wall_s"] for r in selected)
        assert math.isclose(fps, expected["fps"], abs_tol=1e-9)
        for key, quantiles in expected["latency"].items():
            for label, value in quantiles.items():
                actual = percentile([s[key] for s in samples], int(label[1:]))
                assert math.isclose(actual, value, abs_tol=1e-8), (profile, key, label)
        total += len(samples)
    assert total == summary["total_valid_frames"] == 2664
    proof = json.loads((folder/"output_equivalence.json").read_text())
    assert digest(root/proof["reference"]) == proof["reference_sha256"]
    assert len(proof["runs"]) == 9 and all(r["semantic_equal_to_reference"] for r in proof["runs"])
    provenance = json.loads((root/"reports/published/provenance.json").read_text())
    for row in provenance["files"]:
        assert digest(root/row["path"]) == row["sha256"], row["path"]
        if row["path"].endswith(".csv"):
            assert row["source_sha256"] == row["sha256"]
    videos = json.loads((folder/"deployment_validation.json").read_text())["videos"]
    for name, record in videos.items():
        video = root/f"docs/media/tartan-rgbt-{name}.mp4"
        assert video.stat().st_size == record["bytes"] < 10_000_000
        assert digest(video) == record["sha256"]
        stream = next(s for s in json.loads((folder/f"{name}_video.json").read_text())["streams"] if s["codec_type"] == "video")
        assert stream["codec_name"] == "h264" and int(stream["nb_frames"]) == 296
    return total


def check(root=ROOT):
    paths = public_files(root)
    candidates = {str(p) for p in paths}
    assert MEDIA <= candidates, "Published MP4 files must not be ignored"
    patterns = [
        re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
        re.compile(r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"),
        re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----"),
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})\b"),
    ]
    errors = []
    for relative in paths:
        name, path = str(relative), root/relative
        if relative.parts[0] not in PUBLIC_DIRS and name not in ROOT_FILES:
            errors.append(f"Unexpected public file: {name}")
        if path.suffix in {".pt", ".pth", ".onnx", ".engine", ".npy", ".npz", ".zip"} or (path.suffix == ".mp4" and name not in MEDIA):
            errors.append(f"Local artifact in public files: {name}")
        if path.stat().st_size >= 25_000_000:
            errors.append(f"Oversized release file: {name}")
        if path.suffix in {".mp4", ".jpg", ".png"}:
            continue
        content = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern.search(content):
                errors.append(f"Private host/path or credential pattern in {name}")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", content):
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                target = unquote(target.split("#")[0]).strip("<>")
                resolved = (path.parent/target).resolve()
                try:
                    linked = str(resolved.relative_to(root.resolve()))
                except ValueError:
                    linked = ""
                if linked not in candidates:
                    errors.append(f"Missing public link in {name}: {target}")
    if errors:
        raise RuntimeError("\n".join(errors))
    frames = check_measurements(root)
    result = {"public_files": len(paths), "bytes": sum((root/p).stat().st_size for p in paths),
              "timing_samples_verified": frames, "videos_verified": len(MEDIA),
              "checks": "candidate files, host/credential patterns, relative Markdown links, evidence hashes, pooled percentiles, video hashes"}
    print(json.dumps(result, indent=2))
    return paths


if __name__ == "__main__":
    check()
