"""Curate measured evidence from a local report archive without publishing host details."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_report(report):
    report = copy.deepcopy(report)
    report.pop("environment", None)
    model = report.get("model", {})
    report["model"] = {k: model[k] for k in ["name", "repository", "source_commit", "checkpoint_sha256", "model_sha256",
                      "engine_sha256", "classes", "inputs", "outputs", "backend", "runtime", "precision", "cuda_graph"] if k in model}
    report["publication_note"] = "Original measured timings and configuration; host environment dump and upstream model architecture omitted."
    return report


def prepare(source_root):
    source_root = Path(source_root)
    destination = Path("reports/published")
    destination.mkdir(parents=True, exist_ok=True)
    source = source_root/"jetson/tartan_latency"
    target = destination/"tartan_latency"
    target.mkdir(exist_ok=True)
    provenance = []

    def save_json(path, data, original):
        path.write_text(json.dumps(data, indent=2)+"\n")
        provenance.append({"path": str(path), "sha256": digest(path),
                           "source_relative_path": str(original.relative_to(source_root)), "source_sha256": digest(original)})

    baseline_path = source_root/"jetson/tartan_fixed/fixed_corrected.outputs.json"
    baseline = json.loads(baseline_path.read_text())
    output_proof = []
    summary = json.loads((source/"benchmark_summary.json").read_text())
    for run in summary["runs"]:
        name = Path(run["report"]).name
        original = source/name
        save_json(target/name, clean_report(json.loads(original.read_text())), original)
        csv_path = original.with_suffix(".csv")
        shutil.copyfile(csv_path, target/csv_path.name)
        provenance.append({"path": str(target/csv_path.name), "sha256": digest(csv_path),
                           "source_relative_path": str(csv_path.relative_to(source_root)), "source_sha256": digest(csv_path)})
        output_path = original.with_suffix(".outputs.json")
        actual = json.loads(output_path.read_text())
        if actual != baseline:
            raise ValueError(f"Reference output mismatch: {output_path.name}")
        output_proof.append({"run": original.stem, "frames": len(actual), "semantic_equal_to_reference": True,
                             "source_output_sha256": digest(output_path)})
        run["report"] = str(target/name)
    save_json(target/"benchmark_summary.json", summary, source/"benchmark_summary.json")
    (target/"output_equivalence.json").write_text(json.dumps({"reference": "reports/jetson/tartan_fixed/fixed_corrected.outputs.json",
        "reference_sha256": digest(baseline_path), "runs": output_proof,
        "method": "Parsed every original per-frame output and compared with the reference; repeated output files omitted from publication."}, indent=2)+"\n")
    for name in ["tests_trt.log", "tests_cpu.log", "deployment_validation.json"]:
        shutil.copyfile(source/name, target/name)
    for name in ["balanced", "responsive"]:
        save_json(target/f"{name}_video.json", json.loads((source/f"{name}_3_video.json").read_text()), source/f"{name}_3_video.json")
    fixed = Path("reports/jetson/tartan_fixed")
    fixed.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(baseline_path, fixed/baseline_path.name)
    save_json(fixed/"fixed_corrected.json", clean_report(json.loads((baseline_path.parent/"fixed_corrected.json").read_text())), baseline_path.parent/"fixed_corrected.json")
    for folder, names in {
        "tartan_fixed": ["fixed_sequence_validation.json", "fixed_scene_bundle_adjustment.json", "clock_diagnosis.json", "depth_diagnosis.json"],
        "tartan_max": ["registration_equivalence.json", "fp16_prediction_comparison.json"],
    }.items():
        dest = destination/folder
        dest.mkdir(exist_ok=True)
        for name in names:
            original = source_root/"jetson"/folder/name
            save_json(dest/name, json.loads(original.read_text()), original)
    # Keep the short sustained-run history without dangling paths to omitted reports.
    original = source_root/"jetson/tartan_max/sustained_summary.json"
    sustained = json.loads(original.read_text())
    for run in sustained["runs"]:
        run["original_report_name"] = Path(run.pop("report")).name
    save_json(destination/"tartan_max/sustained_summary.json", sustained, original)
    shutil.copyfile(source_root/"model_provenance.json", "reports/model_provenance.json")
    (destination/"provenance.json").write_text(json.dumps({"scope": "Measured values retained; JSON host metadata removed; timing CSVs byte-for-byte unchanged; original file hashes recorded.", "files": provenance}, indent=2)+"\n")
    print(f"Published {len(provenance)} evidence records; verified {sum(r['frames'] for r in output_proof)} output groups.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path, help="Local archive of the original reports directory")
    prepare(parser.parse_args().source_root)
