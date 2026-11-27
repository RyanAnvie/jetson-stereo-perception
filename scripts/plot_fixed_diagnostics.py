import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot():
    folder = Path("reports/jetson/tartan_fixed")
    clock = json.loads((folder/"clock_diagnosis.json").read_text())
    whole = json.loads((folder/"fixed_sequence_validation.json").read_text())
    depth = json.loads((folder/"depth_diagnosis.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for group, label in [(clock["whole_sequence"], "whole sequence")]+[(b, f"{b['interval_s'][0]}-{b['interval_s'][1]} s") for b in clock["temporal_blocks"]]:
        x = np.array([r["thermal_clock_correction_s"] for r in group["curve"]])*1000
        y = [r["mean_correlation"] for r in group["curve"]]
        axes[0].plot(x, y, label=label, linewidth=2 if label == "whole sequence" else 1)
    axes[0].axvline(70, color="black", linestyle="--", linewidth=.8)
    axes[0].set(xlabel="Effective thermal time correction (ms)", ylabel="Motion correlation", title="Independent within-modality motion")
    axes[0].legend(fontsize=8)
    rows = whole["records"]
    for key, label in [("old_fixed_no_ecc", "old fixed extrinsics, ECC off"),
                       ("refit_fixed_with_time_compensation", "refit fixed rig + clock / motion")]:
        axes[1].plot([r["index"] for r in rows], [r["gradient_magnitude_correlation"][key] for r in rows], label=label, linewidth=.9)
    bad = whole["uncompensated_frames"]
    axes[1].scatter(bad, [-.02]*len(bad), marker="|", color="red", s=45, label="motion gate rejected")
    axes[1].set(xlabel="Frame index (all 296 valid groups)", ylabel="Gradient magnitude correlation", title="Full-frame image proxy; not pixel truth")
    axes[1].legend(fontsize=7, loc="upper left")
    x = [r["independent_stereo_depth_m"] for r in depth["records"]]
    y = [r["published_depth_m"] for r in depth["records"]]
    axes[2].scatter(x, y, s=4, alpha=.25)
    axes[2].plot([min(x), max(x)], [min(x), max(x)], color="black", linewidth=1, label="equal depth")
    axes[2].set(xscale="log", yscale="log", xlabel="Independent stereo estimate (m)", ylabel="Published depth (m)", title="2,845 selected subpixel stereo matches")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.suptitle("Fixed RGB-thermal extrinsics: timing, registration, and depth diagnostics (no ECC)", fontsize=13)
    output = Path("results/jetson/tartan_fixed/diagnostics.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    plot()
