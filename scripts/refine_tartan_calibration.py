import argparse
from multimodal_perception.scene_calibration import refine_scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene-constrained RGB-D / thermal line bundle adjustment")
    parser.add_argument("--root", default="data/tartan")
    parser.add_argument("--profile", default="configs/tartan_thermal_profile.json")
    parser.add_argument("--annotations", default="configs/tartan_scene_lines.json")
    parser.add_argument("--heldout", default="configs/tartan_alignment_observations.json")
    parser.add_argument("--output", default="configs/tartan_scene_refinement.json")
    parser.add_argument("--report", default="reports/jetson/tartan_joint/scene_bundle_adjustment.json")
    args = parser.parse_args()
    refine_scene(args.root, args.profile, args.annotations, args.heldout, args.output, args.report)
