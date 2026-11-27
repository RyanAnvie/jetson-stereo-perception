import argparse
from multimodal_perception.joint_pipeline import run_joint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGB + thermal joint recognition on the registered TartanRGBT sequence")
    parser.add_argument("--config", default="configs/tartan_joint.yaml")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--video")
    parser.add_argument("--backend", choices=["tensorrt", "onnxruntime"])
    parser.add_argument("--report", default="reports/jetson/tartan_latency/joint_fp32.json")
    args = parser.parse_args()
    run_joint(args.config, args.frames, args.video, args.report, args.backend)
