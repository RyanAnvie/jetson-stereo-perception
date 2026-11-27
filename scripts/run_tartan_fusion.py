import argparse

from multimodal_perception.thermal_pipeline import run_fusion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TartanRGBT RGB detection + estimated depth + thermal association")
    parser.add_argument("--config", default="configs/tartan_fusion.yaml")
    parser.add_argument("--frames", type=int, default=0, help="0 = all valid groups")
    parser.add_argument("--backend", choices=["onnxruntime", "tensorrt"])
    parser.add_argument("--engine")
    parser.add_argument("--video", help="Intermediate .avi; encode with scripts/encode_video.py")
    parser.add_argument("--paced", action="store_true", help="Follow source timestamps; discard groups more than 100 ms late")
    parser.add_argument("--report", default="reports/jetson/tartan/fp32_fused.json")
    args = parser.parse_args()
    run_fusion(args.config, args.frames, args.video, args.paced, args.report, args.backend, args.engine)
