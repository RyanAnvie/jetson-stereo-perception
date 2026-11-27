import argparse

from multimodal_perception.kitti_pipeline import run_kitti


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + calibrated KITTI stereo, with independent offline reference checks")
    parser.add_argument("--config", default="configs/kitti.yaml")
    parser.add_argument("--mode", choices=["benchmark", "evaluate"], default="benchmark")
    parser.add_argument("--report", default="reports/kitti_benchmark.json")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--rate-hz", type=float)
    parser.add_argument("--drop-late", action="store_true")
    parser.add_argument("--video")
    parser.add_argument("--backend", choices=["onnxruntime", "tensorrt"])
    parser.add_argument("--engine", help="Target-built TensorRT engine with matching .json build provenance")
    args = parser.parse_args()
    run_kitti(args.config, args.report, args.mode, args.paced, args.rate_hz, args.drop_late, args.video, args.frames,
              args.backend, args.engine)
