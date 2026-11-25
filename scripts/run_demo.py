import argparse
import json

from multimodal_perception.replay import environment, run_replay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay time-matched RGB/thermal/estimated depth")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--frames", type=int, default=0, help="0 = all valid groups")
    parser.add_argument("--video", help="Intermediate .avi, then use scripts/encode_video.py")
    parser.add_argument("--paced", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Report OS/Python without dataset or model")
    parser.add_argument("--report", default="reports/benchmark_pc.json")
    args = parser.parse_args()
    if args.smoke:
        print(json.dumps(environment(), indent=2))
    else:
        run_replay(args.config, args.frames, args.video, args.paced, args.report, args.show)
