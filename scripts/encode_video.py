import argparse
import json
from pathlib import Path
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode a GitHub-compatible H.264 MP4")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--metadata", default="reports/video_metadata.json")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", args.input, "-an", "-c:v", "libx264", "-preset", "slow",
                    "-vf", "scale=in_range=full:out_range=tv", "-color_range", "tv",
                    "-crf", "25", "-maxrate", "1600k", "-bufsize", "3200k", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(output)], check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(output)], check=True, capture_output=True, text=True)
    metadata = json.loads(probe.stdout)
    metadata["bytes"] = output.stat().st_size
    metadata["under_10MB"] = output.stat().st_size < 10_000_000
    Path(args.metadata).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not metadata["under_10MB"]:
        raise ValueError("Video exceeds 10 MB; reduce bitrate or duration before GitHub upload")
    print(f"Ready: {output} ({output.stat().st_size/1e6:.2f} MB)")
