"""Create looping README previews from the reviewed MP4s using system FFmpeg."""
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FILTER = "fps=5,scale=480:300:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render():
    results = []
    originals = json.loads((ROOT/"reports/published/tartan_latency/media_validation.json").read_text())["videos"]
    for name in ["balanced", "responsive"]:
        source = ROOT/f"docs/media/tartan-rgbt-{name}.mp4"
        output = source.with_suffix(".gif")
        assert sha256(source) == originals[name]["sha256"]
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
                        "-i", str(source), "-filter_complex_threads", "1", "-filter_complex", FILTER,
                        "-loop", "0", str(output)], check=True)
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
                           "-show_streams", "-show_format", "-of", "json", str(output)], text=True))
        stream = probe["streams"][0]
        assert stream["codec_name"] == "gif"
        assert int(stream["nb_read_frames"]) == 148
        assert abs(float(probe["format"]["duration"])-29.6) < .01
        assert output.stat().st_size < 10_000_000
        assert b"NETSCAPE2.0\x03\x01\x00\x00\x00" in output.read_bytes()
        results.append({"path": str(output.relative_to(ROOT)), "sha256": sha256(output), "bytes": output.stat().st_size,
                        "source": str(source.relative_to(ROOT)), "source_sha256": sha256(source),
                        "width": stream["width"], "height": stream["height"], "frames": int(stream["nb_read_frames"]),
                        "fps": 5, "duration_s": float(probe["format"]["duration"]), "loop": "infinite"})
    report = {"purpose": "Full-duration animated README previews; spatial and temporal downsampling only, not a processing benchmark",
              "filter": FILTER, "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0], "previews": results}
    (ROOT/"reports/published/readme_previews.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    render()
