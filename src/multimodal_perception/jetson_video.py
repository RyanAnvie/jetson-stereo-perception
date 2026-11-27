"""Bounded pipe to Jetson's VIC / NVENC H.264 encoder, no software fallback."""
from pathlib import Path
import shutil
import subprocess

import numpy as np


class JetsonH264Writer:
    def __init__(self, path, fps=10, size=(1280, 800), bitrate=1800000):
        path = Path(path)
        if path.suffix.lower() != ".mp4":
            raise ValueError("Jetson H.264 output must have an .mp4 suffix")
        binary = shutil.which("gst-launch-1.0")
        if binary is None:
            raise RuntimeError("Jetson GStreamer is required for hardware H.264 output")
        self.size, self.closed = size, False
        self.log_path = path.with_suffix(".encoder.log")
        self.log = self.log_path.open("w")
        self.command = [binary, "-q", "fdsrc", "fd=0", "!", "rawvideoparse", "format=bgr",
                        f"width={size[0]}", f"height={size[1]}", f"framerate={fps}/1", "!",
                        "videoconvert", "!", "video/x-raw,format=BGRx", "!", "nvvidconv", "!",
                        "video/x-raw(memory:NVMM),format=NV12", "!", "nvv4l2h264enc",
                        "maxperf-enable=true", f"bitrate={bitrate}", f"iframeinterval={fps}", "num-B-Frames=0",
                        "!", "h264parse", "!", "qtmux", "faststart=true", "!", "filesink", "sync=false",
                        f"location={path}"]
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            raise

    def isOpened(self):
        return not self.closed and self.process.poll() is None

    def write(self, frame):
        if not self.isOpened():
            raise RuntimeError(f"Jetson encoder stopped; see {self.log_path}")
        if frame.shape != (self.size[1], self.size[0], 3) or frame.dtype != np.uint8 or not frame.flags.c_contiguous:
            raise ValueError("Hardware encoder requires contiguous BGR uint8 frames")
        try:
            self.process.stdin.write(memoryview(frame))
        except BrokenPipeError as exc:
            raise RuntimeError(f"Jetson encoder pipe failed; see {self.log_path}") from exc

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError(f"Jetson encoder did not drain; see {self.log_path}")
            if code:
                raise RuntimeError(f"Jetson encoder exited with {code}; see {self.log_path}")
        finally:
            self.log.close()
