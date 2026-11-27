import cv2
import numpy as np

from multimodal_perception.dashboard import (
    ACCENT, BACKGROUND, FOREGROUND, MUTED, cloud_view, depth_colors, fit_image, text)


def render_fusion_dashboard(sample, detections, fusion, timing, elapsed_s, runtime_label, max_m=30):
    canvas = np.full((800, 1280, 3), BACKGROUND, dtype=np.uint8)
    text(canvas, "JETSON / RGB + THERMAL FUSION", 24, 35, 0.86, FOREGROUND, 2)
    text(canvas, "TartanRGBT real replay | source-derived projection + CAD nominal extrinsics", 24, 61, 0.45, MUTED)
    text(canvas, f"{elapsed_s:05.1f} s   FRAME {sample['index']:03d}", 987, 34, 0.55, ACCENT)
    text(canvas, runtime_label, 961, 61, 0.39, MUTED)
    cv2.line(canvas, (24, 76), (1256, 76), (62, 61, 51), 1)
    rgb = sample["rgb"].copy()
    thermal = cv2.applyColorMap(sample["thermal"], cv2.COLORMAP_INFERNO)
    associated = 0
    for detection in detections:
        identity = detection["id"]
        x1, y1, x2, y2 = [int(v) for v in detection["bbox_xyxy"]]
        cv2.rectangle(rgb, (x1, y1), (x2, y2), ACCENT, 2)
        text(rgb, f"#{identity} {detection['class']} {detection['score']:.2f}", x1, max(20, y1-8), 0.5, ACCENT, 2)
        point = detection["position"]["point_m"]
        if point is not None:
            text(rgb, f"Z {point[2]:.2f} m (estimated)", x1, min(525, y2+20), 0.45, ACCENT)
        association = detection["thermal"]
        if association["status"] == "associated_nominal":
            associated += 1
            a, b, c, d = association["bbox_xyxy"]
            cv2.rectangle(thermal, (a, b), (c-1, d-1), ACCENT, 2)
            text(thermal, f"#{identity} RGB projected | I50 {association['intensity_p10_p50_p90'][1]:.0f}",
                 max(0, min(a, 300)), max(22, b-8), 0.44, ACCENT, 1)
    text(canvas, "01  RGB / YOLOX NANO DETECTIONS", 24, 101, 0.51, ACCENT)
    fit_image(canvas, rgb, 24, 112, 600, 286)
    text(canvas, f"{len(detections)} predictions | {associated} thermal associations | false positives possible", 24, 420, 0.42, MUTED)
    text(canvas, "02  THERMAL / PROJECTED RGB FOOTPRINTS", 656, 101, 0.51, ACCENT)
    fit_image(canvas, thermal, 656, 112, 600, 286)
    skew = abs(sample["rgb_timestamp_s"] - sample["thermal_timestamp_s"]) * 1000
    text(canvas, f"delta {skew:.2f} ms | FFC clear | processed 0-255 intensity, NOT Celsius", 656, 420, 0.41, MUTED)
    text(canvas, "03  FOUNDATIONSTEREO / ESTIMATED Z", 24, 454, 0.51, ACCENT)
    fit_image(canvas, depth_colors(sample["depth"], max_m), 24, 465, 600, 286)
    text(canvas, f"Display 0-{max_m:g} m | supplied depth estimate | no independent depth ground truth", 24, 772, 0.40, MUTED)
    text(canvas, "04  THERMAL COLORED 3D / RGB CAMERA FRAME", 656, 454, 0.49, ACCENT)
    # Rendering subsamples the already fused cloud; numerical association uses all points.
    colors = (cv2.applyColorMap(fusion["intensity_u8"].reshape(-1, 1), cv2.COLORMAP_INFERNO).reshape(-1, 3)
              if len(fusion["intensity_u8"]) else np.empty((0, 3), dtype=np.uint8))
    canvas[465:751, 656:1256] = cloud_view(fusion["points"][::8], colors[::8], detections)
    text(canvas, f"{len(fusion['points']):,} fused samples | {timing['processing_ms']:.1f} ms | approximate registration", 656, 772, 0.40, MUTED)
    return canvas
