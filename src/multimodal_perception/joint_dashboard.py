import cv2
import numpy as np
from multimodal_perception.dashboard import BACKGROUND, FOREGROUND, MUTED, ACCENT, cloud_view, fit_image, text


def render_joint_dashboard(sample, detections, fusion, timing, offset):
    canvas = np.full((800, 1280, 3), BACKGROUND, dtype=np.uint8)
    text(canvas, "RGB + THERMAL / JOINT RECOGNITION", 24, 35, .84, FOREGROUND, 2)
    text(canvas, "Jetson AGX Orin | LCAFNet M3FD | both modalities enter the network", 24, 61, .45, MUTED)
    text(canvas, f"SOURCE {offset:05.1f}s / {sample['index']:03d}", 958, 34, .52, ACCENT)
    text(canvas, "VALID SAMPLES / NO HELD GAP FRAMES", 958, 61, .37, MUTED)
    cv2.line(canvas, (24, 76), (1256, 76), (62, 61, 51), 1)
    rgb = sample["rgb"].copy()
    thermal = cv2.applyColorMap(sample["thermal"], cv2.COLORMAP_INFERNO)
    for detection in detections:
        label = f"#{detection['id']} {detection['class']} {detection['score']:.2f}"
        box = detection["rgb_bbox_xyxy"]
        if box is not None:
            a, b, c, d = [int(v) for v in box]
            cv2.rectangle(rgb, (a, b), (c, d), ACCENT, 2)
            text(rgb, label, max(0, a), max(20, b-8), .5, ACCENT, 2)
        a, b, c, d = [int(v) for v in detection["bbox_xyxy"]]
        cv2.rectangle(thermal, (a, b), (c, d), ACCENT, 2)
        text(thermal, label, min(a, 380), max(20, b-6), .45, ACCENT, 1)
    text(canvas, "01  RGB / JOINT DETECTIONS PROJECTED BACK", 24, 101, .49, ACCENT)
    fit_image(canvas, rgb, 24, 112, 600, 286)
    text(canvas, f"{len(detections)} joint predictions | no TartanRGBT fine-tuning | verify false positives", 24, 420, .40, MUTED)
    text(canvas, "02  THERMAL / JOINT NETWORK OUTPUT", 656, 101, .51, ACCENT)
    fit_image(canvas, thermal, 656, 112, 600, 286)
    skew = abs(sample["rgb_timestamp_s"]-sample["thermal_timestamp_s"])*1000
    text(canvas, f"stamp delta {skew:.2f} ms | FFC clear | intensity, NOT temperature", 656, 420, .41, MUTED)
    text(canvas, "03  REGISTERED RGB + THERMAL OVERLAY", 24, 454, .51, ACCENT)
    overlay = cv2.addWeighted(fusion["registered_rgb"], .60, cv2.applyColorMap(sample["thermal"], cv2.COLORMAP_INFERNO), .40, 0)
    fit_image(canvas, overlay, 24, 465, 600, 286)
    coverage = np.mean(fusion["rgb_observed_mask"] > 0)*100
    alignment = fusion["residual_alignment"]
    shift = alignment["delta_uv"]
    temporal = fusion.get("time_alignment", {})
    if temporal.get("enabled"):
        state = "RGB motion applied" if temporal["compensated"] else "MOTION UNRELIABLE / no compensation"
        label = f"Fixed rig | clock +{temporal['thermal_clock_correction_s']*1000:.0f} ms | {state}"
    else:
        state = f"image shift ({shift[0]:+.1f},{shift[1]:+.1f})px" if alignment["accepted"] else "ECC OFF"
        label = f"Fixed rig | {state} | RGB coverage {coverage:.1f}%"
    text(canvas, label, 24, 772, .38, MUTED)
    text(canvas, "04  THERMAL 3D / ESTIMATED SURFACE POSITION", 656, 454, .48, ACCENT)
    step = max(1, len(fusion["points"])//9000)
    intensity = fusion["intensity_u8"][::step]
    colors = (cv2.applyColorMap(intensity.reshape(-1, 1), cv2.COLORMAP_INFERNO).reshape(-1, 3)
              if len(intensity) else np.empty((0, 3), np.uint8))
    canvas[465:751, 656:1256] = cloud_view(fusion["points"][::step], colors, detections)
    text(canvas, f"Processing {timing['processing_ms']:.1f} ms | estimated depth | not object-center truth", 656, 772, .40, MUTED)
    return canvas
