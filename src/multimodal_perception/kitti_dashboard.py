import cv2
import numpy as np

from multimodal_perception.dashboard import BACKGROUND, FOREGROUND, MUTED, ACCENT, text, fit_image, depth_colors


def render_kitti(sample, depth, detections, evaluation, timing, elapsed_s, config):
    canvas = np.full((800, 1280, 3), BACKGROUND, dtype=np.uint8)
    text(canvas, "YOLO + STEREO GEOMETRY / 3D TARGET POSITIONS", 22, 35, 0.77, FOREGROUND, 2)
    text(canvas, "KITTI raw / 2011_09_26 drive 0005 | Real public data | CPU / FP32", 22, 62, 0.48, MUTED)
    text(canvas, f"Frame {sample['index']:03d} / {elapsed_s:04.1f}s", 1038, 62, 0.48, ACCENT)
    text(canvas, "Green: numbered YOLO predictions     Orange: reference tracklets     XYZ estimates in the table below", 22, 94, 0.46, ACCENT)
    rgb = sample["left"].copy()
    for label in evaluation["labels"]:
        box = np.rint(label["bbox_xyxy"]).astype(int)
        cv2.rectangle(rgb, tuple(box[:2]), tuple(box[2:]), (30, 160, 240), 1)
    for number, detection in enumerate(detections, 1):
        x1, y1, x2, y2 = np.rint(detection["bbox_xyxy"]).astype(int)
        cv2.rectangle(rgb, (x1, y1), (x2, y2), ACCENT, 2)
        caption = f"#{number} {detection['class']} {detection['score']:.2f}"
        caption_width = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
        caption_x = min(max(0, x1), max(0, rgb.shape[1]-caption_width))
        text(rgb, caption, caption_x, max(18, y1-7), 0.45, ACCENT, 1)
    fit_image(canvas, rgb, 20, 106, 1240, 375)
    text(canvas, f"Processing {timing['processing_ms']:.1f} ms | Stereo {timing['stereo_ms']:.1f} ms | Inference {timing['inference_ms']:.1f} ms", 22, 506, .48, MUTED)
    text(canvas, "STEREO DEPTH / IMAGE-ONLY ESTIMATE", 22, 539, .49, ACCENT)
    fit_image(canvas, depth_colors(depth, config["display_depth_max_m"]), 20, 550, 600, 192)
    text(canvas, f"0-{config['display_depth_max_m']:g} m display | valid {np.isfinite(depth).mean()*100:.1f}% | invalid pixels dark", 22, 764, .40, MUTED)
    text(canvas, "3D SURFACE POSITIONS / CAM2 OPTICAL FRAME", 658, 539, .46, ACCENT)
    view = np.full((192, 600, 3), BACKGROUND, dtype=np.uint8)
    for label, x in [("Top 6 / ID", 0), ("X right [m]", 205), ("Y down [m]", 330), ("Z forward [m]", 455)]:
        text(view, label, x, 17, .39, MUTED)
    order = np.argsort([-d["score"] for d in detections], kind="stable")[:6]
    for row, index in enumerate(order):
        detection = detections[index]
        point = detection["position"]["point_m"]
        y = 44+row*27
        text(view, f"#{index+1} {detection['class']}", 0, y, .46, ACCENT)
        if point is None:
            text(view, "insufficient stereo depth", 190, y, .43, MUTED)
        else:
            for column, value in enumerate(point):
                text(view, f"{value:.2f}", 205+column*125, y, .46, FOREGROUND)
    canvas[550:742, 660:1260] = view
    mae = evaluation["frame"]["depth"]["mae_m"]
    coverage = evaluation["frame"]["stereo_coverage_at_lidar"]
    text(canvas, f"Reference depth MAE {mae:.2f} m | coverage {coverage*100:.1f}% | offline check", 658, 764, .40, MUTED)
    text(canvas, "Data: KITTI (Geiger et al.) / CC BY-NC-SA 3.0 | Surface points are not object centers. No thermal or Jetson claims.", 22, 791, .36, MUTED)
    return canvas
