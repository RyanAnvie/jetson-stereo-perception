import cv2
import numpy as np


BACKGROUND = (22, 20, 17)
FOREGROUND = (238, 235, 224)
MUTED = (171, 165, 151)
ACCENT = (161, 220, 61)


def text(image, value, x, y, size=0.5, color=FOREGROUND, thickness=1):
    cv2.putText(image, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def fit_image(canvas, image, x, y, width, height):
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
    left = x + (width-resized.shape[1])//2
    top = y + (height-resized.shape[0])//2
    canvas[top:top+resized.shape[0], left:left+resized.shape[1]] = resized


def depth_colors(depth, max_m):
    valid = np.isfinite(depth) & (depth > 0)
    display = np.zeros(depth.shape, dtype=np.uint8)
    display[valid] = np.clip(depth[valid] / max_m * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(255-display, cv2.COLORMAP_TURBO)
    color[~valid] = (20, 20, 20)
    return color


def draw_cloud_pixels(view, uv, colors, order, inside):
    """Rasterize the exact five pixels of filled radius-1 LINE_8 circles.

    The latest painter wins, including equal-depth stable-order ties. One
    scatter replaces thousands of Python -> OpenCV calls without decimation.
    """
    ordered = order[inside[order]]
    if not len(ordered):
        return
    offsets = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])
    pixels = (uv[ordered, None, :]+offsets).reshape(-1, 2)
    height, width = view.shape[:2]
    valid = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    keys = pixels[valid, 1]*width+pixels[valid, 0]
    ranks = np.repeat(np.arange(len(ordered), dtype=np.int32), 5)[valid]
    winner = np.full(width*height, -1, dtype=np.int32)
    np.maximum.at(winner, keys, ranks)
    occupied = np.flatnonzero(winner >= 0)
    view.reshape(-1, 3)[occupied] = colors[ordered[winner[occupied]]]


def cloud_view(points, colors, detections, width=600, height=286):
    view = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    # An orthographic oblique display of the RGB optical frame, not world pose.
    scale = 14.0
    origin = np.array([width/2, height-28.0])
    visible = np.isfinite(points).all(axis=1) & (np.abs(points[:, 0]) <= 16)
    visible &= (np.abs(points[:, 1]) <= 10) & (points[:, 2] <= 22) & (points[:, 2] > 0)
    selected = points[visible]
    selected_colors = colors[visible]
    u = origin[0] + scale * (selected[:, 0] + 0.25*selected[:, 2])
    v = origin[1] + scale * (0.65*selected[:, 1] - 0.65*selected[:, 2])
    uv = np.column_stack((u, v)).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    order = np.argsort(-selected[:, 2], kind="stable")
    draw_cloud_pixels(view, uv, selected_colors, order, inside)
    base = tuple(origin.astype(int))
    for endpoint, label, color in [((base[0]+70, base[1]), "X right", (120, 150, 250)),
                                    ((base[0], base[1]+23), "Y down", (120, 230, 130)),
                                    ((base[0]+27, base[1]-70), "Z forward", (245, 190, 120))]:
        cv2.arrowedLine(view, base, endpoint, color, 1, cv2.LINE_AA)
        text(view, label, endpoint[0]+4, endpoint[1], 0.35, color)
    text(view, "Oblique view | meters | display crop Z <= 22 m", 10, 18, 0.40, MUTED)
    text(view, f"{len(points):,} valid samples | {len(selected):,} in display crop", 10, 37, 0.40, MUTED)
    for detection in detections[:3]:
        point = detection["position"]["point_m"]
        if point is not None:
            x, y, z = point
            text(view, f"{detection['class']}: ({x:.1f}, {y:.1f}, {z:.1f}) m", 10, height-12, 0.4, ACCENT)
            break
    return view


def render_dashboard(sample, detections, points, pixels, metrics, elapsed_s, max_m=30):
    canvas = np.full((800, 1280, 3), BACKGROUND, dtype=np.uint8)
    text(canvas, "RGB + THERMAL + ESTIMATED DEPTH", 24, 35, 0.84, FOREGROUND, 2)
    sequence = sample["sequence"].replace("undistorted_images_all_cameras_", "")
    text(canvas, f"TartanRGBT / {sequence}   |   REAL DATASET REPLAY", 25, 61, 0.46, MUTED)
    text(canvas, f"{elapsed_s:05.1f} s   FRAME {sample['index']:03d}", 986, 34, 0.57, ACCENT, 1)
    text(canvas, "YOLOX Nano / ONNX / CPU / FP32", 976, 59, 0.40, MUTED)
    cv2.line(canvas, (24, 76), (1256, 76), (62, 61, 51), 1)

    rgb = sample["rgb"].copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(n) for n in detection["bbox_xyxy"]]
        cv2.rectangle(rgb, (x1, y1), (x2, y2), ACCENT, 2)
        text(rgb, f"{detection['class']} {detection['score']:.2f}", x1, max(20, y1-8), 0.5, ACCENT, 2)
        point = detection["position"]["point_m"]
        if point:
            text(rgb, f"Z {point[2]:.2f} m (estimated)", x1, min(525, y2+20), 0.45, ACCENT)
    text(canvas, "01  RGB / OBJECT DETECTION", 24, 101, 0.51, ACCENT)
    fit_image(canvas, rgb, 24, 112, 600, 286)
    person_count = sum(d["class_id"] == 0 for d in detections)
    text(canvas, f"Raw predictions: {len(detections)} | person: {person_count} | false positives possible", 24, 420, 0.42, MUTED)

    text(canvas, "02  THERMAL / TIME-MATCHED VIEW", 656, 101, 0.51, ACCENT)
    thermal = cv2.cvtColor(sample["thermal"], cv2.COLOR_GRAY2BGR)
    fit_image(canvas, thermal, 656, 112, 600, 286)
    offset = abs(sample["thermal_timestamp_s"]-sample["rgb_timestamp_s"])*1000
    state = "FFC: EXCLUDED" if sample["ffc"] else "FFC: clear"
    text(canvas, f"640 x 512 | delta {offset:.2f} ms | {state} | intensity, not Celsius", 656, 420, 0.40, MUTED)

    text(canvas, "03  FOUNDATIONSTEREO / ESTIMATED Z", 24, 454, 0.51, ACCENT)
    fit_image(canvas, depth_colors(sample["depth"], max_m), 24, 465, 600, 286)
    valid_depth = np.isfinite(sample["depth"]) & (sample["depth"] > 0)
    text(canvas, f"Display 0-{max_m:g} m | finite positive {valid_depth.mean()*100:.2f}% | not ground truth", 24, 772, 0.41, MUTED)

    text(canvas, "04  RGB COLORED 3D / CAMERA FRAME", 656, 454, 0.51, ACCENT)
    point_colors = sample["rgb"][pixels[:, 1], pixels[:, 0]]
    canvas[465:751, 656:1256] = cloud_view(points, point_colors, detections)
    text(canvas, f"CPU processing {metrics['processing_ms']:.1f} ms | thermal 3D mapping: BLOCKED", 656, 772, 0.41, MUTED)
    if not sample["valid"]:
        cv2.rectangle(canvas, (0, 300), (1280, 370), (25, 35, 90), -1)
        text(canvas, "INVALID THERMAL GROUP - DISPLAY ONLY / EXCLUDED FROM FUSION", 55, 345, 0.75, FOREGROUND, 2)
    return canvas


def title_card(title, lines):
    canvas = np.full((800, 1280, 3), BACKGROUND, dtype=np.uint8)
    cv2.rectangle(canvas, (70, 148), (78, 650), ACCENT, -1)
    text(canvas, "AXG  /  MULTIMODAL PERCEPTION", 110, 180, 0.6, ACCENT)
    text(canvas, title, 110, 250, 1.05, FOREGROUND, 2)
    for i, line in enumerate(lines):
        text(canvas, line, 110, 318+i*48, 0.60, MUTED)
    text(canvas, "Reproducible code + raw timing samples + explicit scientific limits", 110, 724, 0.50, MUTED)
    return canvas
