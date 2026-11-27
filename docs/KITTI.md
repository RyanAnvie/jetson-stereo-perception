# KITTI RGB 双目基线

这是与 TartanRGBT 主线分开的 RGB-only 路径：YOLOX Nano 检测 + OpenCV StereoSGBM 深度 + 可见表面 XYZ，支持 ONNX Runtime 或 TensorRT 检测。它不使用热像，不能引用 README 的双模态 FPS 作为自己的性能。

数据为 KITTI raw `2011_09_26_drive_0005_sync`，154 组同步图像，左右相机 `image_02` / `image_03`。下载器还获取标定和 tracklets，约 646 MB。原始数据与生成媒体遵循 CC BY-NC-SA 3.0，见 [第三方许可](../THIRD_PARTY.md)。公开包不包含 KITTI 视频或原始数据。

## 运行

完成 [运行环境](JETSON.md) 安装，另装 `requirements/test.txt` 中的 ONNX Runtime：

```bash
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1
.venv-jetson-trt/bin/python scripts/download_kitti.py
.venv-jetson-trt/bin/python scripts/download_model.py
.venv-jetson-trt/bin/python scripts/run_kitti.py --frames 5 --report reports/local/kitti_sample.json
.venv-jetson-trt/bin/python scripts/run_kitti.py --report reports/local/kitti_full.json

# 控制到达速率的回放；drop-late 会明确记录跳过的源索引。
.venv-jetson-trt/bin/python scripts/run_kitti.py \
  --paced --rate-hz 10 --drop-late --report reports/local/kitti_paced.json

# 离线参考对照和渲染，不能当作纯推理吞吐。
.venv-jetson-trt/bin/python scripts/run_kitti.py --mode evaluate \
  --video results/kitti/evaluation.avi --report reports/local/kitti_evaluation.json
```

TensorRT 检测需要另行构建 YOLOX engine：

```bash
.venv-jetson-trt/bin/python scripts/build_jetson_engines.py \
  --model models/weights/yolox_nano.onnx --precision all \
  --report-dir reports/local/build_yolox
JETSON_TEST_ENGINE_DIR=models/weights/jetson \
  .venv-jetson-trt/bin/python -m pytest -q tests/test_backends.py
```

运行器的 `--backend tensorrt --engine models/weights/jetson/yolox_nano_fp32.engine` 选择 GPU 检测。StereoSGBM 仍在 CPU 上。输出路径不同的重复运行可用于比较，不覆盖公共 TartanRGBT 参考。

## 几何与评价定义

输出坐标是 `kitti_cam2_rect`：X 向右、Y 向下、Z 向前，单位米。由发布的投影矩阵分解 `P_i = K_i [I | t_i]`，保留三维平移分量，彩色双目水平基线约 0.532712 m。SGBM 的 1/16 像素视差先转换成实际视差，再解投影关系，不复用 TartanRGBT 内参。

XYZ 是检测框内有限、正深度点的表面中位数，会混入背景，不是物体中心或 tracklet 底部中心。无效深度另行记录。LiDAR 和 tracklets 仅用于离线评价，不进入推理路径。

离线诊断把 LiDAR 变换到左彩色相机坐标，投影碰撞时取最近正 Z，分别统计覆盖和共同有效像素的误差。目标关联使用 IoU 0.5 的一对一贪心匹配及显式类别映射。原始 LiDAR 有扫描时间跨度，未补偿扫描运动，不能将相机/雷达差异当作认证的物理定位误差；也没有官方 KITTI mAP 声明。

`--paced` 按源时间或指定速率调度；`--drop-late` 选择已到达的最新组并记录丢弃。无丢帧模式保留全部输入，可能积累延迟。回放延迟、阶段处理时间和含离线评价/渲染的 wall time 有不同边界，必须按报告字段解释。
