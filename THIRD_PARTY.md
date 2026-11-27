# Third-party data, models and media

Original project code is licensed under [MIT](LICENSE). The MIT grant does not cover third-party datasets, model weights, exported models, engines or the derived dataset media below. Dependencies retain their own licenses.

## KITTI raw

Source: [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/), sequence `2011_09_26_drive_0005_sync`. The official [raw-data downloader](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data_downloader.zip) lists the public object-store route used here. The downloader records URLs, sizes, ETags and SHA-256 hashes in the locally generated `reports/kitti_download_manifest.json`.

KITTI's authors publish the data under [Creative Commons Attribution-NonCommercial-ShareAlike 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/), as stated on the official dataset homepage. KITTI previews/videos are not included in this public release. If you generate and distribute them, retain that license and attribution: credit Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun / KITTI; changes consist of detection overlays, derived stereo depth, position tables and explanatory titles. Keep this attribution and license with the video. This data/media license is separate from the MIT license for original project code and from YOLOX's license.

```bibtex
@article{Geiger2013IJRR,
  author = {Andreas Geiger and Philip Lenz and Christoph Stiller and Raquel Urtasun},
  title = {Vision meets Robotics: The KITTI Dataset},
  journal = {International Journal of Robotics Research (IJRR)},
  year = {2013}
}
```

The official raw development kit was consulted for calibration, scan timing and tracklet definitions. It is stored locally under ignored `data/kitti/reference/`; its source is not vendored into the project.

## TartanRGBT / AnyThermal

Thermal projection reconstruction additionally references the official [TartanRGBT workspace](https://github.com/AnyThermal/tartan_rgbt_ws/tree/b2bf894d4bdb1da17fb25292a94704da812154e7) and its pinned [FLIR thermal driver](https://github.com/AnyThermal/flir_thermal_driver/tree/3e1b1b89586c61c4a50e5e0c8c52c0d762b68f86). File hashes and provenance are in `configs/tartan_thermal_profile.json`; upstream checkouts are local research artifacts, not vendored application code.

Source: [CMU AirLab dataset](https://huggingface.co/datasets/theairlabcmu/TartanRGBT), revision `23475c0fa9190154e6f1fd78f6b82b29f9d508a0`.

The dataset card declares **BSD-3-Clause-Clear**. The official [AnyThermal repository license](https://github.com/castacks/AnyThermal/blob/main/LICENSE), copyright (c) 2026 Parv Maheshwari, is retained verbatim in [docs/licenses/AnyThermal-LICENSE.txt](docs/licenses/AnyThermal-LICENSE.txt). The dataset repository itself has no separate LICENSE file in the inspected manifest. Retain this notice and the source attribution alongside redistributed screenshots/video; do not imply endorsement. The files `docs/media/tartan-rgbt-balanced.mp4`, `docs/media/tartan-rgbt-responsive.mp4` and their JPG previews are derived from this dataset. Changes include geometric registration, detection/position overlays, estimated-depth visualization, layout and annotations. They retain the dataset attribution and BSD-3-Clause-Clear notice; they are not relicensed as MIT.

```bibtex
@misc{maheshwari2026anythermallearninguniversalrepresentations,
  title={AnyThermal: Towards Learning Universal Representations for Thermal Perception},
  author={Parv Maheshwari and Jay Karhade and Yogesh Chawla and Isaiah Adu and Florian Heisen and Andrew Porco and Andrew Jong and Yifei Liu and Santosh Pitla and Sebastian Scherer and Wenshan Wang},
  year={2026},
  eprint={2602.06203},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2602.06203}
}
```

The dataset's `stereo_depth` was computed by [FoundationStereo](https://github.com/NVlabs/FoundationStereo); this project consumes the published estimates and does not train or run that model. It does not claim independent metric depth accuracy.

## YOLOX

Source: [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [release 0.1.1rc0](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0), `yolox_nano.onnx`. Upstream project license: Apache-2.0, retained in [docs/licenses/YOLOX-LICENSE.txt](docs/licenses/YOLOX-LICENSE.txt). Weight checksum, preprocessing, runtime and class mapping are recorded in [reports/model_provenance.json](reports/model_provenance.json).

The implementation follows the upstream [ONNX inference example](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py), including top-left padded BGR input and output grid decoding. The model was trained on COCO; this project does not redistribute COCO training images. The official `assets/dog.jpg` was used locally for a preprocessing sanity check and is not included in Git.

## Runtime and video

Joint recognition uses the official [WenCongWu/LCAFNet](https://github.com/WenCongWu/LCAFNet/tree/491f21019f15abb5b94c3dc832307f567747a461) M3FD checkpoint, source commit `491f21019f15abb5b94c3dc832307f567747a461`. The upstream download URL, checkpoint/ONNX/engine hashes, input contract and checkpoint class order are retained in the joint model and deployment reports. The inspected upstream checkout has no top-level LICENSE file; inherited YOLOv5 source files carry their own notices. This project does not assign a new license to that code or checkpoint. The upstream checkout, original checkpoint, exported ONNX and target engine remain local ignored artifacts, not vendored application source.

NumPy, OpenCV, PyYAML, Hugging Face Hub, ONNX Runtime and pytest are installed dependencies, not vendored source. The Jetson video path invokes system GStreamer and NVIDIA H.264 plugins; FFmpeg/ffprobe are external encoding/inspection tools for other paths. CUDA, TensorRT, GStreamer, FFmpeg and these Python packages are not redistributed. See their distributions for applicable licenses.

The inspected LCAFNet revision does not provide a top-level project LICENSE. This repository supplies a reproducible download/export workflow, not a grant of rights to redistribute or commercially use its source or weights. Consult the upstream authors for the permissions required for your use. No LCAFNet or inherited YOLOv5 source is bundled here.

Model identity: checkpoint SHA-256 `40c041aa51d974f7c7b7f398b45060d420785312e39c33879fd00f1fba49469b`; source commit `491f21019f15abb5b94c3dc832307f567747a461`. The original benchmark model/engine hashes are retained in the [published run](reports/published/tartan_latency/balanced_1.json); newly exported/built artifacts record their own hashes.
