# Third-party data and models

## KITTI raw

Source: [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/), sequence `2011_09_26_drive_0005_sync`. The official [raw-data downloader](https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data_downloader.zip) lists the public object-store route used here. Exact URLs, sizes, ETags and SHA-256 hashes are in `reports/kitti_download_manifest.json`.

KITTI's authors publish the data under [Creative Commons Attribution-NonCommercial-ShareAlike 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/), as stated on the official dataset homepage. The KITTI-derived preview and annotated video in this project are distributed under that same license: credit Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun / KITTI; changes consist of detection overlays, derived stereo depth, position tables and explanatory titles. Keep this attribution and license with the video. This data/media license is separate from YOLOX's code license and does not assign a license to this project's original code.

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

Source: [CMU AirLab dataset](https://huggingface.co/datasets/theairlabcmu/TartanRGBT), revision `23475c0fa9190154e6f1fd78f6b82b29f9d508a0`.

The dataset card declares **BSD-3-Clause-Clear**. The official [AnyThermal repository license](https://github.com/castacks/AnyThermal/blob/main/LICENSE), copyright (c) 2026 Parv Maheshwari, is retained verbatim in [docs/licenses/AnyThermal-LICENSE.txt](docs/licenses/AnyThermal-LICENSE.txt). The dataset repository itself has no separate LICENSE file in the inspected manifest. Retain this notice and the source attribution alongside redistributed screenshots/video; do not imply endorsement. The demo consists of rendered, annotated views of this public dataset.

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

NumPy, OpenCV, PyYAML, Hugging Face Hub, ONNX Runtime and pytest are installed dependencies, not vendored source. FFmpeg is invoked externally to encode the generated video and is not redistributed. See package distributions for their licenses.

Original project code has no project-wide public reuse license assigned yet. Third-party rights and notices remain independent of any later project licensing choice.
