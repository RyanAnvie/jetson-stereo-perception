# 测试视频

两段最终视频直接保存在仓库，均为 TartanRGBT 的 RGB + thermal 联合检测结果。

| 文件 | 大小（字节） | 内容 |
| --- | ---: | --- |
| [balanced](media/tartan-rgbt-balanced.mp4) | 6,950,637 | 默认双上下文、独立调度、FP32 |
| [responsive](media/tartan-rgbt-responsive.mp4) | 6,969,239 | 单上下文、最小预取、FP32 |

两者都是 H.264、1280×800、无音频、296 帧、10 FPS、29.6 秒。预览图来自第 148 帧。视频播放时间经过压缩有效组的时间轴处理，与原始序列采集时长和处理 FPS 不同。不存在保持上一帧填充源空隙的画面；全部有效帧保留，6 组 FFC 无效输入单独计数。画面中的单帧计算计时不能直接换算成并行完整流程 FPS。

| 文件 | SHA-256 |
| --- | --- |
| balanced | `d04d6930a7f2802201e26bd5ff8c98479d72d43adc69b301b17bab01eb43cc8d` |
| responsive | `91f661a3611fb8e50b897efac69fb83eaf0714cbc20b083c8f34ced4ca1ef472` |

逐帧解码验证、无重复 RGB 面板检查和原文件哈希记录在 [deployment_validation.json](../reports/published/tartan_latency/deployment_validation.json)。编码参数见 [balanced](../reports/published/tartan_latency/balanced_video.json) / [responsive](../reports/published/tartan_latency/responsive_video.json)。这些历史记录里的 `results/` 路径为当时生成位置，发布时只复制文件，没有重新编码。

来源：CMU AirLab [TartanRGBT 固定版本](https://huggingface.co/datasets/theairlabcmu/TartanRGBT/tree/23475c0fa9190154e6f1fd78f6b82b29f9d508a0)。作者归属与 BSD-3-Clause-Clear 声明见 [THIRD_PARTY.md](../THIRD_PARTY.md) 及 [完整许可](licenses/AnyThermal-LICENSE.txt)。修改包括几何配准、检测框、估计深度/位置、拼接面板和文字叠加。视频和缩略图不被项目 MIT 许可证重新授权。

README 采用图片链接到 MP4，仓库上传时会一起包含这两个文件，不依赖外部附件。需要 README 内嵌 GitHub 播放器时，可在发布后自行将 MP4 拖入 Issue/PR 评论框并复制生成的附件 URL；当前没有虚构附件地址。GitHub 推荐 H.264，详见 [附件文档](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)。这两个文件也小于 GitHub 普通仓库单文件限制，无需 Git LFS，见 [大文件文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

重新生成请按 [Jetson 指南](JETSON.md) 运行，输出到 `results/demo/`，不覆盖已经发布且有固定哈希的演示。
