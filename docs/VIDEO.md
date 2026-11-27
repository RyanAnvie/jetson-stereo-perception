# 测试视频

两段最终视频直接保存在仓库，均为 TartanRGBT 的 RGB + thermal 联合检测结果。

| 文件 | 大小（字节） | 内容 |
| --- | ---: | --- |
| [balanced](media/tartan-rgbt-balanced.mp4) | 6,950,637 | 默认双上下文、独立调度、FP32 |
| [responsive](media/tartan-rgbt-responsive.mp4) | 6,969,239 | 单上下文、最小预取、FP32 |

两者都是 H.264、1280×800、无音频、296 帧、10 FPS、29.6 秒。保留的 JPG 封面来自第 148 帧。README 使用另行生成的全时长 GIF 动态预览。视频播放时间经过压缩有效组的时间轴处理，与原始序列采集时长和处理 FPS 不同。不存在保持上一帧填充源空隙的画面；全部有效帧保留，6 组 FFC 无效输入单独计数。画面中的单帧计算计时不能直接换算成并行完整流程 FPS。

| 文件 | SHA-256 |
| --- | --- |
| balanced | `d04d6930a7f2802201e26bd5ff8c98479d72d43adc69b301b17bab01eb43cc8d` |
| responsive | `91f661a3611fb8e50b897efac69fb83eaf0714cbc20b083c8f34ced4ca1ef472` |

逐帧解码验证、无重复 RGB 面板检查和原文件哈希记录在 [media_validation.json](../reports/published/tartan_latency/media_validation.json)。编码参数见 [balanced](../reports/published/tartan_latency/balanced_video.json) / [responsive](../reports/published/tartan_latency/responsive_video.json)。元数据中的文件路径对应仓库内的完整 MP4。

来源：CMU AirLab [TartanRGBT 固定版本](https://huggingface.co/datasets/theairlabcmu/TartanRGBT/tree/23475c0fa9190154e6f1fd78f6b82b29f9d508a0)。作者归属与 BSD-3-Clause-Clear 声明见 [THIRD_PARTY.md](../THIRD_PARTY.md) 及 [完整许可](licenses/AnyThermal-LICENSE.txt)。修改包括几何配准、检测框、估计深度/位置、拼接面板和文字叠加。视频、GIF 和缩略图不被项目 MIT 许可证重新授权。

README 直接嵌入循环 GIF，完整 MP4 在下方单独链接。GIF 为 480×300、5 FPS、148 帧、29.6 秒，保持整段播放时长，降低分辨率和采样帧率以控制体积。原 MP4 未重新编码。GIF 是展示预览，不用于精度或性能评价；元数据与来源哈希见 [readme_previews.json](../reports/published/readme_previews.json)。

GitHub 可以根据用户的系统/账号设置暂停动图。如果预览静止，检查 **Settings → Accessibility → Motion → Autoplay animated images**，按个人偏好选择 Enabled；见 [GitHub 官方说明](https://docs.github.com/en/account-and-profile/how-tos/account-settings/managing-accessibility-settings)。README 无法替读者更改此偏好。

生成 GIF 需要安装系统 FFmpeg/ffprobe，然后在仓库根目录运行：

```bash
python3 scripts/render_readme_previews.py
```

如另需带控件的 GitHub MP4 播放器，可将 MP4 上传为附件后引用实际生成的附件 URL；本项目无需外部附件即可显示 GIF 预览。GitHub 的媒体格式和附件大小要求见 [附件文档](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)。

重新生成请按 [Jetson 指南](JETSON.md) 运行，输出到 `results/demo/`，不覆盖已经发布且有固定哈希的演示。
