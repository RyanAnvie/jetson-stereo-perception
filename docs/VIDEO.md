# GitHub 视频交付

新增 KITTI 三维目标演示：`results/kitti/axg_kitti_3d_demo.mp4`，约 24 秒、H.264、1280×800、10 fps，低于 10 MB。精确参数在 `reports/kitti_video_metadata.json`，预览图为 `docs/media/kitti-preview.jpg`。它展示真实检测框、双目深度、XYZ 位置表和离线参考误差。KITTI 派生视频与预览遵循 **CC BY-NC-SA 3.0**，上传时保留 `THIRD_PARTY.md` 中的作者、来源、修改说明和许可链接。

以下保留此前 TartanRGBT 演示的交付信息。

文件：`results/demo/axg_multimodal_demo.mp4`。H.264、1280×800、10 fps、无音频，约 44 秒，低于 10 MB；实际编码参数和大小见 `reports/video_metadata.json`。

视频展示真实 TartanRGBT 数据：RGB 原始检测预测、同步热像、FoundationStereo 估计深度、RGB 着色三维点云。片头片尾说明了热像标定阻塞、检测误检及 Jetson 未测状态。源数据间隔按时间戳保留，10 fps 时间格中停留上一幅图像，不生成中间测量。

1. 在目标 GitHub 仓库的 Issue、PR 或评论输入框拖入 MP4，等待上传完成。
2. 将 GitHub 返回的视频附件地址放到 README 的演示位置。视频仍可作为 Release 附件提供。
3. 随视频保留 `THIRD_PARTY.md` 的来源及许可说明；预览图在 `docs/media/demo-preview.jpg`。

GitHub 官方文档当前给出的免费方案视频附件上限为 10 MB，并推荐 H.264：[Attaching files](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)。本项目自动检查最终 MP4 小于 10,000,000 字节。

没有进行远端上传、提交或推送。`.gitignore` 有意排除所有 MP4、原始视频、数据和权重。将视频作为网页附件上传即可，不需要强行加入 Git 历史。
