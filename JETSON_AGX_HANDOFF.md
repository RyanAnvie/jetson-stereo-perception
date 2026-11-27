# Jetson AGX 交接入口

当前公开版使用固定外参、RGB 运动补偿、GPU 配准和 LCAFNet FP32 双模态检测。

- [README：功能、视频和实测结果](README.md)
- [Jetson 部署：依赖、数据、导出、构建与运行](docs/JETSON.md)
- [性能与延迟：定义、硬件条件和逐帧证据](docs/BENCHMARKS.md)
- [架构：标定、时序补偿和精度限制](docs/ARCHITECTURE.md)
- [开发与发布：检查、打包和本地生成文件](docs/DEVELOPMENT.md)

使用者在自己的 Jetson 上构建 engine。公开资料不依赖某台设备的 IP、账户、密码或绝对工作目录。项目许可证见 [LICENSE](LICENSE)，第三方内容见 [THIRD_PARTY.md](THIRD_PARTY.md)。
