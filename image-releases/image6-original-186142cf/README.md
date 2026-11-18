# 原版 Image6 源码

这里的代码、配置与依赖声明逐文件提取自已提交的 linux/amd64 镜像。文件 SHA-256 见 [发布清单](release.json)。

- 原始归档 SHA-256 为 `186142cf2bc70a95dd2472c09b17316e21e2d4a725b827d37a00bc9a12b5849d`。
- SDK 版本为 `1.0.2`。容器入口为 `python3 /app/harness.py`，平台传入 `describe`、`train` 或 `predict`。
- `contest_submission.py` 提供续训与预测，`configs/` 定义 A、B、C 任务配置，`baselines/` 提供数据处理及模型实现。
- 运行需准备原版镜像或同版 SDK 运行环境，以及平台提供的输入数据和父检查点。

本地验收使用赛事公开数据与模拟宿主。赛事平台隐藏集成绩见平台报告。
