# 智训引擎 Image6 优化版

本版本基于原版 Image6，使用赛事 Harness SDK 1.0.2 和 Linux/amd64。每轮读取平台指定的父检查点及训练、验证数据，提交续训检查点、父子摘要和实际训练记录。

训练和验证数据在同一轮解析一次，再供各候选复用。B 比较两组各训练 1 个 epoch 的候选。预测入口支持平台指定的检查点路径，逐行输出分类或回归结果。

## 本地验证

- 用公开训练和验证数据，固定本地模型回执，各任务单轮重复 3 次。A、B、C 的进程 CPU 时间中位数分别从 0.91、2.58、19.64 秒降至 0.39、0.91、7.70 秒。A、C 在此对照中选出的检查点与原版逐字节一致。
- B 在 24 个十轮本地场景中按整只轴承留出，最终 RMSE 均值从 462.73 分钟降至 445.27 分钟，18 个场景改善、1 个持平、5 个退步。
- 优化版的平台试运行和隐藏评测待验证。V5 源码和实验记录对应早期版本，本页构建使用 `image6-optimized/`。


[打开本地十轮对照的 SwanLab 实验记录](https://swanlab.cn/@lizh-131/smart-training-harness/runs/4c14e4f3e55a/chart)

## 构建与导出

先加载团队原版 Image6 基础镜像。标签为 `smart-training-harness:sdk-1.0.2-v1`，原版归档的 SHA-256 为 `186142cf2bc70a95dd2472c09b17316e21e2d4a725b827d37a00bc9a12b5849d`。在仓库根目录使用能读取本机镜像的 Docker 构建器运行以下命令。

```sh
docker load --input image6-sdk-linux-amd64.tar
docker image inspect smart-training-harness:sdk-1.0.2-v1 --format '{{.Id}}'
docker buildx build --platform linux/amd64 --load --network none --pull=false --provenance=false --sbom=false -t smart-training-harness:sdk-1.0.2-image6-optimized image6-optimized
docker save --platform linux/amd64 --output image6-sdk-optimized-linux-amd64.tar smart-training-harness:sdk-1.0.2-image6-optimized
```

已导出文件为 `image6-sdk-optimized-linux-amd64.tar`，SHA-256 为 `079bc0d3aa1e7fd529d1caa10ea2d304447979e7911aa381f42d97e0f2334b69`。重新构建和导出后，请以新文件的摘要为准。
