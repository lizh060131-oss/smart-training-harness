# 智训引擎 V5 与赛方 Harness SDK 1.0.2

V5 使用赛方 `contest.harness.workspace.v2` 协议。`Dockerfile.sdk` 安装 `1/harness_v2/vendor/contest-harness-sdk-1.0.2.tar.gz`，并使用包内的官方 `/app/harness.py` 模板。选手实现为 `1/harness_v2/contest_submission.py`。原有 `Dockerfile` 和本地批量续训入口可继续用于对照实验。

## 平台逐轮执行

- `describe` 声明 `host_gateway_rpc` 及分类、回归任务。平台分别调度 A、B、C 的轮次。
- `train` 校验父产物清单和检查点摘要，按本轮合同读取公开训练、验证投影。通过 SDK 读历史并调用平台模型，让模型根据父模型、历史和门控代码提出一组续训参数。
- 候选在 `/workspace` 续训。模型输出仅由受限策略解析器读取。回复无法解析时，执行明确标记的保守参数；模型调用失败则终止本轮，由 SDK 报错。
- `submit_candidate` 交付真实续训检查点、策略声明及训练日志。若平台同时提供父模型和候选的公开指标，则据此判断；否则用同一公开验证投影做成对比较。在 `finish` 中记录决策、实际读取的历史标识和本轮平台返回的模型调用号。
- `predict` 使用 `/model` 和平台指定的 `CONTEST_ARTIFACT_PATH`，每输入一条 JSONL 就立即返回一条预测。

API Key 由平台密钥栏管理。镜像通过 SDK 的宿主通道调用模型，镜像内无训练数据、起始权重或密钥。

## 本地构建与协议验证

从赛方页面的「Harness SDK 与使用文档」下载 `contest-harness-sdk-1.0.2.tar.gz`，放入 `1/harness_v2/vendor/`。文件 SHA-256 为 `c676c8beeef69cece9cbec5a633d05dbb9832eb22b6f761913f2315bd1f85133`。本地联调还需要把三份「训练起点」压缩包分别解压至 `data/official_baselines/<包名>/`。

在项目根目录运行：

```powershell
docker buildx build --platform linux/amd64 --load --provenance=false --sbom=false -f .\1\harness_v2\Dockerfile.sdk -t smart-training-harness:competition-v5 .\1\harness_v2
python .\tools\test_contest_sdk_image.py
docker save --platform linux/amd64 -o .\smart-training-harness-competition-v5-sdk102-linux-amd64.tar smart-training-harness:competition-v5
```

验证脚本使用 A/B/C 随包公开数据和检查点，模拟赛方宿主回复。检查只读容器内的实际续训、父模型血缘、保留与采用分支、实际提交的调用回执，以及预测在 stdin 结束前返回。A 连续执行两轮，并在第二轮使用第一轮经过真实训练的产物。模拟回执和模拟公开测量仅用于本地协议验证。真实 PyTRiO 接入及赛方隐藏评测以平台试运行报告为准。
