# 智训引擎

第二届北京市大学生「人工智能+」创新大赛题目九的 Harness 源码。系统从赛方检查点开始，在 CPU 上对气体识别、轴承剩余寿命和发动机剩余寿命模型进行逐批续训。流程包括漂移诊断、受限 PyTRiO 工具调用、候选训练、上线或回滚和实验记录。

## 运行

当前参赛版本为 V5，使用赛方 Harness SDK 1.0.2。平台按轮提供公开数据和父检查点，Harness 通过 SDK 调用 PyTRiO，并提交真实续训得到的候选和选模理由。构建和本地协议验证见 [V5 SDK 接入说明](docs/SDK接入说明.md)。

使用 Python 3.11 可在本地联调单个任务：

```bash
python -m pip install -r 1/harness_v2/requirements.txt
python 1/harness_v2/main.py --task A --data <批次目录> --baseline <官方检查点.npz> --out <输出目录> --rounds 10
```

`--task` 可选 A、B、C，对应的官方检查点须与任务配套。运行时可配置 `PYTRIO_BASE_URL`、`PYTRIO_API_KEY` 和 `PYTRIO_MODEL`，兼容 `PETRIO_` 前缀。启用实时 SwanLab 时再配置 `SWANLAB_API_KEY`。输出目录包含各轮训练检查点、`engine_log.jsonl` 与 `summary.json`。

- [V4 本地批量运行、Docker 打包和独立验证](docs/运行说明.md)
- [系统设计与门控方法](docs/设计说明.md)
- [官方基线与第三方来源声明](docs/开源合规说明.md)
- [工业模型训练记录](https://swanlab.cn/@lizh-131/smart-training-harness)
