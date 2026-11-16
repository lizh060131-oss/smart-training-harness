# 智训引擎 V4 容器接口

容器启动命令为 `python -u /app/harness/entry.py`。接口路径参考队友 V6 镜像，训练、选择和记录沿用本项目的 `RoundEngine`。接口配置见 `/app/CONTRACT.json`。

## 训练与交付

默认 `HARNESS_MODE=loop`，按 A、B、C 顺序运行，每项 10 轮，默认 `--policy auto`。预切分数据按 `batch01.jsonl` 至 `batch10.jsonl` 原顺序使用。也可挂载每个任务的 `train.jsonl`，由入口构造批次。A 按源批次组织，单一源批次按气体类别分层。B 按轴承组织。C 优先按整台发动机和子集组织。若同目录提供 `validation.jsonl`，会检查训练与验证样本 ID 的重叠，B/C 还会按可用的设备标识检查。`--stream official` 仅计算已挂载的 `validation.jsonl` 指标。

输入为只读挂载。默认读取 `/data/<任务>/` 和 `/previous/<任务>/checkpoint.npz`。也识别 `/previous/model/<任务>/checkpoint.npz`、`/previous/output/model/<任务>/checkpoint.npz` 和单任务 `/previous/checkpoint.npz`。指定 `--task A|B|C --checkpoint <文件>` 可直接用指定检查点。镜像中没有训练数据或可回退的模型权重。

每项训练完成后写入 `/model/<任务>/checkpoint.npz`、`/model/<任务>/manifest.json`，并在 `/output/model/<任务>/` 保留副本。`manifest.json` 记录起始和最终检查点的 SHA-256、完成轮数及 PyTRiO 参与轮数。`/output/<任务>/summary.json`、`engine_log.jsonl` 和各候选子目录可追溯训练、选择与模型血缘。续训需要新的输出目录和上次产物作为父检查点。

`HARNESS_DATA`、`HARNESS_PREVIOUS` 和 `HARNESS_MODEL_OUT` 可覆盖挂载路径。显式参数 `--data`、`--previous` 和 `--model-out` 优先。可用 `--windows` 或 `--rounds` 缩短本地联调。

## 预测与输入自检

`HARNESS_MODE=serve` 从 `/model/<任务>/checkpoint.npz` 或 `/output/model/<任务>/checkpoint.npz` 加载产物，也接受 `--checkpoint`。stdin 每行接受 `{"inputs": …}`，可加 `"task": "A"`、`"B"` 或 `"C"`；省略时按特征结构选择。stdout 每行返回一个 `outputs` 对象。A 返回六类分数，B 返回剩余分钟，C 返回剩余循环。Python 入口为 `baselines.predict.predict(model, inputs)`。

`HARNESS_MODE=selftest` 检查任务检查点、数据路径和批次数，以及挂载的独立公开验证数据重叠，随后退出。完整训练请使用 `loop`。

## 大模型与记录

`--policy auto` 读取 `PYTRIO_API_KEY`、`PYTRIO_BASE_URL`、`PYTRIO_MODEL`（也接受 `PETRIO_`）。同组配置完整时，也识别 `SAMPLER_API_KEY`、`SAMPLER_ENDPOINT`、`SAMPLER_MODEL`，或 `HARNESS_LLM_KEY`、`HARNESS_LLM_ENDPOINT`、`HARNESS_LLM_MODEL`。未配置或调用失败时，记录实际采用的规划方式。`--policy llm` 要求配置完整，并检查至少三轮采用模型规划（短联调为实际轮数）。

SwanLab 可用 `SWANLAB_MODE=cloud`、`SWANLAB_API_KEY` 及 `SWANLAB_WORKSPACE` 配置项目上传。`--swan off` 关闭同步，本地训练日志仍正常输出。
