# 构建、容器联调和导出命令见 docs/运行说明.md。
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt

COPY baselines/ ./baselines/
COPY configs/ ./configs/
COPY engine.py main.py pytrio_tools.py bearing_gate.py ./

# 运行时注入 SWANLAB_API_KEY、PETRIO_API_KEY、PETRIO_BASE_URL 和 PETRIO_MODEL。
ENV SWANLAB_MODE=cloud \
    SWANLAB_PROJECT_NAME=smart-training-harness \
    PYTHONUTF8=1

ENTRYPOINT ["python", "main.py"]
