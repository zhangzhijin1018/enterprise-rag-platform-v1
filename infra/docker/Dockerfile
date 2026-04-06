# 后端 + 前端一体化镜像构建文件。
# 构建阶段先打包前端，再把静态资源复制进 Python 服务镜像。

FROM node:20-bookworm-slim AS webbuilder
WORKDIR /build
COPY apps/web/package.json ./
# 优先安装前端依赖，便于利用 Docker 层缓存。
RUN npm install
COPY apps/web/ ./
# 产出前端静态资源目录 `dist/`。
RUN npm run build

FROM python:3.11-slim

WORKDIR /app

# 安装部分 Python 包可能需要编译环境。
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY core ./core
COPY apps/api ./apps/api
COPY apps/worker ./apps/worker
COPY settings.py ./
# 把前端构建产物复制到后端项目目录，供 FastAPI 挂载 `/ui/`。
COPY --from=webbuilder /build/dist ./apps/web/dist

# 安装 Python 依赖和项目本体。
RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV VECTOR_STORE_PATH=/data/vector_store

RUN mkdir -p /data/vector_store

EXPOSE 8000

# 容器启动后直接运行 API 服务。
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
