# 后端 + 前端一体化镜像构建文件。
# 构建阶段先打包前端，再把静态资源复制进 Python 服务镜像。

FROM node:20-slim AS webbuilder
WORKDIR /build
COPY apps/web/package.json apps/web/package-lock.json apps/web/.npmrc ./
# 使用 lockfile 安装前端依赖，保证镜像构建结果更稳定。
RUN npm ci
COPY apps/web/ ./
# 产出前端静态资源目录 `dist/`。
RUN npm run build

FROM python:3.10-slim-bookworm

WORKDIR /app

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

# 安装部分 Python 包可能需要编译环境。
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update -o Acquire::Retries=3 \
    && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY core ./core
COPY apps/api ./apps/api
COPY apps/worker ./apps/worker
COPY settings.py ./
# 把前端构建产物复制到后端项目目录，供 FastAPI 挂载 `/ui/`。
COPY --from=webbuilder /build/dist ./apps/web/dist

# 安装 Python 依赖和项目本体；增大超时并启用重试，降低大包下载失败概率。
RUN pip install --disable-pip-version-check --retries 5 --timeout 300 -e .

ENV PYTHONUNBUFFERED=1
RUN mkdir -p /data/milvus

EXPOSE 8000

# 容器启动后直接运行 API 服务。
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
