# 常用开发命令入口。
# 通过短命令统一安装、启动、测试和容器操作，降低记忆成本。

PYTHON := /opt/anaconda3/envs/tmf_project/bin/python
PIP := $(PYTHON) -m pip
UVICORN := $(PYTHON) -m uvicorn
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff

.PHONY: install dev api test lint docker-up docker-up-aliyun docker-down env

env:
	@echo "Python: $(PYTHON)"
	@$(PYTHON) --version

install:
	# 安装项目本体和 dev 依赖；-e 代表源码改动后无需重复安装。
	$(PIP) install -e ".[dev]"

dev: api

api:
	# 本地开发直接起 FastAPI，并开启热重载。
	$(UVICORN) apps.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	# 执行仓库内全部测试。
	$(PYTEST) tests -q

lint:
	# 这里仅展示检查结果，不让 lint 失败阻断命令。
	$(RUFF) check core apps tests || true

docker-up:
	# 构建并后台启动完整 compose 环境。
	docker compose up --build -d

docker-up-aliyun:
	# 构建时临时使用阿里云 PyPI 镜像，适合网络较慢或大包下载不稳定场景。
	PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple docker compose up --build -d

docker-down:
	# 停止 compose 管理的服务。
	docker compose down
