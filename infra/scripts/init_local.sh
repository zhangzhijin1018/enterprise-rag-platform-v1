#!/usr/bin/env bash
# 本地初始化脚本。
# 用于快速创建虚拟环境、安装依赖并准备基础目录。

set -euo pipefail
# 先切到仓库根目录，确保后续相对路径稳定。
cd "$(dirname "$0")/../.."
# 创建项目专用虚拟环境，避免污染全局 Python。
python -m venv .venv
# 激活虚拟环境后再安装依赖。
source .venv/bin/activate
# 先升级 pip，减少部分依赖安装兼容问题。
pip install -U pip
# 安装项目和 dev 依赖。
pip install -e ".[dev]"
# 如果还没有 `.env`，就从示例文件复制一份。
cp -n .env.example .env || true
# 准备索引目录和评测报告目录。
mkdir -p data/vector_store data/eval_reports
# 给使用者一个下一步提示。
echo "Ready. Activate venv: source .venv/bin/activate && make api"
