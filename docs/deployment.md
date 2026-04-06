# Deployment

## Local

```bash
cp .env.example .env
# 推荐使用 Python 3.10+（如 Conda：conda activate tmf_project）
pip install -e ".[dev]"
make api
```

## Docker Compose

```bash
export OPENAI_API_KEY=...   # 可选
docker compose up --build
```

Prometheus 通过 `infra/prometheus/prometheus.yml` 抓取 `api:8000/metrics`（同一 compose 网络）。

## Kubernetes

```bash
docker build -f infra/docker/Dockerfile -t enterprise-rag-platform:latest .
kubectl apply -f infra/k8s/pvc.yaml
kubectl apply -f infra/k8s/redis.yaml
kubectl apply -f infra/k8s/deployment.yaml
```

按需为 `OPENAI_API_KEY`、向量 PVC 与镜像仓库地址打补丁。

## Observability

- `OTEL_EXPORTER_OTLP_ENDPOINT` — gRPC OTLP 收集端（可选）。
- 日志：结构化 JSON 至 stdout。
