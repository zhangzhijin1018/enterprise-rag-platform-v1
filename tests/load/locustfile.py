"""
Locust load test (run: locust -f tests/load/locustfile.py --host http://localhost:8000)
"""

from locust import HttpUser, between, task


class ChatUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(3)
    def health(self) -> None:
        self.client.get("/healthz")

    @task(1)
    def chat(self) -> None:
        self.client.post(
            "/chat",
            json={"question": "测试问题：如何提交工单？", "top_k": 4, "stream": False},
        )
