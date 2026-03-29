"""Tests for InvoiceAgent HTTP endpoints.

Uses httpx ASGITransport to test the FastAPI app without starting a real server.
"""
from __future__ import annotations

import pytest
import httpx

from invoice_agent.server.app import app

pytestmark = pytest.mark.asyncio


@pytest.fixture
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    async def test_health_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /tasks
# ---------------------------------------------------------------------------

class TestTasks:
    async def test_tasks_returns_3(self, client):
        resp = await client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert len(data["tasks"]) == 3

    async def test_tasks_have_required_keys(self, client):
        resp = await client.get("/tasks")
        data = resp.json()
        for task in data["tasks"]:
            assert "task_id" in task
            assert "name" in task
            assert "difficulty" in task
            assert "required_fields" in task

    async def test_tasks_include_action_schema(self, client):
        resp = await client.get("/tasks")
        data = resp.json()
        assert "action_schema" in data
        assert "observation_schema" in data


# ---------------------------------------------------------------------------
# /reset
# ---------------------------------------------------------------------------

class TestReset:
    async def test_reset_empty_body(self, client):
        """POST /reset with no body returns valid JSON (not 422)."""
        resp = await client.post("/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data

    async def test_reset_with_body(self, client):
        """POST /reset with task_id works."""
        resp = await client.post("/reset", json={"task_id": "easy"})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data

    async def test_reset_with_seed(self, client):
        resp = await client.post("/reset", json={"task_id": "easy", "seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert data["observation"]["invoice_text"]

    async def test_reset_medium_task(self, client):
        resp = await client.post("/reset", json={"task_id": "medium"})
        assert resp.status_code == 200

    async def test_reset_hard_task(self, client):
        resp = await client.post("/reset", json={"task_id": "hard"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /step
# ---------------------------------------------------------------------------

class TestStep:
    async def test_step_empty_body(self, client):
        """POST /step with no body should not error (defaults to submit)."""
        # First reset to have a valid session
        await client.post("/reset", json={"task_id": "easy"})
        resp = await client.post("/step")
        assert resp.status_code == 200

    async def test_step_with_action(self, client):
        await client.post("/reset", json={"task_id": "easy", "seed": 42})
        resp = await client.post("/step", json={
            "action": {
                "action_type": "extract_field",
                "field_name": "invoice_number",
                "field_value": "TEST-123",
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data


# ---------------------------------------------------------------------------
# /baseline
# ---------------------------------------------------------------------------

class TestBaseline:
    async def test_baseline_returns_scores(self, client):
        """POST /baseline returns scores for all 3 tasks."""
        resp = await client.post("/baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert "baseline_scores" in data
        scores = data["baseline_scores"]
        assert "easy" in scores
        assert "medium" in scores
        assert "hard" in scores
        for task_id, result in scores.items():
            assert "score" in result
            assert isinstance(result["score"], float)
            assert 0.0 <= result["score"] <= 1.0

    async def test_baseline_easy_nonzero(self, client):
        """Heuristic baseline should get a nonzero score on easy task."""
        resp = await client.post("/baseline")
        data = resp.json()
        assert data["baseline_scores"]["easy"]["score"] > 0.0


# ---------------------------------------------------------------------------
# /grader
# ---------------------------------------------------------------------------

class TestGraderEndpoint:
    async def test_grader_empty_body(self, client):
        resp = await client.post("/grader")
        assert resp.status_code == 200
        data = resp.json()
        assert "grader_score" in data

    async def test_grader_with_params(self, client):
        resp = await client.post("/grader", json={"task_id": "medium", "seed": 7})
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "medium"
        assert data["seed"] == 7
