"""
tests/test_platform_v1.py

Platform-v1 API smoke tests:
- config submission persistence
- admin approval without deployment
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.main import app


def test_submit_and_approve_strategy_config_flow():
    with TestClient(app) as client:
        submit = client.post(
            "/strategies/submit-config",
            json={
                "submitter_name": "Test User",
                "contact": "test@example.com",
                "mode": "beginner",
                "template_strategy": "momentum",
                "parameters": {"lookback": 24, "aggression": 0.001, "order_qty": 6},
                "visibility": "public",
            },
        )
        assert submit.status_code == 201
        payload = submit.json()
        submission_id = payload["submission_id"]
        assert submission_id.startswith("sub_")
        assert payload["status"] == "pending"

        listed = client.get("/strategies/submissions?limit=50")
        assert listed.status_code == 200
        rows = listed.json()["items"]
        row = next((item for item in rows if item["submission_id"] == submission_id), None)
        assert row is not None
        assert row["template_strategy"] == "momentum"
        assert row["status"] == "pending"

        approved = client.post(
            f"/strategies/submissions/{submission_id}/approve",
            json={"deploy": False},
        )
        assert approved.status_code == 200
        approve_payload = approved.json()
        assert approve_payload["deployed"] is False
        assert approve_payload["submission"]["status"] == "approved"
