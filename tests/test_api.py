from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    mock_classifier = MagicMock()
    mock_classifier.classify.return_value = {
        "classification": "completed",
        "confidence": 0.95,
        "scores": {"waiting_input": 0.02, "processing": 0.03, "completed": 0.95},
    }
    with patch("app.main.classifier", mock_classifier), \
         patch("app.main.settings") as mock_settings:
        mock_settings.api_key = "test-key"
        from app.main import app
        yield TestClient(app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_returns_result(client):
    response = client.post(
        "/classify",
        json={"text": "Done."},
        headers={"x-api-key": "test-key"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["classification"] == "completed"
    assert data["confidence"] == 0.95
    assert "scores" in data


def test_classify_missing_api_key(client):
    response = client.post("/classify", json={"text": "hello"})
    assert response.status_code == 422


def test_classify_wrong_api_key(client):
    response = client.post(
        "/classify",
        json={"text": "hello"},
        headers={"x-api-key": "wrong-key"},
    )
    assert response.status_code == 401


def test_classify_missing_text(client):
    response = client.post(
        "/classify",
        json={},
        headers={"x-api-key": "test-key"},
    )
    assert response.status_code == 422
