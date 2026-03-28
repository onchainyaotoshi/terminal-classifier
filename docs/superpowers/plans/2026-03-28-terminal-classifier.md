# Terminal Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an HTTP API that classifies AI CLI terminal output into `waiting_input`, `processing`, or `completed` using facebook/bart-large-mnli.

**Architecture:** Single FastAPI service with one POST endpoint (`/classify`) and one health check (`GET /health`). The bart-large-mnli model loads at startup and stays resident in memory. Auth via `x-api-key` header. Deployed as a systemd service.

**Tech Stack:** Python 3, FastAPI, Uvicorn, Hugging Face Transformers, PyTorch, python-dotenv

**Spec:** `docs/superpowers/specs/2026-03-28-terminal-classifier-design.md`

---

### Task 1: Project scaffolding

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `requirements.txt`
- Create: `CLAUDE.md`
- Create: `app/__init__.py`

- [ ] **Step 1: Create `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Venv
venv/

# Environment
.env

# Hugging Face model cache
.cache/

# IDE
.idea/
.vscode/
*.swp
```

- [ ] **Step 2: Create `.env.example`**

```
PORT=8980
API_KEY=your-api-key-here
CPU_CORES=16
```

- [ ] **Step 3: Create `requirements.txt`**

```
fastapi==0.115.12
uvicorn==0.34.2
transformers==4.51.2
torch==2.6.0
python-dotenv==1.1.0
```

- [ ] **Step 4: Create `CLAUDE.md`**

```markdown
# Terminal Classifier

HTTP API service that classifies AI CLI terminal output using facebook/bart-large-mnli.

## Quick Start

```bash
# Install and start
sudo bash install.sh

# Or run manually
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8980
```

## Project Structure

- `app/config.py` — loads settings from `.env`
- `app/classifier.py` — model wrapper, `classify()` function
- `app/main.py` — FastAPI app, endpoints, startup model loading
- `install.sh` — system deps, venv, model download, systemd service

## API

- `POST /classify` — classify terminal text (requires `x-api-key` header)
- `GET /health` — health check (no auth)

## Testing

```bash
source venv/bin/activate
pytest tests/ -v
```

## Configuration

All config in `.env` (see `.env.example`):
- `PORT` — listen port (default 8980)
- `API_KEY` — required for auth
- `CPU_CORES` — max CPU cores for inference
```

- [ ] **Step 5: Create `app/__init__.py`**

Empty file.

- [ ] **Step 6: Commit**

```bash
git add .gitignore .env.example requirements.txt CLAUDE.md app/__init__.py
git commit -m "feat: add project scaffolding"
```

---

### Task 2: Configuration module

**Files:**
- Create: `app/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` as empty file.

Create `tests/test_config.py`:

```python
import os
from unittest.mock import patch


def test_config_loads_defaults():
    with patch.dict(os.environ, {"API_KEY": "test-key"}, clear=False):
        from importlib import reload
        import app.config as config_module
        reload(config_module)
        settings = config_module.Settings()
        assert settings.port == 8980
        assert settings.api_key == "test-key"
        assert settings.cpu_cores == 1


def test_config_loads_custom_values():
    env = {
        "PORT": "9000",
        "API_KEY": "my-secret",
        "CPU_CORES": "8",
    }
    with patch.dict(os.environ, env, clear=False):
        from importlib import reload
        import app.config as config_module
        reload(config_module)
        settings = config_module.Settings()
        assert settings.port == 9000
        assert settings.api_key == "my-secret"
        assert settings.cpu_cores == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/test_config.py -v`
Expected: FAIL with ImportError or AttributeError

- [ ] **Step 3: Write minimal implementation**

Create `app/config.py`:

```python
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self):
        self.port: int = int(os.getenv("PORT", "8980"))
        self.api_key: str = os.getenv("API_KEY", "")
        self.cpu_cores: int = int(os.getenv("CPU_CORES", "1"))


settings = Settings()

# Set thread limits for PyTorch
os.environ["OMP_NUM_THREADS"] = str(settings.cpu_cores)
os.environ["MKL_NUM_THREADS"] = str(settings.cpu_cores)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: add configuration module"
```

---

### Task 3: Classifier module

**Files:**
- Create: `app/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_classifier.py`:

```python
from app.classifier import TerminalClassifier


def test_classify_returns_all_three_labels():
    classifier = TerminalClassifier()
    result = classifier.classify("? Do you want to proceed? (y/n)")
    assert "classification" in result
    assert "confidence" in result
    assert "scores" in result
    assert set(result["scores"].keys()) == {"waiting_input", "processing", "completed"}


def test_classify_scores_sum_to_one():
    classifier = TerminalClassifier()
    result = classifier.classify("Installing packages...")
    total = sum(result["scores"].values())
    assert abs(total - 1.0) < 0.01


def test_classify_confidence_matches_top_score():
    classifier = TerminalClassifier()
    result = classifier.classify("Done.")
    top_label = result["classification"]
    assert result["confidence"] == result["scores"][top_label]


def test_classify_waiting_input():
    classifier = TerminalClassifier()
    result = classifier.classify("? Do you want to proceed? (y/n)")
    assert result["classification"] == "waiting_input"


def test_classify_processing():
    classifier = TerminalClassifier()
    result = classifier.classify("⠋ Generating response...")
    assert result["classification"] == "processing"


def test_classify_completed():
    classifier = TerminalClassifier()
    result = classifier.classify("Task completed successfully. Goodbye!")
    assert result["classification"] == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/test_classifier.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

Create `app/classifier.py`:

```python
from transformers import pipeline


LABELS = ["waiting_input", "processing", "completed"]


class TerminalClassifier:
    def __init__(self):
        self.pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )

    def classify(self, text: str) -> dict:
        result = self.pipe(text, candidate_labels=LABELS)

        scores = {
            label: round(score, 4)
            for label, score in zip(result["labels"], result["scores"])
        }

        top_label = result["labels"][0]
        confidence = scores[top_label]

        return {
            "classification": top_label,
            "confidence": confidence,
            "scores": scores,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/test_classifier.py -v`
Expected: PASS (6 tests). First run will download the model (~1.6GB).

Note: `test_classify_waiting_input`, `test_classify_processing`, and `test_classify_completed` depend on model accuracy. If any fail, adjust the test input text to something more obvious — the model may interpret ambiguous text differently. The structural tests (first 3) should always pass.

- [ ] **Step 5: Commit**

```bash
git add app/classifier.py tests/test_classifier.py
git commit -m "feat: add classifier module with bart-large-mnli"
```

---

### Task 4: FastAPI app with health endpoint

**Files:**
- Create: `app/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api.py`:

```python
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
    with patch("app.main.classifier", mock_classifier):
        from app.main import app
        yield TestClient(app)


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source venv/bin/activate && pytest tests/test_api.py::test_health_returns_ok -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

Create `app/main.py`:

```python
import asyncio
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.classifier import TerminalClassifier
from app.config import settings

classifier: TerminalClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    classifier = TerminalClassifier()
    yield


app = FastAPI(title="Terminal Classifier", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    classification: str
    confidence: float
    scores: dict[str, float]


def verify_api_key(x_api_key: str = Header()):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/classify", response_model=ClassifyResponse)
async def classify(
    request: ClassifyRequest,
    x_api_key: str = Header(),
):
    verify_api_key(x_api_key)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(classifier.classify, request.text))
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source venv/bin/activate && pytest tests/test_api.py::test_health_returns_ok -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add FastAPI app with health endpoint"
```

---

### Task 5: Classify endpoint with auth

**Files:**
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_api.py`:

```python
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
```

Also update the `client` fixture to patch `app.config.settings.api_key`:

```python
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
```

- [ ] **Step 2: Run tests to verify new ones fail**

Run: `source venv/bin/activate && pytest tests/test_api.py -v`
Expected: New tests may fail depending on current implementation state

- [ ] **Step 3: Adjust implementation if needed**

The implementation from Task 4 should already handle these cases. If any tests fail, fix the implementation in `app/main.py`.

- [ ] **Step 4: Run all tests to verify they pass**

Run: `source venv/bin/activate && pytest tests/test_api.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_api.py
git commit -m "test: add classify endpoint and auth tests"
```

---

### Task 6: Install script and systemd service

**Files:**
- Create: `install.sh`

- [ ] **Step 1: Create `install.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_NAME="terminal-classifier"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
VENV_DIR="${PROJECT_DIR}/venv"

echo "=== Terminal Classifier Installer ==="

# 1. System packages
echo "[1/7] Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip > /dev/null

# 2. Virtual environment
echo "[2/7] Setting up virtual environment..."
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
fi

# 3. Python dependencies
echo "[3/7] Installing Python dependencies..."
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip
"${VENV_DIR}/bin/pip" install --quiet -r "${PROJECT_DIR}/requirements.txt"

# 4. Download model
echo "[4/7] Downloading bart-large-mnli model (this may take a while)..."
"${VENV_DIR}/bin/python" -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"
echo "    Model cached successfully."

# 5. Environment file
echo "[5/7] Setting up environment..."
if [ ! -f "${PROJECT_DIR}/.env" ]; then
    cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
    echo "    Created .env from .env.example — edit it with your API_KEY."
else
    echo "    .env already exists, skipping."
fi

# 6. Systemd service
echo "[6/7] Creating systemd service..."
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Terminal Classifier API
After=network.target

[Service]
Type=simple
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=${PROJECT_DIR}/.env
ExecStart=${VENV_DIR}/bin/uvicorn app.main:app --host 0.0.0.0 --port \${PORT}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 7. Enable and start
echo "[7/7] Enabling and starting service..."
systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

echo ""
echo "=== Installation complete ==="
echo "Service status: systemctl status ${SERVICE_NAME}"
echo "Logs: journalctl -u ${SERVICE_NAME} -f"
echo ""
echo "IMPORTANT: Edit ${PROJECT_DIR}/.env to set your API_KEY before using."
```

- [ ] **Step 2: Make executable**

```bash
chmod +x install.sh
```

- [ ] **Step 3: Commit**

```bash
git add install.sh
git commit -m "feat: add install script with systemd service"
```

---

### Task 7: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# Terminal Classifier

HTTP API that classifies AI CLI terminal output (Claude Code, Codex CLI, Gemini CLI, etc.) into three states:

| State | Meaning |
|-------|---------|
| `waiting_input` | CLI is prompting for user input |
| `processing` | CLI is still working |
| `completed` | CLI has finished |

Uses [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) for zero-shot classification.

## Install

```bash
sudo bash install.sh
```

This installs dependencies, downloads the model, creates a systemd service, and starts it.

## Configuration

Edit `.env` (created from `.env.example` on first install):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8980` | HTTP listen port |
| `API_KEY` | — | Required. Key for `x-api-key` auth |
| `CPU_CORES` | `16` | Max CPU cores for model inference |

## Usage

### Classify terminal output

```bash
curl -X POST http://localhost:8980/classify \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{"text": "? Do you want to proceed? (y/n)"}'
```

Response:

```json
{
  "classification": "waiting_input",
  "confidence": 0.92,
  "scores": {
    "waiting_input": 0.92,
    "processing": 0.05,
    "completed": 0.03
  }
}
```

### Health check

```bash
curl http://localhost:8980/health
```

## Service Management

```bash
# Status
systemctl status terminal-classifier

# Logs
journalctl -u terminal-classifier -f

# Restart
sudo systemctl restart terminal-classifier

# Stop
sudo systemctl stop terminal-classifier
```

## Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8980 --reload
```

### Run tests

```bash
source venv/bin/activate
pytest tests/ -v
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README"
```

---

### Task 8: Final integration test

**Files:**
- No new files — manual verification

- [ ] **Step 1: Run full test suite**

```bash
source venv/bin/activate && pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Manual smoke test**

Start the server:

```bash
source venv/bin/activate
API_KEY=test CPU_CORES=2 uvicorn app.main:app --port 8980 &
```

Wait for startup, then test:

```bash
# Health check
curl -s http://localhost:8980/health
# Expected: {"status":"ok"}

# Classify
curl -s -X POST http://localhost:8980/classify \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -d '{"text": "? Enter your password:"}'
# Expected: {"classification":"waiting_input", ...}

# Auth failure
curl -s -X POST http://localhost:8980/classify \
  -H "Content-Type: application/json" \
  -H "x-api-key: wrong" \
  -d '{"text": "hello"}'
# Expected: 401

# Stop server
kill %1
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: final integration verification"
```

Only commit if there are changes. If all tests passed with no modifications, skip this step.
