# Terminal Classifier — Design Spec

## Overview

An HTTP API service that classifies AI CLI terminal output (Claude Code, Codex CLI, Gemini CLI, etc.) into one of three states: `waiting_input`, `processing`, or `completed`. The classification is consumed by an orchestrator agent that manages multiple AI CLI sessions.

## Classification Model

**facebook/bart-large-mnli** — a ~400M parameter zero-shot classification model. It takes arbitrary text and a set of candidate labels, returning confidence scores for each label without requiring task-specific training.

- Zero-shot: no labeled training data needed
- Handles unseen phrasing from new/updated CLI tools
- ~1.6GB RAM footprint
- ~50-200ms inference latency on CPU

### Labels

| Label | Meaning |
|-------|---------|
| `waiting_input` | The CLI is prompting the user for input |
| `processing` | The CLI is still working (thinking, generating, etc.) |
| `completed` | The CLI has finished its task |

## Architecture

Single FastAPI service:

1. Loads bart-large-mnli into RAM at startup (stays resident)
2. Exposes `POST /classify` and `GET /health`
3. Authenticates via `x-api-key` header

### Project Structure

```
terminal-classifier/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, startup model loading
│   ├── classifier.py    # Model wrapper, classify() function
│   └── config.py        # Settings from .env
├── install.sh           # System deps + pip install + model download + systemd
├── requirements.txt
├── .env.example
├── .gitignore
├── CLAUDE.md
└── README.md
```

## API Design

### `POST /classify`

**Headers:**
- `x-api-key: <API_KEY>` (required)
- `Content-Type: application/json`

**Request:**
```json
{
  "text": "? Do you want to proceed? (y/n)"
}
```

**Response:**
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

- `classification` — winning label
- `confidence` — score of winning label (0-1)
- `scores` — all label scores for orchestrator-side thresholding

**Errors:**
- `401 Unauthorized` — missing or invalid `x-api-key`
- `422 Unprocessable Entity` — missing or invalid `text` field

### `GET /health`

**Response:**
```json
{
  "status": "ok"
}
```

No authentication required.

## Configuration

`.env.example`:
```
PORT=8980
API_KEY=your-api-key-here
CPU_CORES=16
EXPOSE=0
```

- `PORT` — HTTP listen port (default: 8980)
- `API_KEY` — required for `x-api-key` auth
- `CPU_CORES` — maximum number of CPU cores PyTorch may use for inference; app sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` from this value (up to, not pinned)
- `EXPOSE` — set to `1` to bind on `0.0.0.0` (accessible from network); default `0` binds to `127.0.0.1` (localhost only)

## Runtime & Performance

- Model loaded once at startup, cached in `~/.cache/huggingface/`
- `CPU_CORES` env var sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` as an upper limit (uses up to that many cores, not a hard pin)
- Uvicorn with 1 worker (single model instance in memory)
- Inference runs in a thread pool executor so the async event loop and health endpoint stay responsive
- Expected: ~50-200ms per classification on 16-core CPU

## Install & Systemd

`install.sh` performs:

1. Install system packages (`python3`, `python3-venv`, `pip`)
2. Create Python venv in project directory
3. Install pip dependencies from `requirements.txt`
4. Download and cache bart-large-mnli model
5. Copy `.env.example` to `.env` if `.env` doesn't exist
6. Create `/etc/systemd/system/terminal-classifier.service`:
   - `WorkingDirectory` = project path
   - `EnvironmentFile` = `.env`
   - Runs `venv/bin/uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - `Restart=always`
   - `After=network.target`
7. `systemctl daemon-reload && systemctl enable --now terminal-classifier`

Script is idempotent — safe to run again on reinstall or server migration.

## Git

- Default branch: `main` (not master)
- `.gitignore`: venv, `.env`, `__pycache__`, `.cache`, model files
