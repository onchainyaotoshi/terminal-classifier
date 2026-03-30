# Terminal Classifier

HTTP API that classifies AI CLI terminal output (Claude Code, Codex CLI, Gemini CLI, etc.) into three states:

| State | Meaning |
|-------|---------|
| `waiting_input` | CLI is prompting for user input |
| `processing` | CLI is still working |
| `completed` | CLI has finished |

Uses [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) for zero-shot classification.

## Getting Started

```bash
# Clone the repo
git clone https://github.com/onchainyaotoshi/terminal-classifier.git
cd terminal-classifier

# Run the installer (installs Python, deps, model, and systemd service)
sudo bash install.sh

# Set your API key
nano .env
```

The install script handles everything: system packages, Python venv, pip dependencies, model download (~1.5GB), `.env` setup, and systemd service creation.

## Configuration

Edit `.env` (created from `.env.example` on first install):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8980` | HTTP listen port |
| `API_KEY` | — | Required. Key for `x-api-key` auth |
| `CPU_CORES` | `16` | Max CPU cores for model inference |
| `EXPOSE` | `0` | Set to `1` to bind on `0.0.0.0` (network accessible). Default binds to `127.0.0.1` (localhost only) |

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
