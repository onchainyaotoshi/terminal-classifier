# Terminal Classifier

Classify AI CLI terminal output in real-time. Know exactly when a CLI is idle, waiting for confirmation, or still processing — useful for building automation, orchestration, and remote control for AI coding agents.

Supports **Claude Code**, **Codex CLI**, **Gemini CLI**, and other AI terminal tools.

## How It Works

Send terminal output to the API, get back the current state:

| State | Meaning | Example |
|-------|---------|---------|
| `idle` | Ready for a new command | `❯ ` empty prompt |
| `waiting_confirmation` | Needs user confirmation or selection | `Do you want to proceed? (y/n)` |
| `processing` | Still working | `⠋ Generating response...` |

Uses [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) zero-shot classification + pattern detection for high-confidence results on known CLI prompts.

## Getting Started

```bash
git clone https://github.com/onchainyaotoshi/terminal-classifier.git
cd terminal-classifier

# Installs everything: Python, deps, model (~1.5GB), systemd service
sudo bash install.sh

# Set your API key
nano .env
```

That's it. The service starts automatically after install.

## Quick Test

```bash
# Confirmation prompt
curl -s -X POST http://localhost:9981/classify \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{"text": "Do you want to proceed? (y/n)"}' | python3 -m json.tool
```

```json
{
  "classification": "waiting_confirmation",
  "confidence": 0.92,
  "scores": {
    "idle": 0.03,
    "waiting_confirmation": 0.92,
    "processing": 0.05
  }
}
```

```bash
# Health check (no auth required)
curl http://localhost:9981/health
```

## Use Case: Multi-Terminal Coordination

Terminal A tells Terminal B to do something. How does A know B is done?

1. Terminal A sends command to Terminal B
2. Poll Terminal B's output → `processing`
3. Terminal B finishes, shows prompt → `idle`
4. Terminal A knows B is done, continues

## Configuration

Edit `.env` (created from `.env.example` on first install):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `9981` | HTTP listen port |
| `API_KEY` | — | Required. Key for `x-api-key` auth |
| `CPU_CORES` | `8` | Max CPU cores for model inference |
| `EXPOSE` | `0` | Set to `1` to bind on `0.0.0.0` (network accessible). Default binds to `127.0.0.1` only |

## API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/classify` | POST | `x-api-key` header | Classify terminal text |
| `/health` | GET | None | Health check |

## Service Management

```bash
systemctl status terminal-classifier
journalctl -u terminal-classifier -f
sudo systemctl restart terminal-classifier
sudo systemctl stop terminal-classifier
```

## Development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 9981 --reload
```

```bash
# Run tests
pytest tests/ -v
```

## License

MIT
