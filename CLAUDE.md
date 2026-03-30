# Terminal Classifier

HTTP API service that classifies AI CLI terminal output using facebook/bart-large-mnli.

## Quick Start

```bash
# Install and start
sudo bash install.sh

# Or run manually
source venv/bin/activate
uvicorn app.main:app --port 9981
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
- `PORT` — listen port (default 9981)
- `API_KEY` — required for auth
- `CPU_CORES` — max CPU cores for inference
