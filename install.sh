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
ExecStart=${VENV_DIR}/bin/python ${PROJECT_DIR}/run.py
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
