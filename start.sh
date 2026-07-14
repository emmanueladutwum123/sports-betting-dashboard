#!/usr/bin/env bash
# One-command launcher: sets up a venv on first run, then starts the dashboard.
# Streamlit auto-opens your default browser to the dashboard URL.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Setting up virtual environment (first run only)..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo ""
  echo "Created .env — you need a free API key before this will work:"
  echo "  1. Sign up at https://the-odds-api.com (no credit card, 500 free requests/month)"
  echo "  2. Copy your key, paste it into .env replacing 'your_key_here'"
  echo "  3. Run ./start.sh again"
  exit 1
fi

if grep -q "your_key_here" .env; then
  echo "Edit .env and replace 'your_key_here' with your real key from https://the-odds-api.com"
  exit 1
fi

streamlit run app.py
