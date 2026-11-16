#!/usr/bin/env bash
# One-command local launcher; the journal works without live odds configured.
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env. Set DASHBOARD_PASSWORD before opening the private journal."
fi
streamlit run app.py --server.address 127.0.0.1
