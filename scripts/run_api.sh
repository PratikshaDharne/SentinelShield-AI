#!/usr/bin/env bash
cd "$(dirname "$0")/.." && exec uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT:-8130}"
