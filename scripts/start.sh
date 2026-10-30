#!/usr/bin/env bash
# 启动「听见从前」。启动后在 Windows 的 Chrome 里打开 http://127.0.0.1:8000/setup.html
set -e
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
echo "启动中… 在 Windows 的 Chrome / Edge 里打开： http://127.0.0.1:8000/setup.html"
echo "（关闭：在这个窗口按 Ctrl+C）"
cd backend
exec uvicorn app.main:app --host 127.0.0.1 --port 8000
