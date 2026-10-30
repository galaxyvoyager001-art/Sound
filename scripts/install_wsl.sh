#!/usr/bin/env bash
# 在 WSL（Ubuntu）里一键安装「听见从前」需要的一切。
# 用法：在项目文件夹（Sound）里运行  bash scripts/install_wsl.sh
set -e
cd "$(dirname "$0")/.."

echo "== 1/5 安装系统软件（ffmpeg、Python 工具）=="
sudo apt-get update -y
sudo apt-get install -y ffmpeg python3 python3-pip python3-venv curl git

echo "== 2/5 安装 Node.js（用于构建网页界面）=="
if ! command -v node >/dev/null || [ "$(node -v | cut -c2- | cut -d. -f1)" -lt 18 ]; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt-get install -y nodejs
fi
node -v

echo "== 3/5 安装 Python 依赖（放在项目里的 .venv，不影响系统）=="
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt

echo "== 4/5 下载深度估计模型（约 100 MB）=="
python scripts/download_models.py

echo "== 5/5 构建网页界面 =="
(cd frontend && npm install && npm run build)

if [ ! -f .env ] || ! grep -q "^DASHSCOPE_API_KEY=sk-" .env; then
  echo
  echo "== 还差一步：填写 API Key =="
  python scripts/setup_env.py
fi

echo
echo "安装完成。以后每次启动只需要运行：  bash scripts/start.sh"
