"""一步生成 .env：按提示粘贴 Key 即可（输入时不显示，Key 只写进本机的 .env，不会提交到 git）。

用法：python scripts/setup_env.py
"""
import getpass
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / ".env.example"
TARGET = ROOT / ".env"

ASK = [
    ("DASHSCOPE_API_KEY", "阿里云百炼 API Key（必填，sk- 开头）", True),
    ("FREESOUND_API_KEY", "Freesound Key（可选，直接回车跳过）", False),
    ("ELEVENLABS_API_KEY", "ElevenLabs Key（可选，直接回车跳过）", False),
]


def main():
    base = TARGET if TARGET.exists() else EXAMPLE
    text = base.read_text("utf-8")
    if TARGET.exists():
        print("已经有 .env，将在原文件上更新（直接回车 = 保留原值）。")
    for name, label, required in ASK:
        current = re.search(rf"^{name}=(.*)$", text, re.M)
        has = bool(current and current.group(1).strip())
        while True:
            val = getpass.getpass(f"{label}{'（已有，回车保留）' if has else ''}：").strip()
            if val or has or not required:
                break
            print("  这一项必须填写。")
        if val:
            text = re.sub(rf"^{name}=.*$", f"{name}={val}", text, flags=re.M)
    text = re.sub(r"^DEMO_MODE=.*$", "DEMO_MODE=false", text, flags=re.M)
    TARGET.write_text(text, "utf-8")
    print(f"已写入 {TARGET}（这个文件在 .gitignore 里，不会被提交）。")
    print("下一步：cd backend && uvicorn app.main:app --host 127.0.0.1 --port 8000")


if __name__ == "__main__":
    main()
