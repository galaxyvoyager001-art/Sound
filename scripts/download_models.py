"""下载本地深度估计模型 Depth Anything V2 Small（ONNX，约 99 MB，Apache-2.0）。"""
import sys
import urllib.request

from _common import ROOT

URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
dst = ROOT / "models" / "depth_anything_v2_small.onnx"
dst.parent.mkdir(exist_ok=True)
if dst.exists() and "--force" not in sys.argv:
    print("已存在：", dst)
else:
    print("下载", URL)
    urllib.request.urlretrieve(URL, dst)
    print("完成：", dst, dst.stat().st_size // 1_000_000, "MB")
