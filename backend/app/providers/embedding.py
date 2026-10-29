"""文本向量：百炼 OpenAI 兼容 /embeddings；没有 Key 时用本地字符 n-gram 哈希向量兜底。"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import httpx
import numpy as np

from .base import ProviderError, usage, with_retry

log = logging.getLogger("embedding")


class Embedder:
    name = "base"
    dim = 256

    async def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(Embedder):
    """字符 1~3 gram 哈希到 512 维再归一化：对同义词无能为力，但对“馄饨担”~“馄饨担子”这类有效。"""
    name = "hash"
    dim = 512

    async def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        t = f"^{text.strip().lower()}$"
        for n in (1, 2, 3):
            for i in range(len(t) - n + 1):
                h = int(hashlib.md5(t[i:i + n].encode()).hexdigest()[:8], 16)
                v[h % self.dim] += 1.0 / n
        nrm = np.linalg.norm(v)
        return v / nrm if nrm else v


class DashScopeEmbedder(Embedder):
    name = "dashscope"

    def __init__(self, api_key: str, base_url: str, model: str, cache_path: Path, dim: int = 512):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.cache_path = cache_path
        self._cache: dict[str, list[float]] = {}
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text())
            except Exception:
                self._cache = {}

    def _key(self, t: str) -> str:
        return hashlib.sha1(f"{self.model}|{self.dim}|{t}".encode()).hexdigest()[:16]

    async def embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self._cache]
        for i in range(0, len(missing), 10):  # 官方限制单次最多 20 条，这里保守用 10
            batch = missing[i:i + 10]

            async def call(batch=batch):
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.post(f"{self.base_url}/embeddings",
                                     json={"model": self.model, "input": batch, "dimensions": self.dim,
                                           "encoding_format": "float"},
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "X-DashScope-DataInspection": "disable"})
                if r.status_code >= 400:
                    raise ProviderError(f"Embedding HTTP {r.status_code}: {r.text[:200]}",
                                        retryable=r.status_code >= 429, status=r.status_code)
                return r.json()

            data = await with_retry(call, attempts=3, what="文本向量")
            usage.add(f"embedding:{self.model}:tokens", (data.get("usage") or {}).get("total_tokens", 0))
            for item in data["data"]:
                self._cache[self._key(batch[item["index"]])] = item["embedding"]
        if missing:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self._cache))
            except Exception:
                pass
        arr = np.array([self._cache[self._key(t)] for t in texts], np.float32)
        nrm = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(nrm, 1e-8)
