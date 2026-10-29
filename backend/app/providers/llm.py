"""大语言模型：百炼 OpenAI 兼容接口。所有结构化输出都经 JSON Schema 约束 + Pydantic 校验。"""
from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any, Callable, Optional, Protocol, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .base import ProviderError, usage, with_retry

log = logging.getLogger("llm")
M = TypeVar("M", bound=BaseModel)
Message = dict[str, str]


class LLM(Protocol):
    async def chat(self, messages: list[Message], model: Optional[str] = None,
                   temperature: float = 0.7, max_tokens: int = 300) -> str: ...

    async def structured(self, messages: list[Message], schema: Type[M], model: Optional[str] = None,
                         temperature: float = 0.2, retries: int = 3) -> M: ...


def strict_schema(model: Type[BaseModel]) -> dict:
    """把 Pydantic 生成的 Schema 改成 strict 模式可接受的形式：
    所有对象 additionalProperties=false、所有属性都 required（可选字段用 null 表示）。"""
    schema = copy.deepcopy(model.model_json_schema())

    def fix(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            # 元组（prefixItems）在部分实现里不支持，改为定长数组
            if "prefixItems" in node:
                items = node.pop("prefixItems")
                node["items"] = items[0] if items else {}
                node["minItems"] = node["maxItems"] = len(items)
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    return schema


def extract_json(text: str) -> Any:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    if not text.startswith(("{", "[")):
        start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
        if start >= 0:
            text = text[start:]
    return json.loads(text)


class DashScopeLLM:
    def __init__(self, api_key: str, base_url: str, default_model: str, timeout: float = 40.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self._schema_unsupported: set[str] = set()
        self._thinking_param_rejected: set[str] = set()

    async def _post(self, body: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {self.api_key}"})
        if r.status_code >= 400:
            raise ProviderError(f"LLM HTTP {r.status_code}: {r.text[:300]}",
                                retryable=r.status_code in (429, 500, 502, 503, 504), status=r.status_code)
        data = r.json()
        u = data.get("usage") or {}
        usage.add(f"llm:{body['model']}:in", u.get("prompt_tokens", 0))
        usage.add(f"llm:{body['model']}:out", u.get("completion_tokens", 0))
        return data

    async def _complete(self, body: dict) -> str:
        model = body["model"]
        if model not in self._thinking_param_rejected:
            body = {**body, "enable_thinking": False}  # 陪聊和抽取都不需要“思考”，降低延迟

        async def call() -> str:
            try:
                data = await self._post(body)
            except ProviderError as e:
                if e.status == 400 and "enable_thinking" in body and "thinking" in str(e):
                    self._thinking_param_rejected.add(model)
                    b2 = {k: v for k, v in body.items() if k != "enable_thinking"}
                    data = await self._post(b2)
                else:
                    raise
            return data["choices"][0]["message"].get("content") or ""

        return await with_retry(call, attempts=3, what=f"LLM {model}")

    async def chat(self, messages: list[Message], model: Optional[str] = None,
                   temperature: float = 0.7, max_tokens: int = 300) -> str:
        body = {"model": model or self.default_model, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        return (await self._complete(body)).strip()

    async def structured(self, messages: list[Message], schema: Type[M], model: Optional[str] = None,
                         temperature: float = 0.2, retries: int = 3) -> M:
        model = model or self.default_model
        js = strict_schema(schema)
        msgs = list(messages)
        last_err = ""
        for attempt in range(retries):
            use_schema = model not in self._schema_unsupported
            body: dict = {"model": model, "messages": msgs, "temperature": temperature}
            if use_schema:
                body["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": schema.__name__, "schema": js, "strict": True}}
            else:
                # 退回 JSON Object 模式：提示词里必须出现“JSON”，并把 Schema 附上
                body["response_format"] = {"type": "json_object"}
                body["messages"] = msgs + [{"role": "system", "content":
                                            "只输出一个符合下面 JSON Schema 的 JSON 对象，不要输出其他文字：\n"
                                            + json.dumps(js, ensure_ascii=False)}]
            try:
                text = await self._complete(body)
            except ProviderError as e:
                if use_schema and e.status == 400:
                    log.info("模型 %s 不支持 json_schema，改用 json_object：%s", model, e)
                    self._schema_unsupported.add(model)
                    continue
                raise
            try:
                return schema.model_validate(extract_json(text))
            except (ValidationError, ValueError) as e:
                last_err = str(e)[:800]
                log.warning("结构化输出校验失败（第 %d 次）：%s", attempt + 1, last_err)
                msgs = msgs + [{"role": "assistant", "content": text[:2000]},
                               {"role": "user", "content": f"上面的输出不符合要求：{last_err}\n请重新输出，只输出合规的 JSON。"}]
        raise ProviderError(f"结构化输出多次校验失败：{last_err}", retryable=False)


class MockLLM:
    """本地演示/测试用：由一个处理函数决定输出。"""

    def __init__(self, handler: Callable[[list[Message], Optional[Type[BaseModel]]], Any]):
        self.handler = handler

    async def chat(self, messages: list[Message], model: Optional[str] = None,
                   temperature: float = 0.7, max_tokens: int = 300) -> str:
        return str(self.handler(messages, None))

    async def structured(self, messages: list[Message], schema: Type[M], model: Optional[str] = None,
                         temperature: float = 0.2, retries: int = 3) -> M:
        out = self.handler(messages, schema)
        if isinstance(out, schema):
            return out
        return schema.model_validate(out)
