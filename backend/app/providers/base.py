"""Provider 公共工具：重试、错误类型、用量统计。"""
from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from typing import Awaitable, Callable, TypeVar

log = logging.getLogger("providers")
T = TypeVar("T")


class ProviderError(RuntimeError):
    """外部服务调用失败（已重试）。上层负责降级，绝不能把它显示给老人。"""

    def __init__(self, msg: str, retryable: bool = True, status: int | None = None):
        super().__init__(msg)
        self.retryable = retryable
        self.status = status


async def with_retry(fn: Callable[[], Awaitable[T]], attempts: int = 3, base_delay: float = 1.0,
                     what: str = "调用") -> T:
    last: Exception | None = None
    for i in range(attempts):
        try:
            return await fn()
        except ProviderError as e:
            last = e
            if not e.retryable:
                raise  # 不可重试的错误原样抛出（保留状态码，供上层决定如何降级）
        except Exception as e:  # 网络错误等
            last = e
        if i < attempts - 1:
            delay = base_delay * (2 ** i) + random.random() * 0.3
            log.warning("%s 失败（第 %d 次），%.1f 秒后重试：%s", what, i + 1, delay, last)
            await asyncio.sleep(delay)
    raise ProviderError(f"{what} 最终失败：{last}", retryable=False,
                        status=last.status if isinstance(last, ProviderError) else None) from last


class Usage:
    """粗略统计调用量，用于费用估算（写入事件日志和报告）。"""

    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)

    def add(self, key: str, amount: float = 1.0) -> None:
        self.counters[key] += amount

    def snapshot(self) -> dict[str, float]:
        return dict(self.counters)


usage = Usage()
