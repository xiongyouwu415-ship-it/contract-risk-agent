"""llm.py —— OpenAI 兼容接口的薄封装（async、带限流与网络重试）。

方案 3：模型名称、地址、密钥、温度配置化管理，便于切换模型与多环境部署。
方案 10.2：全链路 QPS 由令牌桶统一控制，所以每次调用前必须先拿令牌 ——
把限流放在这一层，任何 Agent 都绕不过去。

这里刻意只做「发一次请求 + 记账」，不做任何重试策略上的业务判断：
任务级重试归调度器，只有网络抖动这类瞬时错误才在这一层重试。
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

import config


@dataclass
class ChatResult:
    """一次模型调用的返回：文本 + token 记账（成本告警要靠它）。"""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    """模型调用失败（重试后仍失败），交给调度器按任务级重试处理。"""


class LLMClient:
    def __init__(self, api_key: str, limiter=None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: Optional[float] = None):
        from openai import AsyncOpenAI          # 懒加载：纯离线自测不需要它

        self.model = model or config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.limiter = limiter                  # 令牌桶，可为 None（自测/离线）
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or config.LLM_BASE_URL,
            timeout=config.LLM_TIMEOUT,
        )
        self.calls = 0                          # 累计调用次数，用于观测

    async def chat(self, messages: List[Dict[str, str]]) -> ChatResult:
        """发一次对话请求。messages 是标准的 role/content 列表。"""
        if self.limiter is not None:
            await self.limiter.acquire()

        last_err: Optional[Exception] = None
        for attempt in range(config.LLM_MAX_RETRY + 1):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                self.calls += 1
                text = (resp.choices[0].message.content or "").strip()
                usage = getattr(resp, "usage", None)
                return ChatResult(
                    text=text,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                )
            except Exception as e:                       # 网络抖动 / 限流 / 5xx
                last_err = e
                if attempt >= config.LLM_MAX_RETRY:
                    break
                wait = config.RETRY_BACKOFF * (2 ** attempt)
                logging.warning(f"[llm] 第 {attempt + 1} 次调用失败：{type(e).__name__}: {e}，{wait:.1f}s 后重试")
                await asyncio.sleep(wait)

        raise LLMError(f"模型调用失败（已重试 {config.LLM_MAX_RETRY} 次）：{last_err}")

    async def chat_text(self, prompt: str) -> ChatResult:
        """单条 user 消息的便捷入口。"""
        return await self.chat([{"role": "user", "content": prompt}])


class TokenBucket:
    """令牌桶：限的是「全链路 LLM 调用速率」，不是任务数。

    方案 10.2 第二层防护 —— 达到阈值就暂缓派发，防止短时间批量爆发触发服务限流。
    纯 asyncio 实现，不引入第三方依赖。
    """

    def __init__(self, rate: float = config.QPS_LIMIT, burst: int = config.QPS_BURST):
        self.rate = float(rate)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.updated = 0.0                    # 单调时钟，配合 _refill
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self.updated == 0.0:
                self.updated = now
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            while self.tokens < 1.0:
                need = (1.0 - self.tokens) / self.rate
                await asyncio.sleep(need)
                now = loop.time()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
            self.tokens -= 1.0
