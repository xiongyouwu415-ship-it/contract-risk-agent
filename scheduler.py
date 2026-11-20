"""scheduler.py —— 自研轻量任务调度器（方案 4）。

四项核心能力，逐条对应：
  4.2-1 条件路由      RouteTable：状态 → 下一站，表驱动，业务代码里不写 if 链
  4.2-2 并行分支执行  fan_out()：条款列表展开为并行分支，信号量限流
  4.2-3 失败回溯与幂等重试  retry_failed()：只重跑失败分支，已完成的不重算
  4.2-4 SQLite 状态快照    由各 Agent 在每一轮写库，调度器负责「取用干净快照」

4.3 三层控制：并发（信号量）/ QPS（令牌桶，挂在 LLM 客户端上）/ 单任务 token 熔断（在各 Agent 内）。

判定 Agent 与复核 Agent 都是「一套模板 + N 份快照」，所以这里并行的永远是
(task_id, 输入) 的组合，而不是 N 个 Agent 实例。
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

import config
import db


# ==================================================================== 4.2-1 条件路由
class RouteTable:
    """状态 → 下一站的查表器。

    key 是 (节点名, 状态)，value 是下一站节点名。
    查不到就走 default —— 让「未定义状态」落进兜底路径，而不是抛异常炸掉整条流水线。
    """

    def __init__(self, table: Dict[tuple, str], default: str = "end"):
        self.table = dict(table)
        self.default = default

    def next(self, node: str, status: str) -> str:
        return self.table.get((node, status), self.default)


# 证据校验后的三路分流（方案 7.4 表 5）：这是整条流水线唯一的条件分支点
EVIDENCE_ROUTE = RouteTable({
    ("evidence", db.EVIDENCE_OK): "review",        # 定位成功 → 采信路径进复核
    ("evidence", db.EVIDENCE_FAIL): "review",      # 定位失败 → 降级路径进复核
    ("evidence", db.EVIDENCE_NONE): "regression",  # 缺少有效证据 → 判无效，入回归池
    ("judge", db.FAILED): "regression",            # 判定失败 → 也沉淀，不做无声丢弃
}, default="end")


# ==================================================================== 并发与重试
@dataclass
class BranchResult:
    """一个并行分支的执行结果。"""
    key: str
    ok: bool
    value: Any = None
    attempts: int = 1
    error: Optional[str] = None


class Scheduler:
    def __init__(self, max_concurrent: Optional[int] = None,
                 max_retry: Optional[int] = None,
                 backoff: Optional[float] = None):
        self.max_concurrent = max_concurrent or config.MAX_CONCURRENT_TASKS
        self.max_retry = config.TASK_MAX_RETRY if max_retry is None else max_retry
        self.backoff = config.RETRY_BACKOFF if backoff is None else backoff
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.stats = {"branches": 0, "retried": 0, "failed": 0}

    # ------------------------------------------------------------ 4.2-2 Fan-out
    async def fan_out(self, items: List[Any],
                      worker: Callable[[Any], Awaitable[Any]],
                      name_of: Callable[[Any], str] = str) -> List[BranchResult]:
        """把列表展开成并行分支，受信号量约束；单分支异常不影响其他分支。

        return_exceptions=True 的语义就在这里：一个条款判定崩了，
        不该让整份合同的其他条款一起白跑。
        """
        async def _one(item: Any) -> BranchResult:
            key = name_of(item)
            async with self.semaphore:
                try:
                    value = await worker(item)
                    return BranchResult(key=key, ok=True, value=value)
                except Exception as e:                       # noqa: BLE001 分支级兜底
                    logging.error(f"[sched] 分支 {key} 执行异常：{type(e).__name__}: {e}")
                    return BranchResult(key=key, ok=False, error=f"{type(e).__name__}: {e}")

        results = await asyncio.gather(*[_one(it) for it in items])
        self.stats["branches"] += len(results)
        self.stats["failed"] += sum(1 for r in results if not r.ok)
        return list(results)

    # ------------------------------------------------------------ 4.2-3 幂等重试
    async def retry_failed(self, results: List[BranchResult], items: List[Any],
                           worker: Callable[[Any], Awaitable[Any]],
                           name_of: Callable[[Any], str] = str,
                           is_done: Optional[Callable[[Any], bool]] = None
                           ) -> List[BranchResult]:
        """只重跑失败分支。已完成的分支结果原样保留，不重复调模型。

        is_done(item) 为真表示这条分支其实已经产出结论（比如快照里是 FINISHED，
        只是上次进程被杀在写库前后），此时不再重跑，直接算成功 —— 这就是
        「重试不产生重复副作用」的落点。
        """
        by_key = {name_of(it): it for it in items}
        for attempt in range(1, self.max_retry + 1):
            pending = [r for r in results if not r.ok]
            if not pending:
                break
            wait = self.backoff * attempt
            logging.info(f"[sched] 第 {attempt} 轮重试：{len(pending)} 个失败分支，"
                         f"{wait:.1f}s 后重跑 {[r.key for r in pending]}")
            await asyncio.sleep(wait)

            for r in pending:
                item = by_key.get(r.key)
                if item is None:
                    continue
                if is_done is not None and is_done(item):
                    r.ok, r.error = True, None
                    logging.info(f"[sched] {r.key} 快照已产出结论，跳过重跑")
                    continue

                async with self.semaphore:
                    try:
                        r.value = await worker(item)
                        r.ok, r.error = True, None
                        r.attempts += 1
                        self.stats["retried"] += 1
                    except Exception as e:                    # noqa: BLE001
                        r.error = f"{type(e).__name__}: {e}"
                        r.attempts += 1
                        logging.warning(f"[sched] {r.key} 重试仍失败：{r.error}")
        return results


# ==================================================================== 观测
def summary(results: List[BranchResult]) -> Dict[str, int]:
    return {
        "total": len(results),
        "ok": sum(1 for r in results if r.ok),
        "failed": sum(1 for r in results if not r.ok),
        "retried": sum(1 for r in results if r.attempts > 1),
    }
