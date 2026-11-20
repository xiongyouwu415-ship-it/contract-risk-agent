"""config.py —— 全系统的可调参数，集中一处，运行期可改。

对应开发方案「3 技术选型」与「10 可靠性与成本防护」：
模型名称、地址、密钥、温度全部配置化；三层防护的参数（并发上限、QPS、token 阈值）
与证据校验的相似度阈值同样走配置，不散落在业务代码里。
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 路径
DB_PATH = os.environ.get("CONTRACT_AGENT_DB", os.path.join(BASE_DIR, "data", "contract_agent.db"))
SAMPLE_CONTRACT_DIR = os.path.join(BASE_DIR, "rule")        # 36 份样例合同
GOLDEN_DIR = os.path.join(BASE_DIR, "final_gt")             # 人工标注金标
LAW_DB_PATH = os.path.join(BASE_DIR, "data", "laws.json")   # 法条检索工具的数据文件
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")      # 前端上传的合同正文
WEB_DIR = os.path.join(BASE_DIR, "web")                     # 前端静态资源


# ---------------------------------------------------------------- LLM 接入
# OpenAI 兼容接口，换模型只改这里，Agent 代码不动
LLM_BASE_URL = os.environ.get("CONTRACT_AGENT_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("CONTRACT_AGENT_MODEL", "deepseek-chat")
LLM_API_KEY_ENV = "DEEPSEEK_API_KEY"
LLM_TEMPERATURE = 0.0          # 判定类任务要可复现，温度给 0
LLM_TIMEOUT = 60.0
LLM_MAX_RETRY = 2              # 单次调用层面的网络重试（与任务级重试区分开）


# ---------------------------------------------------------------- 解析 Agent
# 正则只找「疑似条款起点」，不做语义判断（方案 5.2）
ANCHOR_PATTERN = (
    r"(?m)^[ \t\u3000]*("
    r"[一二三四五六七八九十]+[、．.]"          # 一、 二、
    r"|第[一二三四五六七八九十百零〇\d]+条"      # 第X条
    r"|（[一二三四五六七八九十]+）"             # （一）
    r"|\([一二三四五六七八九十]+\)"             # (一)
    r"|[0-9]{1,2}[、．.]\s*[^\d\s]"            # 1、 2.
    r")"
)
# 小到不值得送 LLM 判边界的候选片段（字符数）
MIN_CANDIDATE_CHARS = 15
# 组装后的条款少于这个长度，视为噪音丢弃
MIN_TERM_CHARS = 20

# 复杂度打标：命中任一特征即判「复杂」，放开轮次上限（方案 6.4 / 10.1）
COMPLEX_MARKERS = ("详见", "附件", "另行", "参照", "前述", "第", "及其", "或", "以及", "如", "若")
COMPLEX_MIN_CHARS = 180


# ---------------------------------------------------------------- 判定 / 复核 Agent
MAX_ROUNDS_SIMPLE = 4          # 简单条款 3~4 轮
MAX_ROUNDS_COMPLEX = 12        # 复杂条款 10~15 轮，取中值控成本
MAX_TOOL_CALLS_PER_ROUND = 1   # 一轮只允许一个工具调用，防止模型一次甩一堆


# ---------------------------------------------------------------- 证据校验
SIMILARITY_THRESHOLD = 0.85    # 方案 7.3：达到阈值即定位成功
MATCH_STEP_COARSE = 4          # 粗扫步长，命中候选后再细扫 ±8
MATCH_REFINE_RADIUS = 8
MATCH_MIN_ANCHOR = 4           # 锚点最短字符数，低于此值不再做锚点定位


# ---------------------------------------------------------------- 金标评测（metrics.py，仅离线评分，不影响运行时）
GOLDEN_IOU_STRONG = 0.8        # IoU ≥ 此值：定位成功（自动采信档）
GOLDEN_IOU_MIN = 0.3           # IoU 下限：低于此值落入漏检档（沿用原代码 ≥0.3 命中口径）
TOO_BROAD_RECALL = 0.9         # 「证据范围过宽」初始建议值：GT 覆盖率 ≥ 此值
TOO_BROAD_PRECISION = 0.5      # 「证据范围过宽」：精确率 ≤ 此值视为引用过宽（初始值，待业务校准）


# ---------------------------------------------------------------- 调度器三层防护
MAX_CONCURRENT_TASKS = 5       # 第一层：同时运行的判定任务数
QPS_LIMIT = 8.0                # 第二层：全链路 LLM 调用速率（次/秒）
QPS_BURST = 4                  # 令牌桶容量
TOKEN_LIMIT_PER_TASK = 12000   # 第三层：单任务 token 超限直接熔断
TASK_MAX_RETRY = 2             # 任务级重试次数（幂等，只重跑失败分支）
RETRY_BACKOFF = 1.5            # 秒，指数退避基数


# ---------------------------------------------------------------- 服务层
# 是否用离线模拟模型：不花钱、不需要密钥，用来跑通界面与链路，结论不具参考价值
MOCK_LLM = os.environ.get("CONTRACT_AGENT_MOCK", "0").lower() in ("1", "true", "yes", "on")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024     # 前端上传上限
MAX_CONTRACT_CHARS = 200_000           # 单份合同正文上限，防超长文本打爆上下文


def api_key() -> str:
    """从环境变量取密钥；没配就抛，避免报一个看不懂的 401。"""
    key = os.getenv(LLM_API_KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"没读到环境变量 {LLM_API_KEY_ENV}。请先设置模型密钥再运行。"
        )
    return key
