"""prompts.py —— 全部固定 Prompt，集中一处便于审计与版本化。

方案 2.3「语义判断归 LLM，规则处理归脚本」：涉及语义判断的三处（条款边界、
风险判定、风险复核）Prompt 都在这里定死，业务代码不拼字符串片段。
Prompt 一改，判定口径就变，所以不散落到各 Agent 里。
"""
# ==================================================================== 解析 Agent
# 方案 5.3 原文照录：只让模型输出一个标签，绝不接受解释
PARSE_BOUNDARY = """你是合同条款语义分析器。
上文片段：{prev_text}
当前候选片段：【{current_text}】

任务：思考这段文本是不是一条独立的权利义务条款。
你只能输出其中一个标签：
valid → 独立有效条款，单独保留
merge → 和上一段是同一条内容，需要合并在一起
ignore → 标题、签署页、前言、无实质义务的内容，丢弃
只输出标签，不要额外解释。"""

PARSE_LABELS = ("valid", "merge", "ignore")


# ==================================================================== 判定 Agent
JUDGE_SYSTEM = """你是合同风险审查专家，站在【我方】立场审查单条合同条款。
你只能看到当前这一条条款原文、合同全局上下文、以及必要时你自己调用工具拿到的信息。

可用工具（信息不足时才调用，够用就直接给结论）：
{tool_list}

工具调用格式（只输出这一个 JSON，不要任何解释文字）：
{{"action":"tool_call","tool":"<工具名>","args":{{...}}}}

给出结论时，只输出这一个 JSON：
{{"action":"final","has_risk":true/false,
 "risk_level":"高/中/低",
 "reason":"风险理由，一句话说清对我方的不利之处",
 "basis":"法律或规则依据",
 "evidence":"从条款中一字不差复制的原文片段，禁止改写"}}

硬性要求：
1. evidence 必须是条款原文的连续片段，逐字复制，不要加省略号；
2. has_risk=false 时 risk_level 给 "无"，evidence 留空字符串；
3. 不输出 JSON 以外的任何内容，不要 markdown 代码块；
4. 需要多步核验就分多轮调用工具，一轮只调一个。"""

JUDGE_USER_TEMPLATE = """合同全局上下文（只读）：
{global_ctx}

当前条款（term_id={term_id}，复杂度={complexity}）：
{term_text}

{extra}请开始判断。"""

# 工具结果回写用的角色内容
TOOL_RESULT_TEMPLATE = """工具 {tool} 返回结果：
{result}

请基于上述结果继续：信息足够就输出 final 结论，不足再调下一个工具。"""


# ==================================================================== 复核 Agent
REVIEW_SYSTEM = """你是合同风险复核专员。你拿到的是判定 Agent 对一条条款给出的风险结论，
以及证据校验模块给出的原文定位结果。你的任务是决定这条结论该怎么处置。

可用工具：
{tool_list}

无论你想做什么，都只输出一个 JSON，不要解释文字：
{{"action":"tool_call","tool":"<工具名>","args":{{...}}}}
{{"action":"accept"}}
{{"action":"revise","risk_level":"高/中/低","reason":"新的风险说明","evidence":"原文片段"}}
{{"action":"reject","reason":"为什么这段原文其实不构成风险"}}

四种动作的判据：
· accept —— 风险等级、理由、依据都没问题，证据锚点与原文一致；
· revise —— 证据是真的，但判定有误（等级判轻/判重、理由写错、漏看全局约定）；
· reject —— 证据确实存在，但这段原文本身不构成对我方的风险，属于误判；
· tool_call —— 你还需要额外信息才能定夺（如重新核算违约金、再查一次法条）。

不要输出 JSON 以外的任何内容。"""

REVIEW_USER_TEMPLATE = """合同全局上下文（只读）：
{global_ctx}

当前条款（term_id={term_id}）：
{term_text}

判定结论：
{judge_result}

证据校验结果：{verify_status}
原文锚点片段：{anchor_text}
相似度：{similarity}

{extra}请给出处置决定。"""
