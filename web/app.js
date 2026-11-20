/* ==========================================================================
   合同风险审查多 Agent 系统 —— 前端逻辑（原生 JS，无框架、无构建）
   状态只有一份 S；所有视图都是 S 的纯函数渲染，避免多处 DOM 各自 update 失同步。
   ========================================================================== */
'use strict';

const $ = (sel) => document.querySelector(sel);

const RISK_RANK = { 高: 0, 中: 1, 低: 2, 无: 3 };
const LEVEL_CLASS = { 高: 'lv-high', 中: 'lv-mid', 低: 'lv-low', 无: 'lv-none' };
const LEVEL_TEXT = { 高: '高危', 中: '中危', 低: '低危', 无: '无风险' };
const ACTION_CLASS = {
  采信: 'tag-mute', 修正: 'tag-warn', 驳回: 'tag-mute',
  判无效: 'tag-bad', 无风险: 'tag-ok', 复核失败: 'tag-bad', 判定失败: 'tag-bad',
};
const EVIDENCE_CLASS = {
  定位成功: 'tag-ok', 定位失败: 'tag-warn', 缺少有效证据: 'tag-bad',
};

const S = {
  health: null,
  contracts: [],
  current: null,
  detail: null,
  tab: 'risks',
  riskFilter: 'all',
  termFilter: 'all',
  query: '',
  fresh: false,
  task: null,
  timer: null,
  sourceOpen: false,
  hlIndex: {},
  regression: null,
};

/* ---------------------------------------------------------------- 工具 */
const esc = (s) => String(s == null ? '' : s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

const levelClass = (lv) => LEVEL_CLASS[lv] || 'lv-none';
const levelText = (lv) => LEVEL_TEXT[lv] || (lv ? lv + '危' : '—');

const fmt = (n) => {
  if (n == null) return '—';
  if (n >= 10000) return (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
  return String(n);
};
const pct = (v) => (v == null ? '—' : (v * 100).toFixed(0) + '%');
const pct1 = (v) => (v == null ? '—' : (v * 100).toFixed(1) + '%');

const fmtTime = (iso) => {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (x) => String(x).padStart(2, '0');
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
};

const period = (s) => (s == null ? '—' : (Math.round(s * 10) / 10) + 's');

function toast(msg, isErr = false) {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'toast show' + (isErr ? ' err' : '');
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.className = 'toast'; }, isErr ? 4200 : 2400);
}

async function api(path, opts = {}) {
  const init = { ...opts };
  if (init.body && !(init.body instanceof FormData)) {
    init.headers = { 'Content-Type': 'application/json', ...(init.headers || {}) };
    if (typeof init.body !== 'string') init.body = JSON.stringify(init.body);
  }
  const res = await fetch(path, init);
  let data = null;
  try { data = await res.json(); } catch (_) { data = null; }
  if (!res.ok) {
    const msg = (data && (data.detail || data.message)) || `请求失败（${res.status}）`;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return data;
}

/* ---------------------------------------------------------------- 启动 */
async function boot() {
  try {
    S.health = await api('/api/health');
    renderMode();
    renderSideFoot();
  } catch (e) {
    toast('后端未就绪：' + e.message, true);
    $('#modeChip').textContent = '后端离线';
    return;
  }
  await loadContracts();

  const cid = new URLSearchParams(location.search).get('c')
    || (S.contracts[0] && S.contracts[0].contract_id);
  if (cid) await selectContract(cid);
  else renderMain();
}

function renderMode() {
  const h = S.health;
  const chip = $('#modeChip');
  if (!h) return;
  if (h.mock) {
    chip.className = 'chip chip-mock';
    chip.textContent = '离线模拟';
    chip.title = '未调用真实模型，结论仅用于演示流程，不具备参考价值';
  } else {
    chip.className = 'chip chip-real';
    chip.textContent = h.model;
    chip.title = `真实模型：${h.model}`;
  }
}

function renderSideFoot() {
  const h = S.health;
  if (!h) return;
  $('#sideFoot').innerHTML = `
    并发 ${h.limits.concurrency} · QPS ${h.limits.qps}<br>
    单任务熔断 ${fmt(h.limits.token_per_task)} tokens
    ${h.mock ? '<br><span style="color:#a35f04">离线模拟模式，不产生费用</span>' : ''}`;
}

/* ---------------------------------------------------------------- 合同库 */
async function loadContracts() {
  const data = await api('/api/contracts');
  S.contracts = data.items || [];
  renderList();
}

function renderList() {
  const q = S.query.trim().toLowerCase();
  const items = S.contracts.filter((c) => !q || c.file_name.toLowerCase().includes(q));
  $('#contractCount').textContent = items.length;

  const box = $('#contractList');
  if (!items.length) {
    box.innerHTML = `<div class="empty" style="padding:24px 12px;border:none;background:transparent">
      <div class="empty-desc">${q ? '没有匹配的合同' : '合同库为空，先上传一份'}</div></div>`;
    return;
  }

  box.innerHTML = items.map((c) => {
    const bits = [];
    if (c.terms_total) bits.push(`${c.terms_total} 条款`);
    if (c.risk_total) bits.push(`<span class="ci-risk">${c.risk_total} 风险</span>`);
    if (!c.parsed) bits.push('未解析');
    return `
      <button class="contract-item ${c.contract_id === S.current ? 'active' : ''}"
              data-cid="${esc(c.contract_id)}" type="button" title="${esc(c.file_name)}">
        <div class="ci-top">
          ${c.running ? '<i class="pulse-dot"></i>' : ''}
          <span class="ci-name">${esc(c.file_name)}</span>
          ${c.source === 'upload' ? '<span class="tag">上传</span>' : ''}
        </div>
        <div class="ci-meta">
          ${bits.map((b, i) => (i ? '<i class="ci-dot"></i>' : '') + b).join('')}
          ${c.reviewed_at ? `<i class="ci-dot"></i>${fmtTime(c.reviewed_at)}` : ''}
        </div>
      </button>`;
  }).join('');
}

async function selectContract(cid) {
  if (!cid) return;
  S.current = cid;
  S.task = null;
  S.regression = null;
  S.sourceOpen = false;
  $('#sourcePanel').classList.remove('open');
  renderList();

  try {
    S.detail = await api('/api/contracts/' + encodeURIComponent(cid));
  } catch (e) {
    toast('读取合同失败：' + e.message, true);
    S.detail = null;
    renderMain();
    return;
  }

  S.tab = hasRisks() ? 'risks' : (S.detail.parsed ? 'terms' : 'risks');
  renderMain();

  if (S.detail.running && S.detail.running.task_id) watch({ task_id: S.detail.running.task_id });
  const url = new URL(location.href);
  url.searchParams.set('c', cid);
  history.replaceState(null, '', url);
}

async function reloadDetail() {
  if (!S.current) return;
  S.detail = await api('/api/contracts/' + encodeURIComponent(S.current));
  renderMain();
  renderList();
}

/* ---------------------------------------------------------------- 数据派生 */
const finals = () => (S.detail && S.detail.result && S.detail.result.final) || [];
const hasRisks = () => finals().some((f) => f.has_risk);

function statsOf() {
  const list = finals();
  const risks = list.filter((f) => f.has_risk);
  const byLevel = { 高: 0, 中: 0, 低: 0 };
  risks.forEach((f) => { if (byLevel[f.risk_level] != null) byLevel[f.risk_level] += 1; });
  const report = (S.detail && S.detail.report) || {};
  const ev = report.evidence || {};
  const cost = report.cost || {};
  return {
    terms: (S.detail && S.detail.terms.length) || 0,
    risks: risks.length,
    high: byLevel.高, mid: byLevel.中, low: byLevel.低,
    needsHuman: list.filter((f) => f.needs_human).length,
    located: ev['定位成功率'],
    duration: (S.detail && S.detail.result && S.detail.result.duration_s),
    tokens: cost.judge_tokens,
    calls: cost.llm_calls,
    recall: (report.golden || {}).recall,
  };
}

/* ---------------------------------------------------------------- 主视图 */
function renderMain() {
  const main = $('#main');
  main.innerHTML = `<div class="main-inner">${S.detail ? contractView() : welcomeView()}</div>`;
  if (S.sourceOpen) renderSource();
}

function welcomeView() {
  return `
    <div class="empty" style="padding:64px 24px">
      <div class="empty-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"
             stroke-linecap="round" stroke-linejoin="round">
          <path d="M6 3h8l4 4v14H6z"></path><path d="M14 3v4h4"></path><path d="M9 13h6"></path>
        </svg>
      </div>
      <div class="empty-title">选择左侧的一份合同开始审查</div>
      <p class="empty-desc">
        系统会先把合同切成条款，再对每条条款并行判定风险，随后用规则程序校验证据是否真在原文里，
        最后由复核 Agent 给出采信 / 修正 / 驳回的处置决定。
      </p>
      <div class="empty-actions">
        <button class="btn btn-primary" data-action="paste" type="button">粘贴一份合同</button>
        <button class="btn btn-ghost" data-action="upload" type="button">上传文件</button>
      </div>
    </div>`;
}

function contractView() {
  const d = S.detail;
  const st = statsOf();
  return `
    ${headerView(d, st)}
    ${runBannerView()}
    ${d.parsed ? statGridView(st) : ''}
    ${d.parsed ? tabsView(st) : unparsedView()}
    <div id="tabBody">${d.parsed ? tabBodyView(st) : ''}</div>
  `;
}

function headerView(d, st) {
  const meta = d.meta || {};
  const chips = Object.entries(meta).map(([k, v]) =>
    `<span class="meta-chip"><span>${esc(k)}</span><b>${esc(v)}</b></span>`).join('');

  const last = d.last_run;
  const sub = [
    `${d.raw_chars} 字`,
    d.terms.length ? `${d.terms.length} 条条款` : '',
    last ? `最近审查 ${fmtTime(last.created_at)}` : '尚未审查',
    last && last.status === 'failed' ? '<span style="color:#c02626">上次任务失败</span>' : '',
  ].filter(Boolean).join(' · ');

  const running = S.task && S.task.status === 'running';
  return `
    <section class="contract-head">
      <div class="ch-top">
        <div style="min-width:0">
          <h2 class="ch-title">${esc(d.file_name)}</h2>
          <div class="ch-sub"><span>${sub}</span>
            ${d.source === 'upload' ? '<span class="tag">上传合同</span>' : '<span class="tag">样例合同</span>'}
          </div>
        </div>
        <div class="ch-actions">
          <label class="switch" title="丢弃已有快照，全部重新判定">
            <input type="checkbox" id="freshToggle" ${S.fresh ? 'checked' : ''}> 重新分析
          </label>
          <button class="btn btn-ghost" data-action="open-source" type="button">查看原文</button>
          <button class="btn btn-ghost" data-action="start-parse" type="button"
                  ${running ? 'disabled' : ''}>仅解析</button>
          <button class="btn btn-primary" data-action="start-full" type="button"
                  ${running ? 'disabled' : ''}>${running ? '审查中…' : '开始审查'}</button>
          ${d.source === 'upload' ? '<button class="btn btn-ghost btn-sm" data-action="delete" type="button">删除</button>' : ''}
        </div>
      </div>
      ${chips ? `<div class="meta-chips">${chips}</div>` : ''}
    </section>`;
}

function runBannerView() {
  const t = S.task;
  if (!t || t.status !== 'running') return '';
  const p = t.progress || {};
  return `
    <div class="run-banner" id="runBanner">
      <div class="rb-top">
        <i class="pulse-dot"></i>
        <span class="rb-stage" id="rbStage">${esc(p.stage || '排队中')}</span>
        <span class="rb-pct" id="rbPct">${p.percent || 0}%</span>
      </div>
      <div class="progress"><i class="progress-bar" id="rbBar" style="width:${p.percent || 0}%"></i></div>
      <div class="rb-note" id="rbNote">已用时 ${period(t.elapsed_s)} · 判定 ${p.judge_done || 0}/${p.terms_total || 0} 条 ·
        复核 ${p.review_done || 0}/${p.evidence_total || 0} 条</div>
    </div>`;
}

function updateRunBanner() {
  const t = S.task;
  if (!t) return;
  const p = t.progress || {};
  if (!$('#runBanner')) { renderMain(); return; }
  $('#rbStage').textContent = p.stage || '';
  $('#rbPct').textContent = (p.percent || 0) + '%';
  $('#rbBar').style.width = (p.percent || 0) + '%';
  $('#rbNote').textContent =
    `已用时 ${period(t.elapsed_s)} · 判定 ${p.judge_done || 0}/${p.terms_total || 0} 条 · 复核 ${p.review_done || 0}/${p.evidence_total || 0} 条`;
}

function statGridView(st) {
  const card = (label, value, unit, note, cls = '') => `
    <div class="stat-card">
      <div class="sc-label">${label}</div>
      <div class="sc-value ${cls}">${value}${unit ? `<small>${unit}</small>` : ''}</div>
      ${note ? `<div class="sc-note">${note}</div>` : ''}
    </div>`;
  return `
    <div class="stat-grid">
      ${card('条款', st.terms, '条', '解析 Agent 切分')}
      ${card('风险', st.risks, '条', `高 ${st.high} · 中 ${st.mid} · 低 ${st.low}`, st.risks ? 'is-high' : 'is-none')}
      ${card('待人工', st.needsHuman, '条', '定位失败或复核失败')}
      ${card('定位成功率', st.located == null ? '—' : pct(st.located), '', '证据校验口径', '')}
      ${card('耗时', st.duration == null ? '—' : Math.round(st.duration), 's', '端到端')}
      ${card('判定 token', st.tokens ? fmt(st.tokens) : '—', '', st.calls ? `${st.calls} 次模型调用` : '')}
    </div>`;
}

function tabsView(st) {
  const tabs = [
    { id: 'risks', label: '风险清单', count: st.risks },
    { id: 'terms', label: '条款全览', count: st.terms },
    { id: 'snapshots', label: '任务快照', count: st.terms },
    { id: 'metrics', label: '验收指标', count: null },
    { id: 'regression', label: '回归池', count: S.regression ? S.regression.items.length : null },
  ];
  return `<nav class="tabs">${tabs.map((t) => `
    <button class="tab ${S.tab === t.id ? 'active' : ''}" data-action="tab" data-tab="${t.id}" type="button">
      ${t.label}${t.count != null ? `<span class="tab-count">${t.count}</span>` : ''}
    </button>`).join('')}</nav>`;
}

function unparsedView() {
  return `
    <div class="empty" style="margin-top:16px">
      <div class="empty-title">这份合同还没有切分成条款</div>
      <p class="empty-desc">
        先跑一次解析：正则锚点找疑似条款起点，再由模型判定每段是独立条款、并入上文还是丢弃。
        只想看切分质量的话选「仅解析」，不产生模型费用。
      </p>
      <div class="empty-actions">
        <button class="btn btn-primary" data-action="start-full" type="button">开始审查</button>
        <button class="btn btn-ghost" data-action="start-parse" type="button">仅解析</button>
      </div>
    </div>`;
}

function tabBodyView(st) {
  switch (S.tab) {
    case 'terms': return termsView();
    case 'snapshots': return snapshotsView();
    case 'metrics': return metricsView();
    case 'regression': return regressionView();
    default: return risksView(st);
  }
}

/* ---------------------------------------------------------------- 风险清单 */
function risksView(st) {
  const list = finals().filter((f) => f.has_risk)
    .sort((a, b) => (RISK_RANK[a.risk_level] ?? 9) - (RISK_RANK[b.risk_level] ?? 9)
      || (a.term_no || 0) - (b.term_no || 0));

  const chips = [
    { id: 'all', label: '全部', n: list.length },
    { id: '高', label: '高危', n: st.high },
    { id: '中', label: '中危', n: st.mid },
    { id: '低', label: '低危', n: st.low },
    { id: 'human', label: '待人工', n: st.needsHuman },
  ];
  const filtered = list.filter((f) =>
    S.riskFilter === 'all' ? true : (S.riskFilter === 'human' ? f.needs_human : f.risk_level === S.riskFilter));

  const filterRow = `<div class="filter-row">${chips.map((c) => `
    <button class="filter-chip ${S.riskFilter === c.id ? 'active' : ''}"
            data-action="filter" data-filter="${c.id}" type="button">
      ${c.label}<span class="n">${c.n}</span>
    </button>`).join('')}</div>`;

  if (!list.length) {
    const parseOnly = (S.detail.result || {}).mode === 'parse';
    return `${filterRow}
      <div class="empty">
        <div class="empty-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"
               stroke-linecap="round" stroke-linejoin="round"><path d="m5 13 4 4L19 7"></path></svg>
        </div>
        <div class="empty-title">${parseOnly ? '只完成了条款解析' : '本次审查未发现风险条款'}</div>
        <p class="empty-desc">${parseOnly
          ? '条款已切分完毕，但还没有跑判定与复核。点上方「开始审查」得到完整的风险结论。'
          : '所有条款判定为无风险。逐条结论可在「条款全览」里核对，包括判定理由与轮次。'}</p>
      </div>`;
  }

  const cards = filtered.map((f) => {
    const ev = f.verify_status && f.verify_status !== '不适用'
      ? `<span class="tag ${EVIDENCE_CLASS[f.verify_status] || ''}">${esc(f.verify_status)}${f.similarity ? ' ' + f.similarity.toFixed(2) : ''}</span>`
      : '';
    return `
      <article class="risk-card ${levelClass(f.risk_level)}" data-action="locate"
               data-term="${esc(f.term_id)}" title="点击定位到原文">
        <div class="rc-head">
          <span class="lv-badge ${levelClass(f.risk_level)}">${levelText(f.risk_level)}</span>
          <span class="rc-no">第 ${f.term_no} 条</span>
          ${f.action ? `<span class="tag ${ACTION_CLASS[f.action] || ''}">${esc(f.action)}</span>` : ''}
          ${ev}
          ${f.needs_human ? '<span class="tag tag-warn">待人工</span>' : ''}
          <span class="rc-spacer"></span>
          <button class="link-btn" data-action="locate" data-term="${esc(f.term_id)}" type="button">定位原文 ↗</button>
        </div>
        <p class="rc-reason">${esc(f.reason || '（无理由）')}</p>
        ${f.basis ? `<div class="rc-row"><span class="rc-k">依据</span><span>${esc(f.basis)}</span></div>` : ''}
        ${f.evidence ? `<blockquote class="rc-quote">${esc(f.evidence)}</blockquote>` : ''}
      </article>`;
  }).join('');

  return `${filterRow}
    <div class="risk-list">${cards || '<div class="empty"><div class="empty-desc">该筛选下没有条款</div></div>'}</div>`;
}

/* ---------------------------------------------------------------- 条款全览 */
function termsView() {
  const terms = (S.detail.terms || []).slice().sort((a, b) => a.term_no - b.term_no);
  const withRisk = terms.filter((t) => t.final && t.final.has_risk).length;
  const chips = [
    { id: 'all', label: '全部', n: terms.length },
    { id: 'risk', label: '有风险', n: withRisk },
    { id: 'clean', label: '无风险', n: terms.length - withRisk },
  ];
  const filtered = terms.filter((t) => {
    const r = !!(t.final && t.final.has_risk);
    return S.termFilter === 'all' || (S.termFilter === 'risk' ? r : !r);
  });

  const rows = filtered.map((t) => {
    const f = t.final || {};
    const lv = f.has_risk ? f.risk_level : (f.action === '无风险' ? '无' : null);
    const side = [];
    if (lv) side.push(`<span class="lv-badge ${levelClass(lv)}">${levelText(lv)}</span>`);
    if (f.action) side.push(`<span class="tag ${ACTION_CLASS[f.action] || ''}">${esc(f.action)}</span>`);
    if (t.complexity === 'complex') side.push('<span class="tag tag-info">复杂</span>');
    if (t.judge && t.judge.rounds) side.push(`<span class="tag tag-mute">${t.judge.rounds} 轮</span>`);
    if (!f.action && t.judge) side.push(`<span class="tag">${esc(t.judge.status)}</span>`);

    return `
      <div class="term-row ${f.has_risk ? 'risk' : ''}" data-action="locate" data-term="${esc(t.term_id)}">
        <div class="tr-no">第 ${t.term_no} 条</div>
        <div class="tr-text">${esc(t.term_text)}</div>
        <div class="tr-side">${side.join('') || '<span class="tr-empty">未判定</span>'}</div>
      </div>`;
  }).join('');

  return `
    <div class="filter-row">${chips.map((c) => `
      <button class="filter-chip ${S.termFilter === c.id ? 'active' : ''}"
              data-action="term-filter" data-filter="${c.id}" type="button">
        ${c.label}<span class="n">${c.n}</span></button>`).join('')}
    </div>
    <div class="term-list">${rows || '<div class="empty"><div class="empty-desc">没有条款</div></div>'}</div>`;
}

/* ---------------------------------------------------------------- 任务快照 */
function snapshotsView() {
  const terms = (S.detail.terms || []).slice().sort((a, b) => a.term_no - b.term_no);
  if (!terms.length) return '<div class="empty"><div class="empty-desc">还没有任务快照</div></div>';

  const rows = terms.map((t) => {
    const j = t.judge, e = t.evidence, r = t.review;
    const judgeTag = j
      ? `<span class="tag ${j.status === 'FINISHED' ? 'tag-ok' : (j.status === 'FAILED' ? 'tag-bad' : 'tag-warn')}">${esc(j.status)}</span>`
      : '<span class="tr-empty">—</span>';
    const evTag = e
      ? `<span class="tag ${EVIDENCE_CLASS[e.verify_status] || ''}">${esc(e.verify_status)}</span>`
      : '<span class="tr-empty">—</span>';
    return `
      <tr data-action="locate" data-term="${esc(t.term_id)}" style="cursor:pointer">
        <td>第 ${t.term_no} 条${t.complexity === 'complex' ? ' <span class="tag tag-info">复杂</span>' : ''}</td>
        <td>${judgeTag}</td>
        <td class="num">${j ? `${j.rounds}/${j.max_rounds}` : '—'}</td>
        <td class="num">${j ? fmt(j.tokens) : '—'}</td>
        <td>${evTag}${e && e.similarity ? ` <span class="tr-empty">${e.similarity.toFixed(2)}</span>` : ''}</td>
        <td>${r && r.action ? `<span class="tag ${ACTION_CLASS[r.action] || ''}">${esc(r.action)}</span>` : '—'}</td>
        <td class="num">${j && j.tool_calls ? j.tool_calls.length : 0}</td>
      </tr>`;
  }).join('');

  return `
    <div class="table-wrap">
      <table class="snap">
        <thead><tr>
          <th>条款</th><th>判定状态</th><th>轮次</th><th>tokens</th>
          <th>证据校验</th><th>复核动作</th><th>工具调用</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p class="mc-desc" style="margin-top:10px">
      轮次 = 已用轮次 / 复杂度对应的上限（简单 4 轮、复杂 12 轮）；tokens 为该任务累计消耗，
      超过阈值会熔断。快照逐轮落库，中断后重跑只补未完成的分支。
    </p>`;
}

/* ---------------------------------------------------------------- 验收指标 */
function metricsView() {
  const r = S.detail.report;
  if (!r || !Object.keys(r).length) {
    return `<div class="empty"><div class="empty-desc">这次运行没有生成验收报告（仅解析模式不产出指标）</div></div>`;
  }
  const ev = r.evidence || {};
  const rv = r.review || {};
  const cost = r.cost || {};
  const g = r.golden || {};
  const rec = r.recovery || {};

  const card = (title, value, ratio, desc, cls = '') => `
    <div class="metric-card">
      <div class="mc-title">${title}</div>
      <div class="mc-value">${value}</div>
      ${ratio == null ? '' : `<div class="mc-bar"><i class="${cls}" style="width:${Math.max(2, Math.min(100, ratio * 100))}%"></i></div>`}
      <div class="mc-desc">${desc}</div>
    </div>`;

  const evRatio = ev['定位成功率'];
  const goldenHtml = g.has_golden ? `
    <div class="section-title">金标比对 <span>与 final_gt 人工标注按字符区间 IoU ≥ 0.3 判定命中</span></div>
    <div class="metrics-grid">
      ${card('金标召回', pct(g.recall), g.recall, `标注 ${g.golden_total} 条，命中 ${g.golden_hit} 条`, 'bar-none')}
      ${card('高危漏检', String(g.high_risk_miss ?? 0), null,
        g.high_risk_miss ? '高危标注未被命中，优先排查' : '高危标注全部命中', 'bar-high')}
      ${card('误报', String(g.false_alarm ?? 0), null, '有风险结论未能对上任何标注', 'bar-high')}
    </div>` : `
    <div class="section-title">金标比对</div>
    <p class="mc-desc">该合同没有人工标注，跳过召回比对。</p>`;

  return `
    <div class="metrics-grid">
      ${card('证据定位成功率', pct1(evRatio), evRatio,
        `定位成功 ${ev['定位成功'] || 0} · 定位失败 ${ev['定位失败'] || 0} · 缺证 ${ev['缺少有效证据'] || 0}`)}
      ${card('复核修正率', pct1(rv['修正率']), rv['修正率'],
        `已复核 ${rv['已复核'] || 0} 条，修正 ${rv['修正'] || 0} 条 · 修正率高说明判定口径不稳`)}
      ${card('复核驳回率', pct1(rv['驳回率']), rv['驳回率'],
        `驳回 ${rv['驳回'] || 0} 条 · 驳回意味着证据真实但结论不成立`)}
      ${card('单条款 token', cost.tokens_per_term == null ? '—' : String(cost.tokens_per_term), null,
        `判定累计 ${fmt(cost.judge_tokens)} tokens，模型调用 ${cost.llm_calls || 0} 次`)}
    </div>
    ${goldenHtml}
    <div class="section-title">调度与恢复</div>
    <div class="metrics-grid">
      ${card('并行分支', String(rec.branches || 0), null, '本次 Fan-out 的判定任务数')}
      ${card('重试分支', String(rec.retried || 0), null, '失败后被重新派发的任务数；0 表示一次通过')}
      ${card('端到端耗时', period(r.duration_s), null, `平均每条款 ${period((r.duration_s || 0) / Math.max(1, r.terms_total))}`)}
    </div>`;
}

/* ---------------------------------------------------------------- 回归池 */
function regressionView() {
  if (!S.regression) {
    loadRegression();
    return '<div class="empty"><div class="empty-desc">加载中…</div></div>';
  }
  const items = S.regression.items || [];
  if (!items.length) {
    return `<div class="empty">
      <div class="empty-title">回归池是空的</div>
      <p class="empty-desc">判定给出风险但引用的证据在原文里找不到时，结论会被判无效并沉淀到这里，
      供后续补充证据与回归测试。</p></div>`;
  }
  return items.map((it) => {
    const det = it.detail && Object.keys(it.detail).length ? JSON.stringify(it.detail, null, 1) : '';
    return `
      <div class="reg-card">
        <div class="reg-head">
          <span class="tag tag-bad">无效结论</span>
          <span>${esc(it.contract_id)} · ${esc(it.term_id.split('-T').pop())} 条</span>
          <span style="margin-left:auto">${fmtTime(it.created_at)}</span>
        </div>
        <div class="reg-reason">${esc(it.reason)}</div>
        ${det ? `<div class="reg-detail">${esc(det)}</div>` : ''}
      </div>`;
  }).join('');
}

async function loadRegression() {
  try {
    S.regression = await api('/api/regression');
  } catch (e) {
    S.regression = { items: [] };
    toast('读取回归池失败：' + e.message, true);
  }
  if (S.tab === 'regression') {
    const body = $('#tabBody');
    if (body) body.innerHTML = tabBodyView(statsOf());
  }
}

/* ---------------------------------------------------------------- 原文抽屉 */
function openSource(termId) {
  S.sourceOpen = true;
  $('#sourcePanel').classList.add('open');
  $('#sourcePanel').setAttribute('aria-hidden', 'false');
  renderSource();
  if (termId) requestAnimationFrame(() => locate(termId));
}

function closeSource() {
  S.sourceOpen = false;
  $('#sourcePanel').classList.remove('open');
  $('#sourcePanel').setAttribute('aria-hidden', 'true');
}

function renderSource() {
  const d = S.detail;
  if (!d) return;
  const text = d.clean_text || '';
  $('#sourceTitle').textContent = d.file_name;
  $('#sourceMeta').textContent = `${d.raw_chars} 字 · ${d.terms.length} 条条款`;

  const anchors = finals()
    .filter((f) => f.has_risk && Number.isFinite(f.anchor_start) && f.anchor_end > f.anchor_start)
    .sort((a, b) => a.anchor_start - b.anchor_start);

  let html = '';
  let cursor = 0;
  let idx = 0;
  S.hlIndex = {};
  for (const f of anchors) {
    if (f.anchor_start < cursor) continue;             // 锚点重叠时只保留靠前的一个
    html += esc(text.slice(cursor, f.anchor_start));
    html += `<mark class="hl ${levelClass(f.risk_level)}" id="hl-${idx}">`
      + esc(text.slice(f.anchor_start, f.anchor_end)) + '</mark>';
    S.hlIndex[f.term_id] = idx;
    idx += 1;
    cursor = f.anchor_end;
  }
  html += esc(text.slice(cursor));
  $('#sourceBody').innerHTML = html || '<span class="tr-empty">（无原文）</span>';
  $('#sourceBody').scrollTop = 0;
}

function locate(termId) {
  if (!S.sourceOpen) openSource();
  const body = $('#sourceBody');
  const idx = S.hlIndex[termId];
  if (idx != null) {
    const el = document.getElementById('hl-' + idx);
    if (el) {
      el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      el.classList.remove('pulse');
      void el.offsetWidth;                              // 强制重排，让动画能重播
      el.classList.add('pulse');
      return;
    }
  }
  const t = (S.detail.terms || []).find((x) => x.term_id === termId);
  if (!t) return;
  const ratio = t.char_start / Math.max(1, (S.detail.clean_text || '').length);
  body.scrollTop = Math.max(0, ratio * body.scrollHeight - body.clientHeight / 2);
  toast('该条款没有可定位的证据锚点，已按位置跳转');
}

/* ---------------------------------------------------------------- 任务 */
async function startReview(mode) {
  if (!S.current) return;
  try {
    const r = await api(`/api/contracts/${encodeURIComponent(S.current)}/review`, {
      method: 'POST',
      body: { fresh: S.fresh, parse_only: mode === 'parse' },
    });
    toast(r.reused ? '该合同已有任务在跑，继续跟随进度' : (mode === 'parse' ? '开始解析…' : '开始审查…'));
    watch({ task_id: r.task_id, status: 'running', progress: { stage: '排队中', percent: 2 } });
    S.detail.running = S.task;
    renderMain();
  } catch (e) {
    toast(e.message, true);
  }
}

function watch(task) {
  S.task = task;
  stopPolling();
  renderMain();

  const tick = async () => {
    try {
      const t = await api('/api/tasks/' + encodeURIComponent(S.task.task_id));
      S.task = t;
      if (t.status === 'running') {
        updateRunBanner();
      } else {
        stopPolling();
        await loadContracts();
        await reloadDetail();
        if (t.status === 'done') {
          const s = t.summary || {};
          toast(t.mode === 'parse'
            ? `解析完成，切出 ${s.terms_total || 0} 条条款`
            : `审查完成：${s.risk_total || 0} 条风险，其中待人工 ${s.needs_human || 0} 条`);
        } else {
          toast('任务失败：' + (t.error || '未知错误'), true);
        }
      }
    } catch (e) {
      stopPolling();
      toast('进度查询失败：' + e.message, true);
    }
  };
  tick();
  S.timer = setInterval(tick, 1200);
}

function stopPolling() {
  if (S.timer) { clearInterval(S.timer); S.timer = null; }
}

/* ---------------------------------------------------------------- 上传 / 粘贴 */
async function uploadFile(file) {
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  try {
    toast('正在上传…');
    const r = await api('/api/contracts/upload', { method: 'POST', body: fd });
    await loadContracts();
    await selectContract(r.contract_id);
    toast(`已存入合同库：${r.file_name}（${r.chars} 字）`);
  } catch (e) {
    toast(e.message, true);
  }
}

async function submitPaste() {
  const name = $('#pasteName').value.trim() || '粘贴的合同.txt';
  const text = $('#pasteText').value;
  if (text.trim().length < 50) { toast('正文太短，至少 50 字', true); return; }
  $('#pasteHint').textContent = '提交中…';
  try {
    const r = await api('/api/contracts/text', { method: 'POST', body: { name, text } });
    closeModal();
    await loadContracts();
    await selectContract(r.contract_id);
    toast(`已存入合同库：${r.file_name}（${r.chars} 字）`);
  } catch (e) {
    $('#pasteHint').textContent = '';
    toast(e.message, true);
  }
}

function openModal() {
  $('#modal').classList.add('open');
  $('#modal').setAttribute('aria-hidden', 'false');
  $('#pasteName').value = '';
  $('#pasteText').value = '';
  $('#pasteHint').textContent = '';
  setTimeout(() => $('#pasteText').focus(), 60);
}

function closeModal() {
  $('#modal').classList.remove('open');
  $('#modal').setAttribute('aria-hidden', 'true');
}

async function deleteContract() {
  if (!S.current) return;
  if (!confirm('删除这份上传的合同？解析结果、快照与结论会一并清除。')) return;
  try {
    await api('/api/contracts/' + encodeURIComponent(S.current), { method: 'DELETE' });
    S.current = null;
    S.detail = null;
    await loadContracts();
    const next = S.contracts[0];
    if (next) await selectContract(next.contract_id); else renderMain();
    toast('已删除');
  } catch (e) {
    toast(e.message, true);
  }
}

/* ---------------------------------------------------------------- 事件绑定 */
function initEvents() {
  $('#contractList').addEventListener('click', (ev) => {
    const el = ev.target.closest('[data-cid]');
    if (el) selectContract(el.dataset.cid);
  });

  $('#main').addEventListener('click', (ev) => {
    const el = ev.target.closest('[data-action]');
    if (!el) return;
    const action = el.dataset.action;

    if (action === 'tab') {
      S.tab = el.dataset.tab;
      renderMain();
    } else if (action === 'filter') {
      S.riskFilter = el.dataset.filter;
      renderMain();
    } else if (action === 'term-filter') {
      S.termFilter = el.dataset.filter;
      renderMain();
    } else if (action === 'locate') {
      locate(el.dataset.term);
    } else if (action === 'open-source') {
      openSource();
    } else if (action === 'start-full') {
      startReview('full');
    } else if (action === 'start-parse') {
      startReview('parse');
    } else if (action === 'delete') {
      deleteContract();
    } else if (action === 'paste') {
      openModal();
    } else if (action === 'upload') {
      $('#fileInput').click();
    }
  });

  $('#main').addEventListener('change', (ev) => {
    if (ev.target.id === 'freshToggle') S.fresh = ev.target.checked;
  });

  $('#search').addEventListener('input', (ev) => {
    S.query = ev.target.value;
    renderList();
  });

  $('#btnUpload').addEventListener('click', () => $('#fileInput').click());
  $('#fileInput').addEventListener('change', (ev) => {
    const file = ev.target.files && ev.target.files[0];
    ev.target.value = '';
    uploadFile(file);
  });

  $('#btnPaste').addEventListener('click', openModal);
  $('#modalClose').addEventListener('click', closeModal);
  $('#pasteSubmit').addEventListener('click', submitPaste);
  $('#modal').addEventListener('click', (ev) => { if (ev.target.id === 'modal') closeModal(); });

  $('#sourceClose').addEventListener('click', closeSource);

  document.addEventListener('keydown', (ev) => {
    if (ev.key !== 'Escape') return;
    if ($('#modal').classList.contains('open')) closeModal();
    else if (S.sourceOpen) closeSource();
  });
}

initEvents();
boot();
