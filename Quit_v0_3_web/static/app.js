'use strict';

// ── Workflow Steps ────────────────────────────────────────────────────────────
const STEP_DEFS = [
  'PLAN',
  'VALIDATE_BRIEF',
  'RETRIEVE',
  'READ',
  'IDEATE',
  'IDEA_EVAL',
  'BUILD_SPEC',
  'CODE',
  'CODE_EVAL',
  'WRITE',
  'WRITE_EVAL',
  'EXTRACT',
];
const STEPS = STEP_DEFS.map(id => ({ id, label: t('step.' + id) }));
const STEP_IDS = STEPS.map(s => s.id);

// ── App State ─────────────────────────────────────────────────────────────────
const state = {
  startAt:     'PLAN',
  stopAfter:   'WRITE',
  currentStep: null,   // step currently running
  jobId:       null,
  autoScroll:  true,
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const stepBar       = $('step-bar');
const selStartAt    = $('sel-start-at');
const selStopAfter  = $('sel-stop-after');
const logPane       = $('log-pane');
const btnRun        = $('btn-run');
const btnCancel     = $('btn-cancel');
const runStatus     = $('run-status');
const btnAutoScroll = $('btn-autoscroll');
const runIdInput    = $('run-id-input');

// ── Build Step Bar ────────────────────────────────────────────────────────────
function buildStepBar() {
  stepBar.innerHTML = '';
  STEPS.forEach((step, i) => {
    const node = document.createElement('div');
    node.className = 'step-node';

    const chip = document.createElement('button');
    chip.className = 'step-chip';
    chip.dataset.stepId = step.id;
    chip.title = t('step.tooltip', { id: step.id });

    const labelSpan = document.createElement('span');
    labelSpan.textContent = step.label;
    chip.appendChild(labelSpan);

    chip.addEventListener('click', e => {
      if (e.ctrlKey || e.metaKey) {
        state.startAt = step.id;
        selStartAt.value = step.id;
      } else {
        state.stopAfter = step.id;
        selStopAfter.value = step.id;
      }
      updateStepHighlights();
    });

    node.appendChild(chip);
    stepBar.appendChild(node);

    if (i < STEPS.length - 1) {
      const arrow = document.createElement('span');
      arrow.className = 'step-arrow';
      arrow.textContent = '→';
      stepBar.appendChild(arrow);
    }
  });
}

// ── Populate Step Selects ─────────────────────────────────────────────────────
function buildStepSelects() {
  [selStartAt, selStopAfter].forEach(sel => {
    STEPS.forEach(step => {
      const opt = document.createElement('option');
      opt.value = step.id;
      opt.textContent = step.label;
      sel.appendChild(opt);
    });
  });
  selStartAt.value = state.startAt;
  selStopAfter.value = state.stopAfter;

  selStartAt.addEventListener('change', () => {
    state.startAt = selStartAt.value;
    updateStepHighlights();
  });
  selStopAfter.addEventListener('change', () => {
    state.stopAfter = selStopAfter.value;
    updateStepHighlights();
  });
}

// ── Update Step Highlights ────────────────────────────────────────────────────
function updateStepHighlights() {
  const startIdx = STEP_IDS.indexOf(state.startAt);
  const stopIdx  = STEP_IDS.indexOf(state.stopAfter);
  const runIdx   = state.currentStep ? STEP_IDS.indexOf(state.currentStep) : -1;

  document.querySelectorAll('.step-chip').forEach((chip, i) => {
    // preserve child spans
    const labelSpan = chip.querySelector('span:not(.step-badge)');
    chip.className = 'step-chip';
    if (labelSpan) chip.appendChild(labelSpan);

    // remove old badges
    chip.querySelectorAll('.step-badge').forEach(b => b.remove());

    const inRange = (startIdx <= stopIdx)
      ? (i >= startIdx && i <= stopIdx)
      : (i === startIdx);

    if (inRange)        chip.classList.add('in-range');
    if (i === startIdx) { chip.classList.add('is-start'); _addBadge(chip, 'S'); }
    if (i === stopIdx)  { chip.classList.add('is-stop');  _addBadge(chip, 'E'); }
    if (i === runIdx)   { chip.classList.add('is-running'); _addBadge(chip, '●'); }
  });
}

function _addBadge(chip, text) {
  const b = document.createElement('span');
  b.className = 'step-badge';
  b.textContent = text;
  chip.appendChild(b);
}

// ── Config: load / read / reset ───────────────────────────────────────────────
async function loadDefaultConfig() {
  try {
    const res = await fetch('/api/config');
    if (!res.ok) return;
    const cfg = await res.json();
    applyConfigToForm(cfg);
  } catch {}
}

function applyConfigToForm(cfg) {
  const p = cfg.project || {};
  setValue('project-id',       p.project_id   || '');
  setValue('project-topic',    p.topic        || '');
  setValue('project-domain',   p.domain       || '');
  setValue('project-objective',p.objective    || '');

  const llm = cfg.llm || {};
  const prov = llm.provider || 'openai';
  setProvider(prov);
  _updateTempRange(prov);
  _switchLLMFields(prov);

  setValue('llm-model',    llm.model    || '');
  setValue('llm-base-url', llm.base_url || '');
  setValue('llm-api-key',  llm.api_key  || '');
  const temp = llm.temperature ?? 0.7;
  setValue('llm-temperature',      temp);
  setValue('llm-temperature-vllm', temp);
  $('temp-badge').textContent      = parseFloat(temp).toFixed(2);
  const tvb = $('temp-badge-vllm');
  if (tvb) tvb.textContent = parseFloat(temp).toFixed(2);

  // vLLM fields
  const vllm = cfg.local_vllm || {};
  setValue('vllm-model-path',     vllm.model_path          || '');
  setValue('vllm-served-name',    vllm.served_model_name   || '');
  setValue('vllm-host',           vllm.host                || '127.0.0.1');
  setValue('vllm-port',           vllm.port                ?? 8000);
  setValue('vllm-tensor-parallel',vllm.tensor_parallel_size ?? 2);
  setValue('vllm-gpu-margin',     vllm.gpu_memory_safety_margin ?? 0.20);
  setValue('vllm-startup-timeout',vllm.startup_timeout_seconds  ?? 500);
  if (vllm.gpu_memory_utilization != null) setValue('vllm-gpu-util', vllm.gpu_memory_utilization);
  const kaCb = $('vllm-keep-alive');
  if (kaCb) kaCb.checked = !!vllm.keep_alive;

  const rt = cfg.runtime || {};
  setValue('runtime-max-steps', rt.max_steps ?? 30);
  if (rt.stop_after) {
    state.stopAfter = rt.stop_after;
    selStopAfter.value = rt.stop_after;
  }

  const sb = cfg.search_budget || {};
  setValue('sb-max-queries',   sb.max_queries         ?? 4);
  setValue('sb-max-screened',  sb.max_papers_screened ?? 60);
  setValue('sb-max-selected',  sb.max_papers_selected ?? 15);
  setValue('sb-max-repos',     sb.max_repo_checked    ?? 5);

  const rb = cfg.run_budget || {};
  setValue('rb-min-train', rb.min_train_epochs ?? 30);
  setValue('rb-max-train', rb.max_train_epochs ?? 100);
  setValue('rb-min-eval',  rb.min_eval_epochs  ?? 30);
  setValue('rb-max-eval',  rb.max_eval_epochs  ?? 100);

  const ret = cfg.retrieval || {};
  const sources = ret.sources || ['arxiv', 'openreview'];
  document.querySelectorAll('input[name="sources"]').forEach(cb => {
    cb.checked = sources.includes(cb.value);
  });
  setValue('ret-per-source', ret.per_source_results ?? 15);
  setValue('ret-max-dl',     ret.max_downloads      ?? 15);
  $('ret-download-pdfs').checked = ret.download_pdfs ?? true;
  const readMode = (cfg.read || {}).mode || 'direct_pdf';
  document.querySelectorAll('input[name="read-mode"]').forEach(r => {
    r.checked = (r.value === readMode);
  });

  const agents = cfg.agents || {};
  setValue('ag-planner',  (agents.planner  || {}).max_tokens ?? 4096);
  setValue('ag-research', (agents.research || {}).max_tokens ?? 4096);
  setValue('ag-reviewer', (agents.reviewer || {}).max_tokens ?? 2048);
  setValue('ag-builder',  (agents.builder  || {}).max_tokens ?? 32768);

  updateStepHighlights();
}

function buildConfigFromForm() {
  const sources  = [...document.querySelectorAll('input[name="sources"]:checked')].map(cb => cb.value);
  const readMode = document.querySelector('input[name="read-mode"]:checked')?.value || 'direct_pdf';
  const provider = getProvider();
  const isVllm   = provider === 'local-vllm';

  // temperature comes from the visible slider
  const tempVal = parseFloat(val(isVllm ? 'llm-temperature-vllm' : 'llm-temperature')) || 0.7;

  // build local_vllm section only when needed
  const gpuUtil = val('vllm-gpu-util');
  const localVllm = isVllm ? {
    model_path:               val('vllm-model-path'),
    served_model_name:        val('vllm-served-name'),
    host:                     val('vllm-host') || '127.0.0.1',
    port:                     intVal('vllm-port') || 8000,
    tensor_parallel_size:     intVal('vllm-tensor-parallel') || 1,
    gpu_memory_safety_margin: parseFloat(val('vllm-gpu-margin')) || 0.20,
    startup_timeout_seconds:  intVal('vllm-startup-timeout') || 500,
    gpu_memory_utilization:   gpuUtil ? parseFloat(gpuUtil) : null,
    keep_alive:               !!$('vllm-keep-alive')?.checked,
  } : undefined;

  return {
    project: {
      project_id: val('project-id'),
      topic:      val('project-topic'),
      domain:     val('project-domain'),
      objective:  val('project-objective'),
    },
    runtime: {
      data_dir:   'runs',
      env_file:   '.env',
      stop_after: state.stopAfter,
      max_steps:  intVal('runtime-max-steps') || 30,
    },
    llm: {
      provider:    provider,
      model:       isVllm ? (val('vllm-served-name') || 'model') : val('llm-model'),
      base_url:    isVllm ? null : (val('llm-base-url') || null),
      api_key:     isVllm ? null : (val('llm-api-key')  || null),
      temperature: tempVal,
    },
    ...(localVllm ? { local_vllm: localVllm } : {}),
    search_budget: {
      max_queries:              intVal('sb-max-queries'),
      max_papers_screened:      intVal('sb-max-screened'),
      max_papers_selected:      intVal('sb-max-selected'),
      max_repo_checked:         intVal('sb-max-repos'),
      stop_if_no_new_signal_rounds: 3,
    },
    retrieval: {
      sources,
      per_source_results: intVal('ret-per-source'),
      max_downloads:      intVal('ret-max-dl'),
      download_pdfs:      $('ret-download-pdfs').checked,
      local_database_path:'paper_database/local_papers',
      pdf_dir:            'paper_retrieve',
      timeout_seconds:    200,
      relevance_selection_ratio: 0.7,
    },
    read: {
      mode: readMode,
      max_direct_pdf_mb: 10.0,
      read_workers: 4,
    },
    agents: {
      planner:  { max_tokens: intVal('ag-planner'),  timeout_seconds: 360 },
      research: { max_tokens: intVal('ag-research'), timeout_seconds: 360 },
      reviewer: { max_tokens: intVal('ag-reviewer'), timeout_seconds: 360 },
      builder:  { max_tokens: intVal('ag-builder'),  timeout_seconds: 900 },
    },
    build_budget: { max_code_iterations: 5, max_experiments: 5, max_review_revisions: 2 },
    run_budget: {
      min_train_epochs: intVal('rb-min-train') ?? 30,
      max_train_epochs: intVal('rb-max-train') ?? 100,
      min_eval_epochs:  intVal('rb-min-eval')  ?? 30,
      max_eval_epochs:  intVal('rb-max-eval')  ?? 100,
      experiment_timeout_seconds: 3600,
    },
    dataset:      { max_download_attempts: 3 },
    write:        { expected_main_pages: 8, latex_timeout_seconds: 900 },
  };
}

// ── Provider Buttons ──────────────────────────────────────────────────────────
function setProvider(name) {
  document.querySelectorAll('.provider-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.provider === name);
  });
}
function getProvider() {
  return document.querySelector('.provider-btn.active')?.dataset.provider || 'openai';
}
// providers that don't accept temperature (or require exactly 1)
const NO_TEMP_PROVIDERS = new Set(['o1', 'o3']);
// providers with temperature range 0-1 (not 0-2)
const TEMP_MAX_1 = new Set(['anthropic']);

document.querySelectorAll('.provider-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    setProvider(btn.dataset.provider);
    _updateTempRange(btn.dataset.provider);
    _switchLLMFields(btn.dataset.provider);
  });
});

function _switchLLMFields(provider) {
  const isVllm = provider === 'local-vllm';
  $('llm-standard-fields').style.display = isVllm ? 'none' : '';
  $('llm-vllm-fields').style.display     = isVllm ? '' : 'none';
}

function _updateTempRange(provider) {
  const slider = $('llm-temperature');
  const badge  = $('temp-badge');
  const field  = slider?.closest('.field');
  if (!slider) return;

  if (TEMP_MAX_1.has(provider)) {
    slider.max = '1';
    if (parseFloat(slider.value) > 1) { slider.value = '1'; badge.textContent = '1.00'; }
    if (field) field.title = 'Anthropic supports 0–1';
  } else {
    slider.max = '2';
    if (field) field.title = '';
  }
}

// ── Temperature sliders ───────────────────────────────────────────────────────
$('llm-temperature').addEventListener('input', e => {
  $('temp-badge').textContent = parseFloat(e.target.value).toFixed(2);
});
document.addEventListener('input', e => {
  if (e.target.id === 'llm-temperature-vllm') {
    const b = $('temp-badge-vllm');
    if (b) b.textContent = parseFloat(e.target.value).toFixed(2);
  }
});

// ── API key toggle ────────────────────────────────────────────────────────────
$('btn-toggle-key').addEventListener('click', () => {
  const inp = $('llm-api-key');
  inp.type = inp.type === 'password' ? 'text' : 'password';
});

// ── Collapsible sections ──────────────────────────────────────────────────────
document.querySelectorAll('.toggle-title').forEach(title => {
  title.addEventListener('click', () => {
    const bodyId = title.dataset.target;
    const body   = $(bodyId);
    const isOpen = body.classList.toggle('open');
    title.classList.toggle('open', isOpen);
  });
});

// ── Auto-scroll toggle ────────────────────────────────────────────────────────
btnAutoScroll.classList.add('btn-autoscroll-active');
btnAutoScroll.addEventListener('click', () => {
  state.autoScroll = !state.autoScroll;
  btnAutoScroll.classList.toggle('btn-autoscroll-active', state.autoScroll);
});

// ── Clear log ─────────────────────────────────────────────────────────────────
$('btn-clear-log').addEventListener('click', () => {
  logPane.innerHTML = '';
  appendLog(t('log.cleared'), 'log-empty');
});

// ── Reset config ──────────────────────────────────────────────────────────────
$('btn-reset-config').addEventListener('click', loadDefaultConfig);

// ── Run history modal ─────────────────────────────────────────────────────────
const runsModal   = $('runs-modal');
const runsList    = $('runs-list');

$('btn-load-runs').addEventListener('click', async () => {
  runsList.innerHTML = '<p style="padding:12px;color:var(--text-muted)">' + t('modal.loading') + '</p>';
  runsModal.hidden = false;
  try {
    const res = await fetch('/api/runs');
    const runs = await res.json();
    runsList.innerHTML = '';
    if (!runs.length) {
      runsList.innerHTML = '<p style="padding:12px;color:var(--text-muted)">' + t('modal.noRuns') + '</p>';
      return;
    }
    runs.forEach(name => {
      const item = document.createElement('div');
      item.className = 'run-item';
      item.innerHTML = '<span class="run-item-name">' + escHtml(name) + '</span><span class="run-item-use">' + t('modal.use') + '</span>';
      item.addEventListener('click', () => {
        if (_runsModalSelectHandler) {
          _runsModalSelectHandler(name);
        } else {
          runIdInput.value = name;
          runsModal.hidden = true;
        }
      });
      runsList.appendChild(item);
    });
  } catch {
    runsList.innerHTML = '<p style="padding:12px;color:var(--danger)">' + t('modal.failedLoad') + '</p>';
  }
});

$('btn-close-modal').addEventListener('click', () => { runsModal.hidden = true; });
runsModal.addEventListener('click', e => { if (e.target === runsModal) runsModal.hidden = true; });

$('btn-pick-run').addEventListener('click', () => $('btn-load-runs').click());

// ── Start / Cancel Run ────────────────────────────────────────────────────────
btnRun.addEventListener('click', startRun);
btnCancel.addEventListener('click', cancelRun);

async function startRun() {
  const cfg = buildConfigFromForm();
  const maxSteps = intVal('runtime-max-steps') || 30;
  const runId    = runIdInput.value.trim() || null;

  setRunning(true);
  appendLog(t('run.starting', { startAt: state.startAt, stopAfter: state.stopAfter, maxSteps: String(maxSteps) }), 'pipeline');
  if (runId) appendLog('run_id: ' + runId, 'pipeline');

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config:    cfg,
        start_at:  state.startAt,
        stop_after: state.stopAfter,
        max_steps: maxSteps,
        run_id:    runId,
      }),
    });
    if (!res.ok) {
      const err = await res.text();
      appendLog('[ERROR] ' + err, 'error');
      setRunning(false);
      return;
    }
    const { job_id } = await res.json();
    state.jobId = job_id;
    streamLogs(job_id);
  } catch (e) {
    appendLog('[ERROR] ' + e.message, 'error');
    setRunning(false);
  }
}

async function cancelRun() {
  if (!state.jobId) return;
  try {
    await fetch('/api/jobs/' + state.jobId + '/cancel', { method: 'POST' });
  } catch {}
  appendLog(t('log.cancelled'), 'warn');
  setRunning(false);
}

function streamLogs(jobId) {
  let retries = 0;
  const MAX_RETRIES = 5;

  function connect() {
    const es = new EventSource('/api/stream/' + jobId);

    es.addEventListener('message', e => {
      retries = 0; // reset on successful message
      const data = e.data;
      if (data.startsWith('[DONE:')) {
        const [, status] = data.match(/\[DONE:(\w+):/) || [];
        const cls = status === 'done' ? 'done' : 'error';
        const label = status === 'done' ? t('run.status.done') : t('run.status.error');
        appendLog('[FINISHED] ' + label, cls);
        setRunStatus(status === 'done' ? 'success' : 'error', label);
        setRunning(false);
        state.currentStep = null;
        updateStepHighlights();
        es.close();
        if (state.completedRunId) {
          loadResultsForRun(state.completedRunId);
          state.completedRunId = null;
        }
        return;
      }
      const line = JSON.parse(data);
      const cls  = classifyLine(line);
      appendLog(line, cls);

      const m = line.match(/\[PIPELINE\] step \d+ START (\w+)/);
      if (m) { state.currentStep = m[1]; updateStepHighlights(); }

      const rm = line.match(/^run_id=(.+)$/);
      if (rm) state.completedRunId = rm[1].trim();
    });

    es.onerror = () => {
      es.close();
      retries++;
      if (retries <= MAX_RETRIES) {
        appendLog(t('log.sseLost') + ' (' + retries + '/' + MAX_RETRIES + ')', 'warn');
        setTimeout(connect, 2000);
      } else {
        appendLog(t('log.sseFailed'), 'error');
        setRunning(false);
      }
    };
  }

  connect();
}

function classifyLine(line) {
  if (line.includes('[PIPELINE]')) return 'pipeline';
  if (line.includes('[ERROR]') || line.toLowerCase().includes('error:')) return 'error';
  if (line.startsWith('[DONE')) return 'done';
  if (line.includes('[WARN') || line.includes('Warning')) return 'warn';
  return '';
}

// ── Log helpers ───────────────────────────────────────────────────────────────
function appendLog(text, extraClass = '') {
  // remove placeholder
  const placeholder = logPane.querySelector('.log-empty');
  if (placeholder) placeholder.remove();

  const div = document.createElement('div');
  div.className = ('log-line ' + extraClass).trim();
  div.textContent = text;
  logPane.appendChild(div);
  if (state.autoScroll) {
    logPane.scrollTop = logPane.scrollHeight;
  }
}

// ── UI state helpers ──────────────────────────────────────────────────────────
function setRunning(isRunning) {
  btnRun.disabled    = isRunning;
  btnCancel.disabled = !isRunning;
  if (!isRunning) state.jobId = null;
  if (isRunning) setRunStatus('running', t('run.status.running'));
  const indicator = $('log-indicator');
  if (indicator) indicator.classList.toggle('active', isRunning);
}

function setRunStatus(type, msg) {
  runStatus.className = 'run-status ' + type;
  runStatus.innerHTML = '<span class="status-dot"></span>' + msg;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function val(id)    { return $(id)?.value ?? ''; }
function intVal(id) { return parseInt($(id)?.value) || 0; }
function setValue(id, v) { const el = $(id); if (el) el.value = v; }
function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
document.querySelectorAll('.run-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.run-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    $('tab-' + tab.dataset.tab).classList.add('active');
  });
});

function switchTab(name) {
  document.querySelectorAll('.run-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
}

// ── Results: File Tree ────────────────────────────────────────────────────────
let _resultsRunId = null;

$('btn-results-pick').addEventListener('click', async () => {
  // reuse the runs modal
  $('btn-load-runs').click();
  // override click handler temporarily to load into results instead of run-id-input
  const origHandler = _runsModalSelectHandler;
  _runsModalSelectHandler = (name) => {
    loadResultsForRun(name);
    $('runs-modal').hidden = true;
    _runsModalSelectHandler = origHandler;
  };
});

let _runsModalSelectHandler = null;

async function loadResultsForRun(runId) {
  _resultsRunId = runId;
  $('results-run-label').textContent = runId;
  const body = $('file-tree-body');
  body.innerHTML = '<div class="file-tree-empty">' + t('results.loading') + '</div>';
  switchTab('results');

  try {
    const res = await fetch('/api/runs/' + encodeURIComponent(runId) + '/files');
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    body.innerHTML = '';
    renderTree(data.files, body, 0);
    // auto-open paper_gene folder if exists
    const pdfNode = body.querySelector('[data-rel="paper_gene/main.pdf"]');
    if (pdfNode) pdfNode.click();
  } catch (e) {
    body.innerHTML = '<div class="file-tree-empty" style="color:var(--danger)">' + escHtml(e.message) + '</div>';
  }
}

function renderTree(items, container, depth) {
  items.forEach(item => {
    if (item.type === 'dir') {
      const row = document.createElement('div');
      row.className = 'tree-dir-row';
      row.style.paddingLeft = (10 + depth * 14) + 'px';
      const toggle = document.createElement('span');
      toggle.className = 'tree-toggle';
      toggle.textContent = '▶';
      row.innerHTML = '<span class="tree-icon">📁</span><span class="tree-name">' + escHtml(item.name) + '</span>';
      row.prepend(toggle);

      const children = document.createElement('div');
      children.className = 'tree-children';

      // auto-open top-level important dirs
      const autoOpen = ['paper_gene', 'code', 'results'].includes(item.name) && depth === 0;
      if (autoOpen) { children.classList.add('open'); toggle.classList.add('open'); }

      row.addEventListener('click', () => {
        const open = children.classList.toggle('open');
        toggle.classList.toggle('open', open);
      });

      renderTree(item.children, children, depth + 1);
      container.appendChild(row);
      container.appendChild(children);
    } else {
      const el = document.createElement('div');
      el.className = 'tree-file' + (item.highlight ? ' highlight' : '');
      el.dataset.rel = item.rel;
      el.style.paddingLeft = (10 + depth * 14) + 'px';
      el.innerHTML =
        '<span class="tree-icon">' + fileIcon(item.ext) + '</span>' +
        '<span class="tree-name">' + escHtml(item.name) + '</span>' +
        '<span class="tree-size">' + fmtSize(item.size) + '</span>';
      el.addEventListener('click', () => {
        document.querySelectorAll('.tree-file.active').forEach(f => f.classList.remove('active'));
        el.classList.add('active');
        previewFile(_resultsRunId, item.rel, item.name, item.ext);
      });
      container.appendChild(el);
    }
  });
}

function fileIcon(ext) {
  if (ext === '.pdf') return '📕';
  if (['.py', '.js', '.ts', '.sh'].includes(ext)) return '🐍';
  if (ext === '.json' || ext === '.jsonl') return '{}';
  if (ext === '.md') return '📝';
  if (ext === '.tex' || ext === '.bib') return '📄';
  if (['.png', '.jpg', '.svg'].includes(ext)) return '🖼';
  return '📄';
}
function fmtSize(b) {
  if (b < 1024) return b + 'B';
  if (b < 1024 * 1024) return (b/1024).toFixed(0) + 'K';
  return (b/1024/1024).toFixed(1) + 'M';
}

// ── Preview ───────────────────────────────────────────────────────────────────
async function previewFile(runId, rel, name, ext) {
  const panel = $('preview-panel');
  const url   = '/api/runs/' + encodeURIComponent(runId) + '/file?path=' + encodeURIComponent(rel);

  if (ext === '.pdf') {
    panel.innerHTML =
      '<div class="preview-header">' +
        '<span class="preview-filename">' + escHtml(name) + '</span>' +
        '<a class="btn btn-ghost btn-sm preview-download" href="' + url + '" target="_blank">' + t('file.previewOpen') + '</a>' +
      '</div>' +
      '<div class="preview-content">' +
        '<embed src="' + url + '" type="application/pdf" />' +
      '</div>';
    return;
  }

  // text-based files
  panel.innerHTML =
    '<div class="preview-header">' +
      '<span class="preview-filename">' + escHtml(name) + '</span>' +
      '<a class="btn btn-ghost btn-sm preview-download" href="' + url + '" target="_blank" download>' + t('file.previewDownload') + '</a>' +
    '</div>' +
    '<div class="preview-content" id="preview-content-body">' +
      '<pre class="' + (ext === '.json' || ext === '.jsonl' ? 'preview-json' : 'preview-code') + '">' + t('file.previewLoading') + '</pre>' +
    '</div>';

  try {
    const res  = await fetch(url);
    const text = await res.text();
    const pre  = panel.querySelector('pre');
    if ((ext === '.json' || ext === '.jsonl') && ext !== '.jsonl') {
      try { pre.textContent = JSON.stringify(JSON.parse(text), null, 2); }
      catch { pre.textContent = text; }
    } else {
      pre.textContent = text;
    }
  } catch (e) {
    panel.querySelector('pre').textContent = t('file.previewError', { message: e.message });
  }
}

// ── Directory Browser ─────────────────────────────────────────────────────────
const dirModal       = $('dir-modal');
const dirList        = $('dir-list');
const dirCurrentPath = $('dir-current-path');
let   _dirParent     = null;   // track parent path for "Up" button

$('btn-browse-model').addEventListener('click', () => {
  // start from current value if set, otherwise let server pick default
  const cur = val('vllm-model-path').trim();
  openDirBrowser(cur || null);
});
$('btn-close-dir-modal').addEventListener('click', () => { dirModal.hidden = true; });
dirModal.addEventListener('click', e => { if (e.target === dirModal) dirModal.hidden = true; });

$('btn-dir-up').addEventListener('click', () => {
  if (_dirParent) openDirBrowser(_dirParent);
});

$('btn-dir-select').addEventListener('click', () => {
  const cur = dirCurrentPath.textContent;
  if (cur) setValue('vllm-model-path', cur);
  dirModal.hidden = true;
});

async function openDirBrowser(path) {
  dirModal.hidden = false;
  dirList.innerHTML = '<div class="dir-empty">' + t('results.loading') + '</div>';
  try {
    const url = '/api/browse' + (path ? '?path=' + encodeURIComponent(path) : '');
    const res  = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    dirCurrentPath.textContent = data.current;
    _dirParent = data.parent;
    $('btn-dir-up').disabled = !data.parent;

    dirList.innerHTML = '';
    if (!data.entries.length) {
      dirList.innerHTML = '<div class="dir-empty">' + t('modal.noDirs') + '</div>';
      return;
    }
    data.entries.forEach(entry => {
      const item = document.createElement('div');
      item.className = 'dir-item';
      item.innerHTML = '<span class="dir-item-icon">📁</span><span class="dir-item-name">' + escHtml(entry.name) + '</span>';
      item.title = entry.path;
      item.addEventListener('click', () => openDirBrowser(entry.path));
      dirList.appendChild(item);
    });
  } catch (e) {
    dirList.innerHTML = '<div class="dir-empty" style="color:var(--danger)">' + escHtml(e.message) + '</div>';
  }
}

// ── Language toggle ───────────────────────────────────────────────────────────
$('btn-lang').addEventListener('click', () => {
  const next = getLang() === 'zh' ? 'en' : 'zh';
  setLang(next);
});

// ── Apply i18n to HTML elements with data-i18n attribute ──────────────────────
function applyI18nToDOM() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    if (!key) return;
    el.textContent = t(key);
  });
  // Update document title
  document.title = t('title');
  // Update html lang attr
  document.documentElement.lang = getLang();
  // Update step bar label
  const stepLabel = $('step-bar-label-text');
  if (stepLabel) stepLabel.textContent = t('step.barLabel');
  // Update hint
  const hintEl = document.querySelector('.step-hint');
  if (hintEl) hintEl.innerHTML = t('step.hint').replace(/·/g, '&nbsp;·&nbsp;');
  // Update "Start at" / "Stop after"
  const startLabel = selStartAt?.previousElementSibling;
  if (startLabel) startLabel.textContent = t('step.startAt');
  const stopLabel = selStopAfter?.previousElementSibling;
  if (stopLabel) stopLabel.textContent = t('step.stopAfter');
}

// ── Init ──────────────────────────────────────────────────────────────────────
(function init() {
  applyI18nToDOM();
  buildStepBar();
  buildStepSelects();
  updateStepHighlights();
  loadDefaultConfig();

  // Also update button text that has icon + text
  updateButtonLabels();

  // placeholder log
  appendLog(t('run.status.ready'), 'log-empty');
})();

function updateButtonLabels() {
  // "▶ Start Run" / "■ Cancel" buttons need text+icon preserved
  const btnRunSpan = btnRun.querySelector('span');
  if (btnRunSpan) btnRunSpan.textContent = _lang === 'zh' ? '开始运行' : 'Start Run';
  const btnCancelSpan = btnCancel.querySelector('span');
  if (btnCancelSpan) btnCancelSpan.textContent = _lang === 'zh' ? '取消' : 'Cancel';
}
