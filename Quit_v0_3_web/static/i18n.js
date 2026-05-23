'use strict';

const I18N = {
  zh: {
    title: 'QUIT Agent — 人机协同研究自动化平台',

    // Steps
    'step.PLAN': '制定计划',
    'step.VALIDATE_BRIEF': '验证简报',
    'step.RETRIEVE': '检索文献',
    'step.READ': '阅读论文',
    'step.IDEATE': '创意生成',
    'step.IDEA_EVAL': '创意评估',
    'step.BUILD_SPEC': '构建规格',
    'step.CODE': '编写代码',
    'step.CODE_EVAL': '代码评估',
    'step.WRITE': '撰写论文',
    'step.WRITE_EVAL': '论文评估',
    'step.EXTRACT': '提取',
    'step.tooltip': '{id}\n点击 = 在此停止\nCtrl+点击 = 从此开始',

    // Header
    'header.badge': 'v0.3',
    'btn.runs': '📂 运行记录',
    'btn.reset': '↺ 重置',

    // Step bar
    'step.barLabel': '流水线',
    'step.hint': '点击 = 在此停止 · Ctrl+点击 = 从此开始',
    'step.startAt': '开始于',
    'step.stopAfter': '终止于',

    // Config panel
    'config.title': '配置',
    'config.project': '项目',
    'config.llm': 'LLM 提供商',
    'config.runtime': '运行时',
    'config.searchBudget': '搜索预算',
    'config.runBudget': '运行预算',
    'config.retrieval': '检索',
    'config.agentTokens': 'Agent 令牌上限',

    // Project
    'field.projectId': '项目 ID',
    'field.topic': '研究主题',
    'field.domain': '研究领域',
    'field.objective': '研究目标',

    // LLM
    'field.model': '模型',
    'field.baseUrl': 'Base URL',
    'field.apiKey': 'API Key',
    'field.temperature': '温度系数',
    'field.modelPath': '模型路径',
    'field.servedName': '服务模型名称',
    'field.host': '主机',
    'field.port': '端口',
    'field.tensorParallel': '张量并行数',
    'field.gpuMargin': 'GPU 安全余量',
    'field.startupTimeout': '启动超时 (秒)',
    'field.gpuUtil': 'GPU 显存利用率',
    'field.keepAlive': '运行结束后保持服务存活',
    'btn.toggleKey': '显示/隐藏',
    'btn.browseModel': '浏览目录',

    // Runtime
    'field.maxSteps': '最大步数',

    // Search Budget
    'field.maxQueries': '最大查询数',
    'field.maxScreened': '最大筛选数',
    'field.maxSelected': '最大选择数',
    'field.maxRepos': '最大仓库数',

    // Run Budget
    'field.minTrainEpochs': '最小训练轮数',
    'field.maxTrainEpochs': '最大训练轮数',
    'field.minEvalEpochs': '最小评估轮数',
    'field.maxEvalEpochs': '最大评估轮数',

    // Retrieval
    'field.sources': '来源',
    'field.perSource': '每源结果数',
    'field.maxDownloads': '最大下载数',
    'field.readMode': '读取模式',
    'field.downloadPdfs': '下载 PDF',
    'source.local': '本地',
    'source.arxiv': 'arXiv',
    'source.openreview': 'OpenReview',
    'readMode.directPdf': 'direct_pdf',
    'readMode.localText': 'local_text',

    // Agents
    'agent.planner': '规划器',
    'agent.research': '研究员',
    'agent.reviewer': '审核员',
    'agent.builder': '构建器',

    // Run panel
    'run.id': '运行 ID',
    'run.idHint': '（留空自动生成）',
    'run.status.ready': '就绪。配置完毕后点击 ▶ 开始运行。',
    'run.status.running': '⟳ 运行中…',
    'run.status.done': '✓ 运行完成。',
    'run.status.error': '✗ 运行出错。',
    'run.starting': '开始运行：{startAt} → {stopAfter}  max_steps={maxSteps}',

    // Tabs
    'tab.log': '输出日志',
    'tab.results': '📁 结果',
    'tab.log.header': '输出',

    // Results
    'results.noRun': '未选择运行记录',
    'results.selectHint': '选择一个运行记录查看结果。',
    'results.loading': '加载中…',
    'results.empty': '未找到运行记录。',
    'results.preview': '📄\n点击文件预览',
    'results.pick': '📂',

    // File icons / types
    'file.previewOpen': '↗ 打开',
    'file.previewDownload': '↓ 下载',
    'file.previewLoading': '加载中…',
    'file.previewError': '加载文件出错：{message}',

    // Modals
    'modal.runs': '已有运行记录',
    'modal.selectDir': '选择模型目录',
    'modal.loading': '加载中…',
    'modal.noRuns': '未找到运行记录。',
    'modal.noDirs': '未找到子目录。',
    'modal.failedLoad': '加载运行记录失败。',
    'modal.use': '使用 →',
    'modal.close': '✕',
    'modal.selectFolder': '✓ 选择此目录',
    'modal.up': '↑ 上级目录',

    // Log
    'log.cleared': '（日志已清除）',
    'log.cancelled': '[已取消] 用户取消了运行。',
    'log.sseLost': '[WARN] SSE 连接丢失，正在重连…',
    'log.sseFailed': '[ERROR] SSE 连接多次重试失败。后端可能仍在运行。',
    'log.clear': '清除',
    'log.auto': '↓ 自动',

    // Running indicator
    'indicator.running': '● 运行中',
  },

  en: {
    title: 'QUIT Agent',

    // Steps
    'step.PLAN': 'Plan',
    'step.VALIDATE_BRIEF': 'Validate',
    'step.RETRIEVE': 'Retrieve',
    'step.READ': 'Read',
    'step.IDEATE': 'Ideate',
    'step.IDEA_EVAL': 'Idea Eval',
    'step.BUILD_SPEC': 'Build Spec',
    'step.CODE': 'Code',
    'step.CODE_EVAL': 'Code Eval',
    'step.WRITE': 'Write',
    'step.WRITE_EVAL': 'Write Eval',
    'step.EXTRACT': 'Extract',
    'step.tooltip': '{id}\nClick = set stop after\nCtrl+click = set start at',

    // Header
    'header.badge': 'v0.3',
    'btn.runs': '📂 Runs',
    'btn.reset': '↺ Reset',

    // Step bar
    'step.barLabel': 'Pipeline',
    'step.hint': 'Click = stop after · Ctrl+click = start at',
    'step.startAt': 'Start at',
    'step.stopAfter': 'Stop after',

    // Config panel
    'config.title': 'Configuration',
    'config.project': 'Project',
    'config.llm': 'LLM Provider',
    'config.runtime': 'Runtime',
    'config.searchBudget': 'Search Budget',
    'config.runBudget': 'Run Budget',
    'config.retrieval': 'Retrieval',
    'config.agentTokens': 'Agent Tokens',

    // Project
    'field.projectId': 'Project ID',
    'field.topic': 'Topic',
    'field.domain': 'Domain',
    'field.objective': 'Objective',

    // LLM
    'field.model': 'Model',
    'field.baseUrl': 'Base URL',
    'field.apiKey': 'API Key',
    'field.temperature': 'Temperature',
    'field.modelPath': 'Model Path',
    'field.servedName': 'Served Model Name',
    'field.host': 'Host',
    'field.port': 'Port',
    'field.tensorParallel': 'Tensor Parallel',
    'field.gpuMargin': 'GPU Safety Margin',
    'field.startupTimeout': 'Startup Timeout (s)',
    'field.gpuUtil': 'GPU Mem Utilization',
    'field.keepAlive': 'Keep server alive after run',
    'btn.toggleKey': 'Show/hide',
    'btn.browseModel': 'Browse',

    // Runtime
    'field.maxSteps': 'Max Steps',

    // Search Budget
    'field.maxQueries': 'Max Queries',
    'field.maxScreened': 'Max Screened',
    'field.maxSelected': 'Max Selected',
    'field.maxRepos': 'Max Repos',

    // Run Budget
    'field.minTrainEpochs': 'Min Train Epochs',
    'field.maxTrainEpochs': 'Max Train Epochs',
    'field.minEvalEpochs': 'Min Eval Epochs',
    'field.maxEvalEpochs': 'Max Eval Epochs',

    // Retrieval
    'field.sources': 'Sources',
    'field.perSource': 'Per Source',
    'field.maxDownloads': 'Max Downloads',
    'field.readMode': 'Read Mode',
    'field.downloadPdfs': 'Download PDFs',
    'source.local': 'Local',
    'source.arxiv': 'arXiv',
    'source.openreview': 'OpenReview',
    'readMode.directPdf': 'direct_pdf',
    'readMode.localText': 'local_text',

    // Agents
    'agent.planner': 'Planner',
    'agent.research': 'Research',
    'agent.reviewer': 'Reviewer',
    'agent.builder': 'Builder',

    // Run panel
    'run.id': 'Run ID',
    'run.idHint': '(leave blank to auto-generate)',
    'run.status.ready': 'Ready. Configure and press ▶ Start Run.',
    'run.status.running': '⟳ Running…',
    'run.status.done': '✓ Run completed.',
    'run.status.error': '✗ Run ended with errors.',
    'run.starting': 'Starting run: {startAt} → {stopAfter}  max_steps={maxSteps}',

    // Tabs
    'tab.log': 'Output Log',
    'tab.results': '📁 Results',
    'tab.log.header': 'Output',

    // Results
    'results.noRun': 'No run selected',
    'results.selectHint': 'Select a run to view results.',
    'results.loading': 'Loading…',
    'results.empty': 'No runs found.',
    'results.preview': '📄\nClick a file to preview',
    'results.pick': '📂',

    // File icons / types
    'file.previewOpen': '↗ Open',
    'file.previewDownload': '↓ Download',
    'file.previewLoading': 'Loading…',
    'file.previewError': 'Error loading file: {message}',

    // Modals
    'modal.runs': 'Existing Runs',
    'modal.selectDir': 'Select Model Directory',
    'modal.loading': 'Loading…',
    'modal.noRuns': 'No runs found.',
    'modal.noDirs': 'No subdirectories found.',
    'modal.failedLoad': 'Failed to load runs.',
    'modal.use': 'Use →',
    'modal.close': '✕',
    'modal.selectFolder': '✓ Select This Folder',
    'modal.up': '↑ Up',

    // Log
    'log.cleared': '(log cleared)',
    'log.cancelled': '[CANCELLED] Run cancelled by user.',
    'log.sseLost': '[WARN] SSE connection lost, reconnecting…',
    'log.sseFailed': '[ERROR] SSE connection lost after max retries. Backend may still be running.',
    'log.clear': 'Clear',
    'log.auto': '↓ Auto',

    // Running indicator
    'indicator.running': '● Running',
  }
};

let _lang = localStorage.getItem('quit-lang') || 'en';

function t(key, params) {
  const dict = I18N[_lang] || I18N['en'];
  let text = dict[key];
  if (text === undefined) {
    // fallback: try en
    text = (I18N['en'] || {})[key];
    if (text === undefined) return key;
  }
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      text = text.replaceAll('{' + k + '}', v);
    }
  }
  return text;
}

function setLang(lang) {
  _lang = lang;
  localStorage.setItem('quit-lang', lang);
  location.reload();
}

function getLang() {
  return _lang;
}
