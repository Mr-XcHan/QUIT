<p align="center">
  <img src="Figure/logo_bg.png" alt="QUIT Agent" width="200"/>
</p>

<p align="center">
  <strong style="font-size:1.6em;">QUIT：面向 AI 科研自动化的人机协同平台</strong><br/>
  <strong><em>Query（检索）&nbsp;·&nbsp; Understand（理解）&nbsp;·&nbsp; Implement（实现）&nbsp;·&nbsp; Tell（撰写）</em></strong><br/>
  <sub>✦ &nbsp; 告别旧式科研方式 &nbsp; ✦</sub>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white"/>
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-orange"/>
</p>

<p align="center">
  中文 | <a href="README.md">English</a>
</p>

---

QUIT 是一个人机协同的科研助手——不是黑盒，而是一条透明的流水线，研究者在每个环节都保持完全的控制权。
其**以制品（artifact）为驱动**的设计消除了长上下文依赖，避免了冗余的 token 消耗。

QUIT 同时完整支持**端到端模式**——只需设定研究主题，流水线即可无干预地自动运行至完成。
使用 DeepSeek-V4-Pro 的端到端成本：每篇论文约 **¥10（约 $1.5）**。

| 阶段 | 功能描述 |
|:---:|---|
| 🔍 **Query（检索）** | 搜索论文、代码仓库和本地文献 |
| 💡 **Understand（理解）** | 提取证据卡、聚类洞见、生成创意 |
| 🔧 **Implement（实现）** | 将选定创意转化为代码、运行实验、审查结果 |
| 📝 **Tell（撰写）** | 根据真实输出起草并审阅论文 |

---

## 🏗️ 系统架构

<p align="center">
  <img src="Figure/Agents.png" alt="QUIT Agent Architecture" width="780"/>
</p>

四个专职 Agent 由中央**编排器 / 状态机**协调：

- 🗺️ **PlannerAgent** — 将用户的研究主题转化为经过验证的 `ResearchBrief`
- 🔬 **ResearchAgent** — 检索论文、提取证据、综合创意
- 🏗️ **BuilderAgent** — 生成实验代码、运行实验、撰写论文
- 🔍 **ReviewerAgent** — 审查创意、代码质量和论文草稿

所有协调通过磁盘上命名的制品文件完成——Agent 之间不共享跨调用的会话记忆。这使整个流水线**可追溯、可复现、可恢复**。

> 🙋 **人机协同控制点：** 研究者可在任意状态后暂停，检查制品文件、编辑中间文件（证据卡、BuildSpec、生成代码、实验结果），并从选定状态继续运行。

---

## 🔄 工作流程

<p align="center">
  <img src="Figure/workflow.png" alt="QUIT Workflow" width="780"/>
</p>

---

## ⚙️ 安装

**环境要求：** Python 3.11+、Git、LaTeX（`texlive` + `latexmk`）

```bash
git clone https://github.com/Mr-XcHan/QUIT.git
cd QUIT
bash setup.sh
```

脚本会在仓库根目录创建 `.venv` 虚拟环境，并安装 Agent 核心和 Web UI。如需本地 LLM 推理（torch + transformers）：

```bash
bash setup.sh --with-local
```

缺少的系统工具（`latexmk`、`bibtex`）将在 setup 结束时以安装提示的形式报告。

---

## 🛠️ 配置

主配置文件为 `Quit_v0_3/config.json`。**默认参数开箱即用**——通常只需填写研究主题和 LLM 凭据。该文件中的所有参数均为默认值，也可在每次运行前直接在 Web UI 中覆盖。

```json
{
  "project": {
    "topic": "Flow Matching for Offline Reinforcement Learning"
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-5.5",
    "api_key_env": "OPENAI_API_KEY"
  }
}
```

将 API Key 保存在 `config.json` 同目录的 `.env` 文件中（切勿提交到版本库）：

```
OPENAI_API_KEY=sk-...
```

支持的 Provider：`anthropic`、`openai`、`deepseek`、`local-vllm`，以及其他 OpenAI 兼容端点。

<details>
<summary>⚙️ 其他可调整的配置项</summary>

| 字段 | 默认值 | 说明 |
|---|---|---|
| `runtime.stop_after` | `null` | 在指定状态后停止（如 `"CODE_EVAL"`） |
| `run_budget.experiment_timeout_seconds` | `3600` | 生成实验的最大运行时间（秒） |
| `retrieval.sources` | `["arxiv"]` | 论文检索来源 |
| `write.expected_main_pages` | `8` | 论文目标页数 |

</details>

---

## 🖥️ 使用方式 — Web UI（推荐）

<p align="center">
  <img src="Figure/interface_main.png" alt="Web UI Screenshot" width="780"/>
</p>

激活虚拟环境并启动服务器：

```bash
source .venv/bin/activate
cd Quit_v0_3_web
python server.py --port 7862
```

然后在浏览器中打开 **http://localhost:7862**。

**▶️ 快速开始：** 填写研究主题，点击 **Start Run**，Agent 将自动逐状态运行，直至 **Stop after** 指定的状态。详细的分步操作指南请参见 [GUIDE.md](GUIDE.md)。

**🙋 人机协同干预：** 由于所有输出均为文件，你可以随时：

- 👁️ 在 **Artifacts** 面板中**查看**制品（证据卡、BuildSpec、代码、结果、论文）
- ⏹️ 在任意状态**暂停**，审查输出后继续运行
- ✏️ **编辑**中间文件，并从选定状态**恢复**——无需重跑之前的阶段
- 🔁 修正后**重跑**指定状态（如编辑 BuildSpec 后重跑 `CODE`）

**📂 浏览历史运行：** 所有运行保存在 `Quit_v0_3/runs/<run_id>/` 下。**Runs** 面板列出每次历史运行及其完整的制品记录、提示词和 LLM 响应。

---

## 📦 关键制品

每次运行在 `runs/<run_id>/` 下生成完整的制品记录：

```
ResearchBrief.json          ← 经过验证的研究计划
EvidenceCards.jsonl         ← 结构化论文证据
IdeaLibrary.jsonl           ← 带证据链接的候选创意
BuildSpec.json              ← 实验与论文合约
code/src/*.py               ← 生成的实验代码
results/metrics.json        ← 实验结果
results/results_table.csv   ← 各方法对比表格
CodePerformanceEval.json    ← LLM 对方法 vs. 基准线的评判
paper_gene/main.tex         ← 生成的 LaTeX 论文
paper_gene/main.pdf         ← 编译后的 PDF 论文
run_trace.json              ← 完整状态转换日志
```

所有提示词和原始 LLM 响应均保存在 `llm/` 目录下，以保证完全可复现。

---

## 📄 生成论文展示

QUIT Agent 在不同研究领域端到端生成的论文：

<table width="100%">
  <tr>
    <td align="center" valign="top" width="33%" style="padding:12px">
      <a href="Demos/offline_rl_rectified_flow.pdf">
        <img src="Demos/offline_rl_rectified_flow_thumb.png" width="150"/>
      </a>
      <br/><br/>
      <b>离线强化学习</b><br/>
      <sub><em>Trust-Region Rectified Flow Offline Actor for Low-Latency Generative Policy Learning</em></sub>
    </td>
    <td align="center" valign="top" width="33%" style="padding:12px">
      <a href="Demos/robot_v2g_dispatch.pdf">
        <img src="Demos/robot_v2g_dispatch_thumb.png" width="150"/>
      </a>
      <br/><br/>
      <b>机器人</b><br/>
      <sub><em>Online Pricing and Energy-Aware Dispatch for Mobile Charging Robot-Mediated V2G Services</em></sub>
    </td>
    <td align="center" valign="top" width="33%" style="padding:12px">
      <a href="Demos/3d_pattern_synthesis.pdf">
        <img src="Demos/3d_pattern_synthesis_thumb.png" width="150"/>
      </a>
      <br/><br/>
      <b>3D 生成</b><br/>
      <sub><em>Decoupled Macro–Micro Generation for Editable Large-Scale 3D Surface Patterns</em></sub>
    </td>
  </tr>
</table>

---

**我们正在招募测试者！** 🚀 用你自己的研究主题试用这条流水线，看它能走多远。
全自动运行的结果我们很欢迎，但我们尤其期待**人机协同的成果**——研究者主动介入引导 Agent、编辑中间制品或调整方向的论文。这类成果往往是最有趣的结果。

如果你希望将自己的输出展示在这里，欢迎提 issue 或 Pull Request。🙌

---

## 📬 联系方式

**韩昕辰（Xinchen）** — isxinchen.han@gmail.com

欢迎随时联系，提问、反馈或探讨合作。

---

## 📜 许可证

Apache 2.0 — 可自由使用和二次开发（包括商业用途），需保留署名并注明修改。详见 [LICENSE](LICENSE)。
