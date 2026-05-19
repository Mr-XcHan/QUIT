<p align="center">
  <img src="Figure/logo_bg.png" alt="QUIT Agent" width="140"/>
</p>

<p align="center">
  <strong style="font-size:1.4em;">QUIT Agent — Web UI 使用指南</strong><br/>
  <em>手把手带你跑完第一条科研 pipeline</em>
</p>

<p align="center">
  中文 | <a href="GUIDE.md">English</a>
</p>

---

## 📋 目录

1. [启动服务器](#1-启动服务器)
2. [界面总览](#2-界面总览)
3. [第一步 — 配置项目](#3-第一步--配置项目)
4. [第二步 — 配置 LLM Provider](#4-第二步--配置-llm-provider)
5. [第三步 — 选择 Pipeline 范围](#5-第三步--选择-pipeline-范围)
6. [第四步 — 启动运行](#6-第四步--启动运行)
7. [第五步 — 监控进度](#7-第五步--监控进度)
8. [Human-in-the-Loop 干预](#8-human-in-the-loop-干预)
9. [浏览历史运行](#9-浏览历史运行)

---

## 1. 启动服务器

```bash
source .venv/bin/activate
cd Quit_v0_3_web
python server.py --port 7862
```

然后在浏览器中打开 **http://localhost:7862**，即可看到 QUIT Agent 界面。

---

## 2. 界面总览

<p align="center">
  <img src="Figure/interface_main.png" alt="Web UI Overview" width="780"/>
</p>

界面分为三个主要区域：

| 区域 | 位置 | 用途 |
|---|---|---|
| 🗂️ **Pipeline bar** | 顶部 | 展示所有 workflow 状态；点击可设置暂停/起始节点 |
| ⚙️ **Configuration panel** | 左侧 | 填写项目信息和 LLM 凭据 |
| 🖥️ **Run panel** | 中/右 | 启动运行、查看实时输出、浏览结果 |

---

## 3. 第一步 — 配置项目

在左侧 **CONFIGURATION** 面板的 **PROJECT** 区域填写以下内容：

- 📁 **Project ID** — 本次研究项目的简短名称（如 `offline-rl`）。留空则自动生成。
- 🔖 **Topic** — 研究问题或主题（如 `flow matching for offline reinforcement learning`）。
- 🏷️ **Domain** — 研究领域（如 `Offline Reinforcement Learning, Policy Improvement`）。
- 📝 **Objective** — 简短描述你希望 agent 实现的目标。写得越具体越好——这会影响整条 pipeline 的走向。

> 💡 **提示：** 一个写得好的 Objective（2–3 句，点明研究缺口和预期贡献）会显著提升后续生成的创意和代码质量。

---

## 4. 第二步 — 配置 LLM Provider

在左侧面板的 **LLM PROVIDER** 区域：

1. 🔘 **点击你的 provider** — 可选 `Anthropic`、`OpenAI`、`DeepSeek`、`Qwen`、`Moonshot`、`GLM`、`LMStudio` 或 `vLLM`。
2. 🤖 **Model** — 填写模型名称（如 `gpt-5.5`、`deepseek-v4-pro`）。
3. 🌐 **Base URL** — 主流 provider 已预填；自定义端点才需要修改。
4. 🔑 **API Key** — 粘贴你的 API Key。仅存储在浏览器 session 中，不会被记录到日志。
5. 🌡️ **Temperature** — 控制创意程度（默认 `0.70`）。越低越聚焦，越高越发散。

> 🔒 为避免每次重新填写 API Key，可将 `.env.example` 复制为 `Quit_v0_3/.env` 并填入你的 Key，agent 启动时会自动加载。

### 🆓 通过 vLLM 使用本地模型（完全免费）

如果你有 GPU 和本地模型（如 Qwen、LLaMA、Mistral），可以用 [vLLM](https://github.com/vllm-project/vllm) 作为本地 OpenAI 兼容服务器，**零 API 费用**运行完整的 QUIT pipeline。

**第一步 — 安装并启动 vLLM：**

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
  --model ../models/Mistral-Small-3.1-24B-Instruct-2503 \
  --served-model-name Mistral-24B \
  --tensor-parallel-size 2 \
  --port 8000
```

`--tensor-parallel-size` 根据你的 GPU 数量调整。单卡用户可省略或设为 `1`。

**第二步 — 在 QUIT 中配置使用本地模型：**

在 Web UI 的 **LLM PROVIDER** 区域填写：

| 字段 | 值 |
|---|---|
| Provider | `vLLM` |
| Model | `Mistral-24B`（需与 `--served-model-name` 一致） |
| Base URL | `http://localhost:8000/v1` |
| API Key | 任意非空字符串（如 `none`） |

> 💡 vLLM 暴露的是 OpenAI 兼容 API，无需改动任何代码——QUIT 与它通信的方式和云端 provider 完全相同，只是不用付费。

---

## 5. 第三步 — 选择 Pipeline 范围

<p align="center">

```
Plan → Validate → Retrieve → Read → Ideate → Idea Eval → Build Spec → Code → Code Eval → Write → Write Eval → Extract
```

</p>

顶部的 **Pipeline bar** 按顺序展示所有 workflow 状态。

- 🖱️ **点击某个状态** → 将其设为 **Stop after** 节点（运行到此处暂停）
- ⌨️ **Ctrl + 点击某个状态** → 将其设为 **Start at** 节点（从此处开始运行）
- 也可以直接使用 Pipeline bar 下方的 **Start at** 和 **Stop after** 下拉框进行精确控制

**常用预设：**

| 目标 | Start at | Stop after |
|---|---|---|
| 完整运行 | `PLAN` | `WRITE_EVAL` |
| 仅文献调研 | `PLAN` | `READ` |
| 生成创意，跳过写作 | `PLAN` | `IDEA_EVAL` |
| 从已有实验结果重跑写作 | `WRITE` | `WRITE_EVAL` |

---

## 6. 第四步 — 启动运行

1. *（可选）* 填写 **Run ID** — 留空则自动生成带时间戳的 ID。
2. ✅ 确认 Topic、Provider 和 Pipeline 范围无误。
3. 🟢 点击 **Start Run**。

按钮变灰并出现 **Running…** 提示，agent 已开始执行 pipeline。

> ⏹️ 随时可点击 **Cancel** 中止。已写入的所有 artifact 都会保留，之后可以继续。

---

## 7. 第五步 — 监控进度

右侧 **Output** 面板实时展示状态转换日志：

```
▸ Starting run: PLAN → WRITE_EVAL  max_steps=30
▸ [PIPELINE] step 0001 START  PLAN
▸ [PIPELINE] step 0001 END    PLAN  →  VALIDATE_BRIEF
▸ [PIPELINE] step 0002 START  VALIDATE_BRIEF
...
```

切换 Tab 可查看更多信息：

- 📋 **Output Log** — 包含状态转换和 agent 消息的完整 pipeline 日志
- 📊 **Results** — 运行完成后的最终指标、表格及生成论文的链接

---

## 8. Human-in-the-Loop 干预

所有 pipeline 输出均为 `Quit_v0_3/runs/<run_id>/` 下的普通文件，你可以随时介入：

| 操作 | 方式 | 适用场景 |
|---|---|---|
| 👁️ **查看** | 在编辑器中打开任意 artifact 文件 | 继续前检查 evidence card、创意或 BuildSpec |
| ✏️ **编辑** | 直接修改 artifact 文件 | 调整方向——如重写 BuildSpec 以修改实验设计 |
| ⏸️ **暂停** | 点击 **Cancel**，或设置 **Stop after** 节点 | 在耗时或高消耗阶段前停下来 |
| ▶️ **继续** | 将 **Start at** 设为下一个状态，点击 **Start Run** | 审查或编辑 artifact 后继续 |
| 🔁 **重跑某状态** | 将 **Start at = Stop after = <状态>**，点击 **Start Run** | 修正输入后重新执行单个状态 |

> 🙋 **示例工作流：** 运行到 `IDEA_EVAL`，阅读 `IdeaLibrary.jsonl`，删除较弱的创意，编辑最佳创意，然后从 `BUILD_SPEC` 继续。

---

## 9. 浏览历史运行

点击右上角 **Runs** 打开历史运行面板。每次历史运行都会列出：

- 🗂️ 其 **Run ID** 和时间戳
- 📄 所有 **artifact**（evidence card、BuildSpec、代码、结果、论文）
- 💬 `llm/` 目录下的完整 **prompt 和 LLM 响应**，用于复现

所有运行存储在 `Quit_v0_3/runs/<run_id>/` 下，随时可查看或重新启动。

**▶️ 从中间状态恢复历史运行：**

可从任意状态重新进入历史运行——已有的 artifact 会全部保留并复用：

1. 📋 从 Runs 面板（或 `runs/` 目录的文件夹名）复制 **Run ID**
2. 🔢 将其粘贴到主面板的 **Run ID** 字段
3. ⌨️ 将 **Start at** 设为你想恢复的状态（如 `WRITE`）
4. 🖱️ 设置 **Stop after** 为目标结束状态
5. 🟢 点击 **Start Run** — agent 从该状态开始，直接使用磁盘上已有的 artifact

> 💡 **示例：** 某次运行完成了 `CODE_EVAL` 但未到达 `WRITE`，只需输入其 Run ID，设置 `Start at = WRITE`，点击 **Start Run** 即可——之前的阶段无需重新计算。

---

<p align="center">
  <sub>返回 <a href="README.zh.md">README 中文版</a> · <a href="README.md">README English</a></sub>
</p>
