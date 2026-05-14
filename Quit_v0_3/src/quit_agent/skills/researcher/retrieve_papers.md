# Skill: Retrieve Papers

## Description
Retrieve a budgeted and relevance-ranked set of candidate papers from local and external sources.

This skill is used by the `RETRIEVE` workflow state. It does not create a separate `SCREEN` state; lightweight screening is part of this skill.

## When To Use
Use this skill when:

- `ResearchBrief.json` exists and has passed validation.
- The workflow needs `PaperCards.jsonl` for the `READ` state.
- A reviewer or evaluator rejects an idea because evidence is missing, retrieval quality is poor, or the candidate paper set is too weak.

This skill can be reached from:

```text
VALIDATE_BRIEF -> RETRIEVE
IDEA_EVAL -> RETRIEVE
READ -> RETRIEVE
```

## Input Artifacts
Required:

```text
ResearchBrief.json
```

Optional:

```text
paper_database/local_papers/
```

The local paper directory is user-managed. It may contain PDF files named approximately like:

```text
2019_ICML_x_BCQ.pdf
2024_iclr_zhang_sparseformer_offline_rl.pdf
```

## Output Artifacts
Required:

```text
PaperCards.jsonl
RepoCards.jsonl
RepoRetrievalReport.json
RetrievalReport.json
```

On failure:

```text
RetrievalFailure.json
```

If PDF download is enabled, external PDFs are saved under:

```text
paper_database/retrieved_papers/<run_id>/
```

## Sources
The configured source order is authoritative. Supported sources:

```text
local
arxiv
openreview
mock
```

Recommended priority:

```text
local -> arxiv -> openreview
```

`mock` should only be used for offline tests or fallback demos.

## Retrieval Policy
The skill must:

- generate search queries from topic, domain, constraints, and optionally LLM query planning
- query each configured source within `per_source_results`
- obey `ResearchBrief.search_budget`
- deduplicate papers by URL or title
- rank by lightweight relevance
- select papers with a hybrid policy: the top `relevance_selection_ratio` portion by relevance, then fill the remainder with source-aware diversity from the remaining relevant candidates
- preserve user-provided local PDFs as seed papers
- filter obviously malformed external records
- truncate to `max_papers_selected`
- stop when budget or no-new-signal conditions are met

## Budget Controls
Use `ResearchBrief.search_budget`:

```text
max_queries
max_papers_screened
max_papers_selected
max_repo_checked
stop_if_no_new_signal_rounds
```

Use retrieval config:

```text
sources
local_database_path
per_source_results
timeout_seconds
use_llm_query_planning
download_pdfs
max_downloads
relevance_selection_ratio
pdf_dir
```

`max_downloads` is a hard cap on PDF download attempts for selected external papers. It does not override `ResearchBrief.search_budget.max_papers_selected`; the selected paper count is still controlled by the brief.

`local_database_path` is the user-managed reference PDF folder and should not be mutated by retrieval. When `download_pdfs=true`, external PDFs are written under `runs/<run_id>/<pdf_dir>/`, for example `runs/20260428121550/paper_retrieve/`.

`relevance_selection_ratio` controls final selection. With `0.7` and `max_papers_selected=20`, the first 14 papers are the highest-scoring candidates and the remaining 6 are chosen from the rest using source-aware diversity.

`max_repo_checked` controls how many linked code repositories are extracted from selected papers. RETRIEVE records repo links only; it does not clone repositories. After IDEA_EVAL passes, BUILD_SPEC tries to clone the repos attached to the approved idea's supporting evidence, in relevance order. If no related repo can be cloned or inspected, CODE uses a generated standalone environment.

## Output Schema
Each record in `PaperCards.jsonl` must match `PaperCard`:

```json
{
  "paper_id": "",
  "title": "",
  "authors": [],
  "year": 2026,
  "venue": "",
  "abstract": "",
  "source": "local_pdf|ICLR|ICML|NeurIPS|arxiv|openreview",
  "paper_url": "",
  "pdf_url": "",
  "code_url": "",
  "query_source": "",
  "status": "found",
  "local_pdf_path": "",
  "retrieval_score": 0.0
}
```

Each record in `RepoCards.jsonl` must match `RepoCard`:

```json
{
  "repo_id": "",
  "repo_url": "",
  "source_paper_id": "",
  "source_title": "",
  "local_repo_path": "",
  "env_files": [],
  "language": "python",
  "framework": "",
  "status": "found|cloned|failed|inspected",
  "relevance_score": 0.0,
  "errors": []
}
```

## Failure Mode
If no papers are selected:

- write `RetrievalFailure.json`
- keep `RetrievalReport.json`
- route according to transition policy, usually back to `PLAN` or `RETRIEVE`

If fewer than 20% of `ResearchBrief.search_budget.max_papers_selected` are selected, treat retrieval as failed even if `PaperCards.jsonl` contains some papers. This prevents weak retrieval runs from advancing to `READ` with only one or two papers.

If one source fails due to timeout or API error, continue with remaining sources.

## Runtime Implementation
Current runtime implementation:

```text
ResearchAgent.retrieve()
tools/retrievers.py
tools/repo_tools.py
```

The skill is selected by workflow state, not by chat history.
