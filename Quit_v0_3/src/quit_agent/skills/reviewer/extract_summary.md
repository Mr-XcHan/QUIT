# Skill: Extract Research Summary (EXTRACT)

## Description
Extract a structured research summary from a completed paper for future reuse.
The summary is stored in `extracts/<project_id>.json` as a rolling queue (max 5 entries).
Future IDEATE stages read this file to avoid repeating prior directions and to build on existing results.

## Input Artifacts
```
paper_gene/main.tex   — final paper source
PaperReview.json      — LLM review (strengths, weaknesses, scores)
BuildSpec.json        — research plan (target_task, method_summary, metrics)
```

## Output Schema
Return exactly one JSON object:

```json
{
  "contribution": "<one sentence: what this paper contributes>",
  "method_novelty": "<what is technically new about the approach>",
  "key_results": "<main quantitative or qualitative findings>",
  "limitations": ["<limitation 1>", "<limitation 2>"],
  "future_directions": ["<concrete next step 1>", "<concrete next step 2>"]
}
```

## Extraction Policy
- `contribution`: one clear sentence summarising the main contribution, grounded in the paper abstract/introduction.
- `method_novelty`: what is technically new — algorithm, architecture, training procedure, etc.
- `key_results`: the most important experimental finding (metric values if available).
- `limitations`: 2–4 honest limitations drawn from the paper or the LLM review weaknesses.
- `future_directions`: 2–4 concrete, actionable next steps that a future run could pursue.

## Runtime Prompt Template
```text
You are extracting a structured research summary from a completed paper.

BuildSpec:
{{build_spec}}

LLM Review:
{{llm_review}}

Paper (main.tex excerpt):
{{paper_tex}}

Extract a research summary and return exactly one JSON object with no surrounding text:

{
  "contribution": "<one sentence>",
  "method_novelty": "<what is technically new>",
  "key_results": "<main findings>",
  "limitations": ["..."],
  "future_directions": ["..."]
}
```
