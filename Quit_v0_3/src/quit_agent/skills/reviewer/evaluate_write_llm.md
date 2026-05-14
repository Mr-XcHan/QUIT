# Skill: LLM Paper Review (WRITE_EVAL)

## Description
Review a generated research paper against an ICML-style review rubric.
This skill is called **after** rule-based WRITE_EVAL checks pass. Its output is
advisory: the `status` field in `PaperReview.json` is determined by rule-based
checks only. However, the `llm_review` block (especially `weaknesses`) is
persisted into `PaperReview.json` and automatically included in the next
WRITE attempt's feedback, so the writer can improve the paper on the next round.

## When To Use
- `paper_gene/main.tex` exists and rule-based WRITE_EVAL found no hard failures.
- A `BuildSpec.json` is available for context.

## Input Artifacts
```
paper_gene/main.tex   — full LaTeX source of the generated paper
BuildSpec.json        — target task, method summary, metrics, success criteria
WriteReport.json      — compile status and page count (optional context)
```

## Output Schema
Return exactly one JSON object (nothing else):

```json
{
  "soundness": 3,
  "presentation": 3,
  "contribution": 3,
  "novelty": 3,
  "overall": 5,
  "confidence": 3,
  "summary": "<2-3 sentence paper summary>",
  "strengths": "<bullet-point strengths>",
  "weaknesses": "<bullet-point weaknesses>",
  "questions": "<questions for the authors>",
  "improvement_hints": [
    "<concrete, actionable suggestion for the writer agent>",
    "<another suggestion>"
  ]
}
```

Score ranges (all integers):
- `soundness`: 1–4 (1=poor, 4=excellent)
- `presentation`: 1–4
- `contribution`: 1–4
- `novelty`: 1–4
- `overall`: 1–10 (mirrors ICML/NeurIPS rating scale)
- `confidence`: 1–5 (1=very uncertain, 5=expert)

`improvement_hints` must be **concrete and actionable** — things the writer
agent can execute in the next WRITE attempt (e.g. "add ablation table comparing
X vs Y", "clarify the computational complexity of the proposed method",
"move related work section earlier to motivate the contribution").

## Review Policy
Score against these ICML-style criteria:

**Soundness**
- Are claims supported by the experiment results shown?
- Is the methodology described clearly enough to be reproducible?
- Are baselines fair and comparisons reasonable?

**Presentation**
- Is the writing clear and well-organized?
- Are figures and tables properly labeled and discussed in the text?
- Does the abstract accurately reflect the paper's contributions?

**Contribution**
- Does the paper clearly state what it contributes over prior work?
- Are the results significant relative to the stated success criteria in BuildSpec?

**Novelty**
- Is the proposed approach clearly distinguished from related work?
- Is the novelty claim in the paper consistent with the `method_summary` in BuildSpec?

**Overall**
- 1–3: reject (fundamental flaws)
- 4–5: weak reject (major issues but potentially fixable)
- 6–7: weak accept (minor issues)
- 8–10: accept (strong paper)

## Runtime Prompt Template
```text
You are an ICML paper reviewer. Review the following generated research paper.

BuildSpec (target task and success criteria):
{{build_spec}}

Paper LaTeX source (paper_gene/main.tex):
{{paper_tex}}

WriteReport (compile metadata):
{{write_report}}

Score the paper on the ICML rubric and return exactly one JSON object with no
surrounding text, markdown, or explanation:

{
  "soundness": <1-4>,
  "presentation": <1-4>,
  "contribution": <1-4>,
  "novelty": <1-4>,
  "overall": <1-10>,
  "confidence": <1-5>,
  "summary": "<2-3 sentence summary of the paper>",
  "strengths": "<bullet list of the paper's strengths>",
  "weaknesses": "<bullet list of the paper's weaknesses>",
  "questions": "<clarification questions for the authors>",
  "improvement_hints": [
    "<concrete actionable suggestion 1>",
    "<concrete actionable suggestion 2>"
  ]
}

Be specific and constructive. improvement_hints must be things the writer agent
can execute in the very next revision (not vague advice like "improve clarity").
```
