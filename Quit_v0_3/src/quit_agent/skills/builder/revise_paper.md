# Skill: Revise Paper (WRITE_REVISE)

## Description
Revise an existing LaTeX paper draft based on LLM reviewer feedback.
The model receives the current `main.tex` and the `improvement_hints` from the
soft LLM review, then decides autonomously which parts to change and how.
It may make targeted edits, rewrite specific sections, or restructure arguments —
whatever it judges necessary to address the hints. The output is a complete,
valid `main.tex`.

## When To Use
- `WRITE_EVAL` hard check passed.
- `WRITE_LLM_EVAL` produced `improvement_hints` in `PaperReview.json`.
- A `paper_gene/main.tex` draft already exists.

## Input Artifacts
```
paper_gene/main.tex     — current LaTeX draft
PaperReview.json        — contains llm_review.improvement_hints and llm_review.weaknesses
BuildSpec.json          — target task, metrics, success criteria (for grounding)
```

## Output Artifact
Overwrite:
```
paper_gene/main.tex
```
Also write:
```
WriteReport.json        — same schema as WRITE stage report
```

## Revision Policy
You are an expert academic writer revising a research paper.

Given the current draft and the reviewer's improvement_hints:
1. **Read the hints carefully.** Each hint is a concrete, actionable suggestion.
2. **Decide what to change.** You may:
   - Make targeted edits to specific sentences or paragraphs.
   - Rewrite a section that has a structural problem.
   - Add missing content (ablation table, proof, clarification).
   - Restructure arguments for clarity.
3. **Do not change what is already good.** Preserve sections that are not
   implicated by the hints. Do not introduce new placeholders or regressions.
4. **Keep all hard constraints:**
   - Abstract ≥ 150 words
   - Introduction ≥ 700 words
   - Related Work ≥ 350 words
   - Method ≥ 1600 words
   - Experiments ≥ 1200 words
   - Conclusion ≥ 200 words
   - Main paper ≥ configured page threshold (excluding appendix)
   - All figures/tables from the original must still be present unless a hint
     explicitly calls for their removal.
   - Use only citation keys already in `references.bib`.

Return ONLY valid LaTeX for `main.tex`. Do not use markdown fences.
The first non-whitespace characters must be `\documentclass{article}`.

## Runtime Prompt Template
```text
You are an expert academic writer revising a research paper based on peer review feedback.

BuildSpec (ground truth for task, metrics, success criteria):
{{build_spec}}

LLM Reviewer feedback:
Weaknesses:
{{weaknesses}}

Improvement hints (address these in your revision):
{{improvement_hints}}

Current main.tex draft:
{{current_tex}}

Revise the paper to address the improvement hints above. You decide which parts
to change and how — targeted edits, section rewrites, or additions as needed.
Do not regress sections that are not implicated by the hints.
Keep all hard length constraints and all existing figures/tables.
Use only citation keys already present in the draft.

Return ONLY valid LaTeX for main.tex. No markdown fences.
The first non-whitespace characters must be \documentclass{article}.
```
