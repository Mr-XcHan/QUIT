# Skill: Replan Research Brief After Workflow Failure

## Description
Regenerate a strict `ResearchBrief` JSON artifact after the previous planning output failed parsing/validation or downstream retrieval produced too little evidence.

This is a failure-recovery skill. It is not the normal first-pass planning prompt. The first PLAN call should use `plan_research_brief`; this skill is selected after `ResearchBriefValidationReport.json` or `RetrievalReport.json` reports `FAIL`.

## When To Use
Use this skill when all of the following are true:

- the workflow is in `PLAN` because `VALIDATE_BRIEF` or `RETRIEVE` failed
- `ResearchBrief.raw.json` exists from the previous attempt
- `ResearchBriefValidationReport.json` or `RetrievalReport.json` exists and has status `FAIL`
- repair was attempted, parsing failed, or retrieval selected too few papers

Do not use this skill for the first planning attempt.

## Input Artifacts
The caller must provide:

```text
original user request
ResearchBrief.raw.json
ResearchBriefValidationReport.json
RetrievalReport.json, when retrieval failed
search budget values
build budget values
```

Runtime prompt variables:

```text
{{user_request}}
{{previous_raw_response}}
{{validation_report}}
{{max_queries}}
{{max_papers_screened}}
{{max_papers_selected}}
{{max_repo_checked}}
{{stop_if_no_new_signal_rounds}}
{{max_code_iterations}}
{{max_experiments}}
{{max_review_revisions}}
```

The skill must use the provided failure report as the source of truth for what was wrong. Do not rely on chat history.

## Output Artifact
Produce a replacement raw JSON candidate to be written as:

```text
ResearchBrief.raw.json
```

The downstream validator will parse, validate, repair if possible, and write:

```text
ResearchBrief.json
ResearchBriefValidationReport.json
```

## Output Rules
- Respond with exactly one JSON object.
- Do not wrap the JSON in Markdown fences.
- Do not include explanatory text before or after the JSON.
- Do not include a thinking process or analysis.
- Start the response with `{` and end it with `}`.
- Fix every validation error shown in the report.
- Preserve the original user intent unless it caused a schema violation.
- Use the provided budget values exactly.
- Copy fallback_policy exactly as shown in the runtime JSON shape.

## Failure Mode
If the previous raw response was not JSON, ignore its structure and rebuild the object from the original user request plus the validation report.

If the previous raw response was close to valid, preserve useful field content while correcting invalid types, missing required fields, and schema violations.

Repeated failure is handled by the orchestrator with a bounded retry limit.

## Runtime Prompt Template
Use the following prompt body when invoking an LLM:

```text
You are repairing a failed planning artifact for an artifact-driven research workflow. You have deep domain expertise across machine learning and related fields.

The previous ResearchBrief candidate failed validation or downstream retrieval returned too little usable evidence. Generate a replacement ResearchBrief JSON object that fixes the failure.

Original user request:
{{user_request}}

Previous raw response:
{{previous_raw_response}}

Failure report:
{{validation_report}}

Use your domain knowledge to enrich the replacement brief beyond the raw user text:
- Broaden or sharpen the topic to improve paper retrieval coverage.
- Add missing subfields, techniques, and evaluation protocols to the domain list.
- Broaden only with directly relevant neighboring subfields; do not turn a narrow request into a generic survey.
- Write 2-3 concise `search_keywords` (2-4 words each) that capture the most important technical concepts for paper retrieval.
- Tighten constraints and acceptance_criteria to reflect research standards in this area.

Fix every validation error or retrieval failure described in the failure report.
Preserve the original user intent unless it caused a schema violation or retrieval dead end.
Use the budget values exactly as provided in the JSON shape. Do not increase or decrease them.
Copy fallback_policy exactly as shown; it is a workflow routing contract, not a research-content field.

Return exactly one valid JSON object and nothing else.
Do not include a thinking process, analysis, explanation, markdown, or code fences.
Start your response with `{` and end it with `}`.

The JSON object must include:
- topic
- objective
- search_keywords
- domain
- constraints
- deliverable
- search_budget
- build_budget
- red_lines
- acceptance_criteria
- fallback_policy

Use this exact shape:

{
  "topic": "<concise research topic, 5-15 words>",
  "objective": "<preserve the user's main objective, quality requirements, and requested output emphasis>",
  "search_keywords": ["<short keyword phrase for paper search, 2-4 words>", "<2nd keyword>", "<3rd keyword>"],
  "domain": ["<domain or subfield>", "..."],
  "constraints": ["<hard constraint the research must respect>", "..."],
  "deliverable": ["<expected workflow output>", "..."],
  "search_budget": {
    "max_queries": {{max_queries}},
    "max_papers_screened": {{max_papers_screened}},
    "max_papers_selected": {{max_papers_selected}},
    "max_repo_checked": {{max_repo_checked}},
    "stop_if_no_new_signal_rounds": {{stop_if_no_new_signal_rounds}}
  },
  "build_budget": {
    "max_code_iterations": {{max_code_iterations}},
    "max_experiments": {{max_experiments}},
    "max_review_revisions": {{max_review_revisions}}
  },
  "red_lines": ["<claim or approach that must not appear in accepted ideas>", "..."],
  "acceptance_criteria": ["<minimum condition an idea must satisfy>", "..."],
  "fallback_policy": {
    "supervise": "emit artifact and stop after repeated failure",
    "code_fail": "return to BUILD_SPEC",
    "write_fail": "return to WRITE"
  }
}
```
