# Skill: Plan Research Brief

## Description
Generate the first-pass `ResearchBrief` JSON artifact from the original user request for an artifact-driven research workflow.

This is the normal first planning skill. It should be used before any validation, retrieval, reading, ideation, building, coding, or writing artifacts exist.

## When To Use
Use this skill when all of the following are true:

- the workflow is starting in `PLAN`
- there is no prior failed `ResearchBrief.raw.json` that needs repair
- the system needs a structured research brief to guide bounded retrieval, ideation, implementation, and writing

Do not use this skill for recovery after validation or retrieval failure. Use `replan_research_brief` for that path.

## Input
The caller provides:

```text
original user request
search budget values
build budget values
```

Runtime prompt variables:

```text
{{user_request}}
{{max_queries}}
{{max_papers_screened}}
{{max_papers_selected}}
{{max_repo_checked}}
{{stop_if_no_new_signal_rounds}}
{{max_code_iterations}}
{{max_experiments}}
{{max_review_revisions}}
```

## Output Artifact
Produce a raw JSON candidate to be written as:

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

## Runtime Prompt Template
```text
You are a research planning assistant with deep expertise across machine learning, computer vision, NLP, and related fields. Your task is to produce a structured ResearchBrief JSON object that will guide an automated artifact-driven research workflow.

The workflow is local-first and does not rely on long chat history. It is driven by plain-text and JSON artifacts, explicit schemas, validators, repair logic, and fallback transitions.

User request:
{{user_request}}

Before writing the JSON, apply your domain knowledge to enrich the request:
- Expand the topic into a precise technical problem statement, naming key challenges, paradigms, or open questions in this area.
- Extend the domain list with relevant subfields, neighboring research areas, and key techniques (e.g. specific architectures, training strategies, evaluation protocols) that a literature search should cover.
- Broaden only with directly relevant neighboring subfields; do not turn a narrow request into a generic survey.
- Write 2-3 concise `search_keywords` (2-4 words each) that capture the most important technical concepts for paper retrieval — these will be used directly as search queries, so keep them short and precise.
- Derive concrete, verifiable constraints from the objective (e.g. offline data assumption, no environment interaction, specific benchmark suites).
- Set acceptance_criteria that reflect meaningful research standards in this field (e.g. outperforms a named baseline, reports confidence intervals, evaluates on standard benchmarks).
- Identify red_lines that would make an idea scientifically unsound or out of scope.

Use this enriched understanding — not just the raw user text — to fill every field of the ResearchBrief.
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
