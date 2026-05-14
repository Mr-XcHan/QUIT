# Skill: Evaluate Idea

## Description
Independently review candidate research ideas against the validated `ResearchBrief`.

This skill is used by the `IDEA_EVAL` workflow state. It must be isolated: judge only the supplied `ResearchBrief` and `IdeaLibrary`, not hidden reasoning, chat history, READ internals, or unrelated context.

## When To Use
Use this skill when:

- `ResearchBrief.json` exists.
- `IdeaLibrary.jsonl` exists.
- IDEATE produced candidate ideas and the workflow needs a PASS / REVISE / REJECT decision.

## Input Artifacts
Required:

```text
ResearchBrief.json
IdeaLibrary.jsonl
```

Use only:

```text
ResearchBrief.red_lines
ResearchBrief.acceptance_criteria
IdeaLibrary idea fields
```

## Output Artifact
Write:

```text
IdeaDecision.json
```

If the raw model output is malformed, write:

```text
IdeaDecision.raw.txt
```

and fall back to deterministic review logic.

## Output Schema
Return exactly one JSON object:

```json
{
  "idea_id": "",
  "decision": "PASS",
  "reason": "",
  "fallback_target": "BUILD_SPEC",
  "violations": [],
  "required_changes": [],
  "missing_evidence": []
}
```

Allowed `decision` values:

```text
PASS
REVISE
REJECT
```

Allowed `fallback_target` values:

```text
RETRIEVE
READ
IDEATE
BUILD_SPEC
STOP
```

## Decision Policy
Use this routing policy:

```text
PASS   -> fallback_target BUILD_SPEC
REVISE because the idea itself needs sharpening -> fallback_target IDEATE
REVISE because the evidence is incomplete or weak -> fallback_target READ
REVISE because the paper set/source quality is insufficient -> fallback_target RETRIEVE
REJECT because missing evidence -> fallback_target READ
REJECT because retrieval/source/paper quality is poor -> fallback_target RETRIEVE
REJECT otherwise -> fallback_target IDEATE
```

PASS only if:

- the idea has concrete `supporting_evidence_ids`
- the novelty claim is specific enough to become a build spec
- it does not violate red lines
- it plausibly satisfies at least one acceptance criterion

REVISE if:

- the idea is promising but vague
- expected gain is underspecified
- target task is too broad
- novelty claim needs sharper wording

REJECT if:

- evidence linkage is missing or insufficient
- the idea violates a red line
- the idea cannot plausibly satisfy acceptance criteria
- retrieval or paper quality is too weak to support ideation

## Runtime Prompt Template
```text
You are an isolated reviewer for an artifact-driven research workflow.

Review only the ResearchBrief and IdeaLibrary provided below.
Do not use chat history, hidden reasoning, READ internals, or any unrelated context.

ResearchBrief:
{{research_brief}}

IdeaLibrary:
{{idea_library}}

Select the strongest idea and produce one IdeaDecision JSON object.

Decision policy:
- PASS only if the idea is evidence-backed, specific, does not violate red_lines, and can plausibly satisfy acceptance_criteria.
- REVISE if the idea is promising but needs sharper novelty, target task, expected gain, or evidence linkage.
- REJECT if evidence is missing, retrieval/source quality is poor, or the idea violates red_lines.

Fallback target rules:
- PASS -> BUILD_SPEC
- REVISE due to idea wording, target task, expected gain, or positioning -> IDEATE
- REVISE due to incomplete/missing evidence links -> READ
- REVISE due to retrieval/source/paper quality -> RETRIEVE
- REJECT due to missing evidence -> READ
- REJECT due to retrieval/source/paper quality -> RETRIEVE
- other REJECT -> IDEATE

Return exactly one valid JSON object and nothing else:
{
  "idea_id": "<selected idea id>",
  "decision": "PASS|REVISE|REJECT",
  "reason": "<concise reason>",
  "fallback_target": "RETRIEVE|READ|IDEATE|BUILD_SPEC|STOP",
  "violations": [],
  "required_changes": [],
  "missing_evidence": []
}
```
