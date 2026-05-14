# Skill: Ideate From Evidence

## Description
Cluster structured `EvidenceCard` artifacts and generate a traceable candidate idea library.

This skill is used by the `IDEATE` workflow state. It does not start coding. It produces candidate research ideas that can be independently reviewed by `IDEA_EVAL`.

## When To Use
Use this skill when:

- `ResearchBrief.json` exists.
- `EvidenceCards.jsonl` exists and READ passed.
- The workflow needs `IdeaLibrary.jsonl`.

## Input Artifacts
Required:

```text
ResearchBrief.json
EvidenceCards.jsonl
```

Optional revision input:

```text
IdeaDecision.json
```

When `IdeaDecision.json` exists with `decision != PASS` and `fallback_target = IDEATE`,
`IDEATE` must use its `reason`, `violations`, and `required_changes` as explicit
revision instructions for the next idea generation attempt.

## Output Artifacts
Required:

```text
IdeaClusters.json
IdeaLibrary.jsonl
IdeaGenerationReport.json
```

Optional debug output:

```text
IdeaGeneration.raw.txt
```

## Runtime Pipeline
The runtime pipeline is:

```text
EvidenceCards.jsonl
  -> lightweight clustering by task/method/claim themes
  -> optional previous IDEA_EVAL feedback injection
  -> optional LLM idea generator
  -> rule-based fallback idea generator
  -> idea validation
  -> IdeaLibrary.jsonl
```

## Output Schema
Each record in `IdeaLibrary.jsonl` must match `IdeaCard`:

```json
{
  "idea_id": "",
  "target_task": "",
  "novelty_claim": "",
  "supporting_evidence_ids": [],
  "expected_gain": ""
}
```

## Validation Rules
Every idea must:

- include at least one `supporting_evidence_ids` entry
- reference only evidence IDs present in `EvidenceCards.jsonl`
- preserve traceability to paper/evidence artifacts
- avoid red-line violations from `ResearchBrief.json`
- keep target tasks inside the `ResearchBrief` topic/domain instead of appending
  unrelated benchmark, policy-optimization, or other cross-domain suffixes
- directly address prior `IdeaDecision.required_changes` when revising

## Failure Mode
If the LLM returns malformed ideas, write `IdeaGeneration.raw.txt` and fall back to deterministic evidence-grounded ideas.

If no valid ideas remain after fallback:

- write `IdeaGenerationReport.json`
- route back to `READ`

## Runtime Implementation
Current runtime implementation:

```text
ResearchAgent.ideate()
validators/idea_validator.py
```
