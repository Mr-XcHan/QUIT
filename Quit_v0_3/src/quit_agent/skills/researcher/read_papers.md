# Skill: Read Papers

## Description
Read selected papers and convert them into structured, traceable evidence artifacts.

This skill is used by the `READ` workflow state. It is not a free-form summarization step. Its output must be evidence-oriented and suitable for downstream `IDEATE`.

## When To Use
Use this skill when:

- `ResearchBrief.json` exists and has passed validation.
- `PaperCards.jsonl` exists and RETRIEVE passed.
- The workflow needs `EvidenceCards.jsonl` for ideation.

## Input Artifacts
Required:

```text
ResearchBrief.json
PaperCards.jsonl
```

Each selected `PaperCard` should include either:

```text
local_pdf_path
```

or enough metadata, especially `title` and `abstract`, to allow partial evidence extraction.

## Output Artifacts
Required:

```text
PaperTexts.jsonl
EvidenceCards.jsonl
EvidenceValidationReport.json
ReadReport.json
```

Optional debug output:

```text
read_raw/<paper_id>.txt
```

## Runtime Pipeline
READ supports two runtime modes:

```text
local_text
direct_pdf
```

`local_text` is the default and is required for local text-only models:

```text
PaperCard
  -> PaperTextExtractor tool
  -> optional LLM evidence reader
  -> rule-based fallback evidence extractor
  -> EvidenceValidator
```

`direct_pdf` is for API providers that can read attached PDF files directly:

```text
PaperCard with local_pdf_path
  -> file-input capable LLM reader
  -> EvidenceCard
  -> metadata PaperText artifact for traceability
```

If `direct_pdf` is configured but the active provider does not expose file-input reading, READ automatically falls back to `local_text`.

The PDF/text extraction tool writes `PaperText` records:

```json
{
  "paper_id": "",
  "title": "",
  "abstract": "",
  "full_text": "",
  "sections": {
    "abstract": "",
    "introduction": "",
    "method": "",
    "experiments": "",
    "limitations": "",
    "conclusion": ""
  },
  "source_path": "",
  "extraction_status": "success|partial|failed",
  "errors": []
}
```

The evidence reader writes `EvidenceCard` records:

```json
{
  "paper_id": "",
  "task": "",
  "method": "",
  "setting": "",
  "claims": [],
  "metrics": [],
  "limitations": [],
  "transferable_idea_seeds": []
}
```

## Validation Rules
READ passes only when every selected paper has at least one `EvidenceCard` linked by `paper_id`.

If some PDFs cannot be parsed but metadata is available, produce a partial `PaperText` and fallback `EvidenceCard` rather than crashing.

If evidence is missing for any selected paper:

- write `EvidenceValidationReport.json`
- write `ReadReport.json`
- route back to `RETRIEVE`

## Failure Mode
If text extraction fails for many papers, READ should still preserve the failure details as artifacts. The orchestrator can then decide whether to retry READ with a better extractor or return to RETRIEVE for better papers.

## Runtime Implementation
Current runtime implementation:

```text
ResearchAgent.read()
tools/paper_reader.py
schemas/paper_text.py
```
