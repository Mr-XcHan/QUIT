from __future__ import annotations

from quit_agent.schemas.enums import WorkflowState


REQUIRED_INPUTS: dict[WorkflowState, list[str]] = {
    WorkflowState.PLAN: [],
    WorkflowState.VALIDATE_BRIEF: ["ResearchBrief.raw.json"],
    WorkflowState.RETRIEVE: ["ResearchBrief.json"],
    WorkflowState.READ: ["ResearchBrief.json", "PaperCards.jsonl"],
    WorkflowState.IDEATE: ["ResearchBrief.json", "EvidenceCards.jsonl"],
    WorkflowState.IDEA_EVAL: ["ResearchBrief.json", "IdeaLibrary.jsonl"],
    WorkflowState.BUILD_SPEC: ["IdeaDecision.json", "IdeaLibrary.jsonl"],
    WorkflowState.CODE: ["BuildSpec.json"],
    WorkflowState.CODE_EVAL: ["BuildSpec.json", "EXPERIMENT_LOG.md"],
    WorkflowState.CODE_LLM_EVAL: ["results/metrics.json", "results/results_table.csv", "BuildSpec.json"],
    WorkflowState.CODE_REVISE: ["CodePerformanceEval.json", "BuildSpec.json"],
    WorkflowState.WRITE: ["BuildSpec.json"],
    WorkflowState.WRITE_EVAL: ["paper_gene/main.tex"],
    WorkflowState.WRITE_LLM_EVAL: ["paper_gene/main.tex", "BuildSpec.json"],
    WorkflowState.WRITE_REVISE: ["paper_gene/main.tex", "PaperReview.json"],
    WorkflowState.WRITE_FINAL_EVAL: ["paper_gene/main.tex"],
    WorkflowState.EXTRACT: ["paper_gene/main.tex", "BuildSpec.json"],
    WorkflowState.STOP: [],
}
