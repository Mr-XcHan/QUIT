from __future__ import annotations

from quit_agent.schemas.enums import FallbackTarget, IdeaDecisionType, WorkflowState
from quit_agent.schemas.review_artifacts import IdeaDecision


def next_after_idea_decision(decision: IdeaDecision) -> WorkflowState:
    if decision.decision == IdeaDecisionType.PASS:
        return WorkflowState.BUILD_SPEC

    fallback_map = {
        FallbackTarget.RETRIEVE: WorkflowState.RETRIEVE,
        FallbackTarget.READ: WorkflowState.READ,
        FallbackTarget.IDEATE: WorkflowState.IDEATE,
        FallbackTarget.BUILD_SPEC: WorkflowState.BUILD_SPEC,
        FallbackTarget.STOP: WorkflowState.STOP,
    }
    if decision.fallback_target in fallback_map:
        return fallback_map[decision.fallback_target]

    if decision.missing_evidence:
        return WorkflowState.READ
    retrieval_terms = " ".join([decision.reason, *decision.violations, *decision.required_changes]).lower()
    if "retriev" in retrieval_terms or "paper" in retrieval_terms or "source" in retrieval_terms:
        return WorkflowState.RETRIEVE
    return WorkflowState.IDEATE
