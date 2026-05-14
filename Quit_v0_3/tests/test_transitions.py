from __future__ import annotations

from quit_agent.orchestrator.transitions import next_after_idea_decision
from quit_agent.schemas.enums import FallbackTarget, IdeaDecisionType, WorkflowState
from quit_agent.schemas.review_artifacts import IdeaDecision


def decision(**kwargs):
    defaults = {
        "idea_id": "idea-1",
        "decision": IdeaDecisionType.PASS,
        "reason": "ok",
        "fallback_target": FallbackTarget.BUILD_SPEC,
    }
    defaults.update(kwargs)
    return IdeaDecision(**defaults)


def test_pass_routes_to_build_spec():
    assert next_after_idea_decision(decision()) == WorkflowState.BUILD_SPEC


def test_revise_routes_to_ideate():
    item = decision(decision=IdeaDecisionType.REVISE, fallback_target=FallbackTarget.IDEATE)
    assert next_after_idea_decision(item) == WorkflowState.IDEATE


def test_revise_missing_evidence_can_route_to_read():
    item = decision(
        decision=IdeaDecisionType.REVISE,
        fallback_target=FallbackTarget.READ,
        missing_evidence=["support"],
    )
    assert next_after_idea_decision(item) == WorkflowState.READ


def test_revise_retrieval_issue_can_route_to_retrieve():
    item = decision(
        decision=IdeaDecisionType.REVISE,
        fallback_target=FallbackTarget.RETRIEVE,
        required_changes=["retrieve a better paper set"],
    )
    assert next_after_idea_decision(item) == WorkflowState.RETRIEVE


def test_reject_missing_evidence_routes_to_read():
    item = decision(
        decision=IdeaDecisionType.REJECT,
        fallback_target=FallbackTarget.READ,
        missing_evidence=["support"],
    )
    assert next_after_idea_decision(item) == WorkflowState.READ


def test_reject_retrieval_issue_routes_to_retrieve():
    item = decision(
        decision=IdeaDecisionType.REJECT,
        reason="retrieval quality poor",
        fallback_target=FallbackTarget.RETRIEVE,
    )
    assert next_after_idea_decision(item) == WorkflowState.RETRIEVE
