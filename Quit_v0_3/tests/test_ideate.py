from __future__ import annotations

from quit_agent.agents.research_agent import ResearchAgent
from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.evidence_card import EvidenceCard
from quit_agent.schemas.enums import FallbackTarget, IdeaDecisionType
from quit_agent.schemas.research_brief import ResearchBrief
from quit_agent.schemas.review_artifacts import IdeaDecision
from quit_agent.tools.retrievers import MockRetriever


def test_ideate_generates_traceable_fallback_ideas(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    agent = ResearchAgent(MockRetriever(), artifacts)
    brief = ResearchBrief(topic="offline RL policy generalization", domain=["offline RL"])
    evidence = [
        EvidenceCard(
            evidence_id="ev-1",
            paper_id="p1",
            task="offline RL under distribution shift",
            method="flow matching policy optimization",
            setting="D4RL normalized return",
            claims=["flow matching can improve policy optimization"],
            metrics=["normalized return"],
            limitations=["limited robustness evaluation"],
            transferable_idea_seeds=["variable horizon flow policy"],
        ),
        EvidenceCard(
            evidence_id="ev-2",
            paper_id="p2",
            task="offline policy improvement",
            method="variance regularization",
            setting="continuous control benchmark",
            claims=["regularization reduces overestimation"],
            metrics=["return"],
            limitations=["sensitive to dataset coverage"],
            transferable_idea_seeds=["variance constrained objective"],
        ),
    ]

    ideas, report, _, _ = agent.ideate(brief, evidence)

    assert report["status"] == "PASS"
    assert report["fallback_used"] is True
    assert ideas
    assert all(idea.supporting_evidence_ids for idea in ideas)
    assert artifacts.path("IdeaClusters.json").exists()
    assert artifacts.path("IdeaLibrary.jsonl").exists()
    assert artifacts.path("IdeaGenerationReport.json").exists()


class IdeaLLM:
    def complete(self, prompt: str) -> str:
        return """[
          {
            "idea_id": "idea-flow-1",
            "target_task": "offline RL distribution shift",
            "novelty_claim": "Use flow matching with variance constraints for robust offline policy improvement.",
            "supporting_evidence_ids": ["ev-1", "ev-2"],
            "expected_gain": "Improve normalized return under dataset shift."
          }
        ]"""


def test_ideate_uses_llm_when_available(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    agent = ResearchAgent(MockRetriever(), artifacts, read_llm=IdeaLLM())
    brief = ResearchBrief(topic="offline RL policy generalization", domain=["offline RL"])
    evidence = [
        EvidenceCard(evidence_id="ev-1", paper_id="p1", task="offline RL", method="flow matching", claims=["claim"]),
        EvidenceCard(evidence_id="ev-2", paper_id="p2", task="offline RL", method="variance constraints", claims=["claim"]),
    ]

    ideas, report, _, _ = agent.ideate(brief, evidence)

    assert report["status"] == "PASS"
    assert report["fallback_used"] is False
    assert ideas[0].idea_id == "idea-flow-1"
    assert ideas[0].supporting_evidence_ids == ["ev-1", "ev-2"]


def test_ideate_revision_prompt_and_fallback_stay_on_brief_domain(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    agent = ResearchAgent(MockRetriever(), artifacts)
    brief = ResearchBrief(
        topic="Interactive 3D Pattern Synthesis via Structure-Geometry Decoupling",
        domain=["3D generation", "interactive editing"],
        constraints=["explicitly separate macro structure from micro geometry", "4x4 seamless tiling"],
        acceptance_criteria=["LPIPS reduction", "seam quality score", "editing latency under 500ms"],
    )
    evidence = [
        EvidenceCard(
            evidence_id="ev-1",
            paper_id="p1",
            task="interactive 3D scene editing",
            method="3D Gaussian splatting with semantic masks",
            setting="user-guided editing",
            claims=["semantic grouping improves local edits"],
            metrics=["LPIPS", "edit consistency"],
            limitations=["No explicit limitations extracted from available text."],
            transferable_idea_seeds=["macro semantic grouping"],
        ),
        EvidenceCard(
            evidence_id="ev-2",
            paper_id="p2",
            task="tileable 3D pattern synthesis",
            method="triplane geometry decoder",
            setting="seamless texture generation",
            claims=["triplanes can reduce memory"],
            metrics=["seam quality", "latency"],
            limitations=["limited interactive control"],
            transferable_idea_seeds=["micro geometry detail propagation"],
        ),
    ]
    decision = IdeaDecision(
        idea_id="idea-1",
        decision=IdeaDecisionType.REVISE,
        reason="target task contains unrelated flow matching policy optimization wording",
        fallback_target=FallbackTarget.IDEATE,
        violations=["target task mismatch"],
        required_changes=["remove flow matching policy optimization", "define LPIPS, seam quality, and latency gains"],
    )

    ideas, report, prompt, _ = agent.ideate(brief, evidence, decision)

    combined = "\n".join([prompt, *[idea.target_task + "\n" + idea.expected_gain + "\n" + idea.novelty_claim for idea in ideas]])
    assert report["status"] == "PASS"
    assert "required_changes" in prompt
    assert "remove flow matching policy optimization" in prompt
    assert "flow matching policy optimization" not in "\n".join(idea.target_task for idea in ideas).lower()
    assert "No explicit limitations extracted" not in combined
    assert any("LPIPS" in idea.expected_gain for idea in ideas)
