from __future__ import annotations

from quit_agent.agents.builder_agent import BuilderAgent
from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.evidence_card import EvidenceCard
from quit_agent.schemas.enums import FallbackTarget, IdeaDecisionType
from quit_agent.schemas.idea_card import IdeaCard
from quit_agent.schemas.repo_card import RepoCard
from quit_agent.schemas.research_brief import ResearchBrief
from quit_agent.schemas.review_artifacts import IdeaDecision


def test_build_spec_fallback_uses_approved_idea_and_evidence(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts)
    brief = ResearchBrief(topic="offline RL flow matching", constraints=["offline only"])
    idea = IdeaCard(
        idea_id="idea-1",
        target_task="offline policy optimization",
        novelty_claim="combine RWFM with pessimistic value estimates",
        supporting_evidence_ids=["ev-1"],
        expected_gain="better normalized return",
    )
    decision = IdeaDecision(
        idea_id="idea-1",
        decision=IdeaDecisionType.PASS,
        reason="good",
        fallback_target=FallbackTarget.BUILD_SPEC,
    )
    evidence = [
        EvidenceCard(
            evidence_id="ev-1",
            paper_id="p1",
            task="offline RL",
            method="Reward-Weighted Flow Matching",
            metrics=["normalized return"],
            limitations=["limited robustness"],
            transferable_idea_seeds=["flow matching loss"],
        )
    ]

    repo_path = tmp_path / "reference_repo"
    repo_path.mkdir()
    requirements_path = repo_path / "requirements.txt"
    requirements_path.write_text("torch\nnumpy\n", encoding="utf-8")
    repo = RepoCard(
        repo_id="owner_repo",
        repo_url="https://github.com/owner/repo",
        source_paper_id="p1",
        source_title="paper",
        local_repo_path=str(repo_path),
        env_files=[str(requirements_path)],
        relevance_score=0.9,
    )

    spec, prompt, response = builder.build_spec(brief=brief, idea=idea, decision=decision, evidence=evidence, repos=[repo])

    assert "ResearchBrief" in prompt
    assert "idea-1" in response
    assert spec.idea_id == "idea-1"
    assert "p1" in spec.citations_required
    assert "normalized return" in spec.metrics
    assert spec.repo_url == "https://github.com/owner/repo"
    assert spec.environment.source == "reference_repo"
    assert "torch" in spec.environment.requirements
    assert artifacts.path("BuildSpec.json").exists()
    assert artifacts.path("RepoCloneReport.json").exists()


class BuildSpecLLM:
    def complete(self, prompt: str) -> str:
        return """{
          "build_id": "build-idea-1",
          "idea_id": "idea-1",
          "target_task": "offline policy optimization",
          "problem_statement": "Improve offline policy optimization.",
          "method_summary": "Use flow matching with pessimistic value estimates.",
          "repo_url": "",
          "implementation_plan": ["implement method"],
          "experiment_plan": ["run D4RL"],
          "baselines": ["IQL"],
          "metrics": ["normalized return"],
          "success_criteria": ["beat IQL"],
          "artifacts_required": {
            "coder": ["working code module"],
            "writer": ["latex section draft"]
          },
          "paper_outline": ["Abstract", "Methods", "Experiments"],
          "citations_required": ["p1"]
        }"""


def test_build_spec_uses_llm_when_available(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts, llm=BuildSpecLLM())
    brief = ResearchBrief(topic="offline RL flow matching")
    idea = IdeaCard(
        idea_id="idea-1",
        target_task="offline policy optimization",
        novelty_claim="combine RWFM with pessimistic value estimates",
        supporting_evidence_ids=["ev-1"],
        expected_gain="better normalized return",
    )
    decision = IdeaDecision(
        idea_id="idea-1",
        decision=IdeaDecisionType.PASS,
        reason="good",
        fallback_target=FallbackTarget.BUILD_SPEC,
    )

    spec, _, _ = builder.build_spec(brief=brief, idea=idea, decision=decision, evidence=[])

    assert spec.baselines == ["IQL"]
    assert spec.citations_required == ["p1"]


class CrossDomainBuildSpecLLM:
    def complete(self, prompt: str) -> str:
        return """{
          "build_id": "build-idea-3",
          "idea_id": "idea-3",
          "target_task": "3D pattern synthesis",
          "problem_statement": "Develop an offline RL method for controllable 3D patterns.",
          "method_summary": "Hierarchical 3D Gaussian pattern generator.",
          "repo_url": "",
          "implementation_plan": ["Define the offline dataset interface."],
          "experiment_plan": ["Run baseline methods on the same offline datasets."],
          "baselines": ["BCQ", "CQL"],
          "metrics": ["PSNR"],
          "success_criteria": ["Measurable improvement over at least one strong offline RL baseline."],
          "artifacts_required": {
            "coder": ["working code module"],
            "writer": ["latex section draft"]
          },
          "paper_outline": ["Abstract", "Methods", "Experiments"],
          "citations_required": ["p1"]
        }"""


def test_build_spec_does_not_apply_domain_specific_baseline_blacklist(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    builder = BuilderAgent(artifacts, llm=CrossDomainBuildSpecLLM())
    brief = ResearchBrief(topic="interactive 3D pattern synthesis", domain=["3D generative models"])
    idea = IdeaCard(
        idea_id="idea-3",
        target_task="controllable 3D pattern synthesis",
        novelty_claim="hierarchical macro rules with micro Gaussian detail",
        supporting_evidence_ids=["ev-1"],
        expected_gain="better controllability and rendering quality",
    )
    decision = IdeaDecision(
        idea_id="idea-3",
        decision=IdeaDecisionType.PASS,
        reason="good",
        fallback_target=FallbackTarget.BUILD_SPEC,
    )
    evidence = [
        EvidenceCard(
            evidence_id="ev-1",
            paper_id="p1",
            task="3D scene representation",
            method="3D Gaussian Splatting",
            metrics=["PSNR"],
        )
    ]

    spec, _, _ = builder.build_spec(brief=brief, idea=idea, decision=decision, evidence=evidence)

    assert spec.baselines == ["BCQ", "CQL"]
    assert "offline RL" in spec.problem_statement
    assert any("offline RL" in criterion for criterion in spec.success_criteria)
