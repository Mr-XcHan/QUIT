from __future__ import annotations

from quit_agent.agents.research_agent import ResearchAgent
from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.paper_card import PaperCard
from quit_agent.schemas.research_brief import ResearchBrief
from quit_agent.tools.paper_reader import LocalPdfTextExtractor
from quit_agent.tools.retrievers import MockRetriever


def test_paper_text_extractor_falls_back_to_metadata_without_pdf(tmp_path):
    paper = PaperCard(
        paper_id="p1",
        title="Flow Matching for Policy Improvement",
        authors=["A. Researcher"],
        year=2025,
        venue="arXiv",
        abstract="We improve policy learning with flow matching and evaluate reward.",
        source="arxiv",
    )

    text = LocalPdfTextExtractor().extract(paper)

    assert text.paper_id == "p1"
    assert text.extraction_status == "partial"
    assert "flow matching" in text.full_text.lower()
    assert text.sections.abstract == paper.abstract


def test_read_writes_paper_texts_evidence_and_reports(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    agent = ResearchAgent(MockRetriever(), artifacts)
    brief = ResearchBrief(topic="flow matching policy generalization", domain=["reinforcement learning"])
    papers = [
        PaperCard(
            paper_id="p1",
            title="Flow Matching for Policy Improvement",
            authors=["A. Researcher"],
            year=2025,
            venue="arXiv",
            abstract="We improve policy learning with flow matching and report reward gains.",
            source="arxiv",
        )
    ]

    evidence, report, _, _ = agent.read(brief, papers)

    assert report["status"] == "PASS"
    assert report["paper_text_count"] == 1
    assert len(evidence) == 1
    assert evidence[0].paper_id == "p1"
    assert artifacts.path("PaperTexts.jsonl").exists()
    assert artifacts.path("EvidenceCards.jsonl").exists()
    assert artifacts.path("EvidenceValidationReport.json").exists()
    assert artifacts.path("ReadReport.json").exists()


class FileReadingLLM:
    def complete_with_files(self, prompt: str, file_paths: list[str]) -> str:
        assert file_paths
        return """{
          "paper_id": "p1",
          "task": "offline policy improvement",
          "method": "direct PDF evidence extraction",
          "setting": "attached PDF",
          "claims": ["file input model read the paper"],
          "metrics": ["reward"],
          "limitations": ["depends on file-input capable provider"],
          "transferable_idea_seeds": ["use direct PDF reading when provider supports files"]
        }"""

    def complete(self, prompt: str) -> str:
        raise AssertionError("direct_pdf mode should use complete_with_files first")


def test_read_direct_pdf_mode_uses_file_capable_llm(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    artifacts = ArtifactManager(tmp_path, "run")
    agent = ResearchAgent(MockRetriever(), artifacts, read_llm=FileReadingLLM(), read_mode="direct_pdf")
    brief = ResearchBrief(topic="flow matching policy generalization", domain=["reinforcement learning"])
    papers = [
        PaperCard(
            paper_id="p1",
            title="Flow Matching for Policy Improvement",
            authors=["A. Researcher"],
            year=2025,
            venue="arXiv",
            abstract="We improve policy learning with flow matching.",
            source="arxiv",
            local_pdf_path=str(pdf_path),
        )
    ]

    evidence, report, _, _ = agent.read(brief, papers)

    assert report["status"] == "PASS"
    assert report["read_mode"] == "direct_pdf"
    assert report["direct_pdf"]["attempted"] == 1
    assert report["direct_pdf"]["succeeded"] == 1
    assert evidence[0].method == "direct PDF evidence extraction"
