from __future__ import annotations

import json

from quit_agent.agents.research_agent import ResearchAgent
from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.research_brief import ResearchBrief, SearchBudget
from quit_agent.tools.retrievers import LocalPaperDatabaseRetriever, PdfDownloader, normalize_candidate


def test_local_paper_database_retriever_ranks_matches(tmp_path):
    db = tmp_path / "papers.jsonl"
    db.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p1",
                        "title": "Offline RL with Conservative Policy Improvement",
                        "authors": ["Alice Zhang"],
                        "year": 2024,
                        "venue": "ICLR",
                        "abstract": "offline reinforcement learning policy generalization",
                        "paper_url": "https://example.test/p1",
                        "source": "ICLR",
                    }
                ),
                json.dumps(
                    {
                        "paper_id": "p2",
                        "title": "Unrelated Vision Model",
                        "authors": ["Bob Li"],
                        "year": 2023,
                        "venue": "CVPR",
                        "abstract": "image segmentation",
                        "paper_url": "https://example.test/p2",
                        "source": "CVPR",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    results = LocalPaperDatabaseRetriever(db).search("offline RL policy improvement", 5)

    assert len(results) == 1
    assert results[0]["paper_id"] == "p1"


def test_local_paper_database_retriever_supports_pdf_directory(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "2024_iclr_zhang_sparseformer_offline_rl.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    results = LocalPaperDatabaseRetriever(pdf_dir).search("offline rl sparseformer", 5)

    assert len(results) == 1
    assert results[0]["title"] == "Sparseformer Offline Rl"
    assert results[0]["year"] == 2024
    assert results[0]["venue"] == "iclr"
    assert results[0]["authors"] == ["zhang"]
    assert results[0]["source"] == "local_pdf"
    assert results[0]["local_pdf_path"] == str(pdf_path)


def test_normalize_candidate_accepts_openreview_author_objects():
    result = normalize_candidate(
        {
            "paper_id": "or-1",
            "title": "Mobile Charging Robot Scheduling",
            "authors": [
                {"fullname": "Mohammed Amine Merzoug", "username": ""},
                {"name": "Ahmed Mostefaoui"},
                {"username": "~Ernesto_Damiani1"},
            ],
            "year": 2025,
        },
        query="mobile charging robot",
        source="openreview",
    )

    assert result["authors"] == [
        "Mohammed Amine Merzoug",
        "Ahmed Mostefaoui",
        "~Ernesto_Damiani1",
    ]


def test_local_pdf_directory_keeps_user_seed_papers_without_query_overlap(tmp_path):
    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "2019_icml_x_bcq.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    results = LocalPaperDatabaseRetriever(pdf_dir).search("flow matching generalization", 5)

    assert len(results) == 1
    assert results[0]["paper_id"] == "2019_icml_x_bcq"
    assert results[0]["retrieval_score"] == 0.25


def test_retrieve_writes_budgeted_paper_cards_and_report(tmp_path):
    db = tmp_path / "papers.jsonl"
    db.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Offline RL with Conservative Policy Improvement",
                "authors": ["Alice Zhang"],
                "year": 2024,
                "venue": "ICLR",
                "abstract": "offline reinforcement learning policy generalization",
                "paper_url": "https://example.test/p1",
                "source": "ICLR",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = ArtifactManager(tmp_path, "run")
    brief = ResearchBrief(
        topic="offline RL policy generalization",
        domain=["offline reinforcement learning"],
        search_budget=SearchBudget(max_queries=2, max_papers_screened=5, max_papers_selected=1),
    )
    papers, report, _, _ = ResearchAgent(LocalPaperDatabaseRetriever(db), artifacts).retrieve(brief)

    assert len(papers) == 1
    assert report["status"] == "PASS"
    assert artifacts.path("PaperCards.jsonl").exists()
    assert artifacts.path("RetrievalReport.json").exists()


def test_retrieve_fails_below_minimum_selected_ratio(tmp_path):
    db = tmp_path / "papers.jsonl"
    db.write_text(
        json.dumps(
            {
                "paper_id": "p1",
                "title": "Offline RL with Conservative Policy Improvement",
                "authors": ["Alice Zhang"],
                "year": 2024,
                "venue": "ICLR",
                "abstract": "offline reinforcement learning policy generalization",
                "paper_url": "https://example.test/p1",
                "source": "ICLR",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts = ArtifactManager(tmp_path, "run")
    brief = ResearchBrief(
        topic="offline RL policy generalization",
        domain=["offline reinforcement learning"],
        search_budget=SearchBudget(max_queries=2, max_papers_screened=5, max_papers_selected=10),
    )

    papers, report, _, _ = ResearchAgent(LocalPaperDatabaseRetriever(db), artifacts).retrieve(brief)

    assert len(papers) == 1
    assert report["minimum_selected"] == 2
    assert report["status"] == "FAIL"
    assert artifacts.path("RetrievalFailure.json").exists()


def test_retrieve_distinguishes_empty_search_from_download_failure(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    retriever = RecordingRetriever()
    brief = ResearchBrief(
        topic="flow matching offline RL",
        search_keywords=["flow matching", "offline RL"],
        domain=["offline reinforcement learning", "diffusion policies"],
        search_budget=SearchBudget(
            max_queries=4,
            max_papers_screened=10,
            max_papers_selected=5,
            stop_if_no_new_signal_rounds=2,
        ),
    )

    papers, report, _, _ = ResearchAgent(
        retriever,
        artifacts,
        downloader=PdfDownloader(tmp_path / "pdfs"),
        min_downloads=2,
    ).retrieve(brief)

    assert papers == []
    assert report["status"] == "FAIL"
    assert report["search_empty"] is True
    assert report["download_fail"] is False
    assert report["download_attempted"] == 0
    assert report["stop_reason"] == "max_queries"
    assert len(retriever.queries) == 4


def test_default_queries_use_short_search_keywords_not_topic_domain_blob(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    agent = ResearchAgent(RecordingRetriever(), artifacts)
    brief = ResearchBrief(
        topic="Flow-Matching Policy Improvement for Offline RL with Conservative Action Regularization",
        search_keywords=["flow matching offline RL", "rectified flow policy", "diffusion policy offline"],
        domain=[
            "Offline Reinforcement Learning",
            "Policy Improvement",
            "Flow Matching",
            "Continuous Normalizing Flows",
            "Diffusion Models for RL",
            "Score-Based Generative Modeling",
        ],
    )

    queries = agent._queries(brief)

    assert queries[:4] == [
        "Flow-Matching Policy Improvement for Offline RL with Conservative Action Regularization",
        "flow matching offline RL",
        "rectified flow policy",
        "diffusion policy offline",
    ]
    assert all("Offline Reinforcement Learning Policy Improvement Flow Matching" not in query for query in queries)


def test_pdf_downloader_filename_is_normalized(tmp_path):
    from quit_agent.schemas.paper_card import PaperCard

    paper = PaperCard(
        paper_id="p1",
        title="SparseFormer: Efficient Offline RL!",
        authors=["Wei Zhang", "A. Smith"],
        year=2024,
        venue="ICLR",
    )

    assert PdfDownloader(tmp_path).filename_for(paper) == "2024_iclr_zhang_sparseformer_efficient_offline_rl.pdf"


class QueryLLM:
    def complete(self, prompt: str) -> str:
        return '["latest offline RL policy generalization", "influential conservative policy improvement"]'


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[dict]:
        self.queries.append(query)
        return []


def test_retrieve_can_use_llm_query_planning(tmp_path):
    artifacts = ArtifactManager(tmp_path, "run")
    retriever = RecordingRetriever()
    brief = ResearchBrief(
        topic="offline RL policy generalization",
        domain=["offline reinforcement learning"],
        search_budget=SearchBudget(max_queries=2, max_papers_screened=5, max_papers_selected=1),
    )

    ResearchAgent(retriever, artifacts, query_llm=QueryLLM()).retrieve(brief)

    assert retriever.queries == [
        "latest offline RL policy generalization",
        "influential conservative policy improvement",
    ]
    assert artifacts.path("RetrieveQueryPlan.json").exists()
