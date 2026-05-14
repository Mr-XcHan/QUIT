from __future__ import annotations

from quit_agent.schemas.paper_card import PaperCard
from quit_agent.tools.repo_tools import RepoManager, normalize_repo_url, repo_id_from_url


def test_repo_url_normalization():
    assert normalize_repo_url("https://github.com/user/project.git") == "https://github.com/user/project"
    assert normalize_repo_url("see https://github.com/user/project/tree/main") == "https://github.com/user/project"
    assert repo_id_from_url("https://github.com/user/project") == "user_project"


def test_repo_manager_extracts_repos_from_papers(tmp_path):
    paper = PaperCard(
        paper_id="p1",
        title="Paper With Code",
        authors=[],
        year=2025,
        venue="arXiv",
        abstract="Code is available at https://github.com/example/offline-rl.",
        source="arxiv",
        paper_url="",
        pdf_url="",
        code_url="",
        query_source="q",
        retrieval_score=0.8,
    )
    manager = RepoManager(tmp_path)

    repos, report = manager.collect_from_papers([paper], max_repos=3, run_id="run")

    assert report["repo_count"] == 1
    assert repos[0].repo_url == "https://github.com/example/offline-rl"
    assert repos[0].source_paper_id == "p1"
