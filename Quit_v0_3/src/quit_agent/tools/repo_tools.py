from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from quit_agent.schemas.paper_card import PaperCard
from quit_agent.schemas.repo_card import RepoCard
from quit_agent.tools.retrievers import safe_slug


GITHUB_RE = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
ENV_FILE_NAMES = {
    "requirements.txt",
    "environment.yml",
    "environment.yaml",
    "pyproject.toml",
    "setup.py",
    "Pipfile",
    "poetry.lock",
    "Dockerfile",
}


class RepoManager:
    """Extract, clone, and inspect code repositories linked to papers."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        timeout_seconds: int = 60,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.timeout_seconds = timeout_seconds

    def collect_from_papers(self, papers: list[PaperCard], *, max_repos: int, run_id: str) -> tuple[list[RepoCard], dict]:
        """Return top repo cards discovered from selected papers.

        The tool prefers explicit PaperCard.code_url, then GitHub URLs found in
        metadata text. It does not clone repositories; cloning happens after an
        idea is approved, when the relevant evidence is known.
        """
        discovered: dict[str, RepoCard] = {}
        for paper in papers:
            for url in self._repo_urls_for_paper(paper):
                key = normalize_repo_url(url)
                if not key:
                    continue
                card = RepoCard(
                    repo_id=repo_id_from_url(key),
                    repo_url=key,
                    source_paper_id=paper.paper_id,
                    source_title=paper.title,
                    relevance_score=paper.retrieval_score,
                )
                existing = discovered.get(key)
                if existing is None or card.relevance_score > existing.relevance_score:
                    discovered[key] = card

        repos = sorted(discovered.values(), key=lambda item: item.relevance_score, reverse=True)[: max(0, max_repos)]
        for repo in repos:
            if repo.local_repo_path:
                repo = self.inspect(repo)

        report = {
            "status": "PASS",
            "repo_count": len(repos),
            "max_repos": max_repos,
            "clone_stage": "BUILD_SPEC",
            "clone_attempted": 0,
            "clone_succeeded": 0,
            "env_file_count": sum(len(repo.env_files) for repo in repos),
        }
        return repos, report

    def inspect(self, repo: RepoCard) -> RepoCard:
        path = Path(repo.local_repo_path)
        if not path.exists() or not path.is_dir():
            return repo
        env_files = []
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and candidate.name in ENV_FILE_NAMES:
                env_files.append(str(candidate))
        repo.env_files = env_files[:16]
        repo.language = self._detect_language(path, repo.env_files)
        repo.framework = self._detect_framework(repo.env_files)
        repo.status = "inspected"
        return repo

    def clone_and_inspect(self, repo: RepoCard) -> RepoCard:
        repo_dir = self.output_dir / repo.repo_id
        repo.local_repo_path = str(repo_dir)
        if not repo_dir.exists():
            try:
                repo_dir.parent.mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(
                    ["git", "clone", "--depth", "1", repo.repo_url, str(repo_dir)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except Exception as exc:
                repo.status = "failed"
                repo.errors.append(str(exc))
                return repo
            if completed.returncode != 0:
                repo.status = "failed"
                repo.errors.append(completed.stderr[-1000:] or "git clone failed")
                return repo
            repo.status = "cloned"
        return self.inspect(repo)

    def _repo_urls_for_paper(self, paper: PaperCard) -> list[str]:
        candidates = []
        if paper.code_url:
            candidates.append(paper.code_url)
        text = " ".join([paper.title, paper.abstract, paper.paper_url, paper.pdf_url])
        candidates.extend(GITHUB_RE.findall(text))
        return candidates

    def _detect_language(self, path: Path, env_files: list[str]) -> str:
        if any(file.endswith((".py", "pyproject.toml", "requirements.txt", "setup.py")) for file in env_files):
            return "python"
        if list(path.glob("**/*.py")):
            return "python"
        if list(path.glob("**/*.ipynb")):
            return "python"
        return ""

    def _detect_framework(self, env_files: list[str]) -> str:
        text = "\n".join(_read_small(Path(item)) for item in env_files)
        lowered = text.lower()
        frameworks = []
        for name in ["torch", "jax", "tensorflow", "gym", "d4rl", "transformers"]:
            if name in lowered:
                frameworks.append(name)
        return ", ".join(frameworks)


def normalize_repo_url(url: str) -> str:
    match = GITHUB_RE.search(url.strip())
    if not match:
        return ""
    parsed = urlparse(match.group(0))
    path = parsed.path.strip("/").rstrip(").,;:").removesuffix(".git")
    parts = path.split("/")
    if len(parts) < 2:
        return ""
    return f"https://github.com/{parts[0]}/{parts[1]}"


def repo_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    return safe_slug(parsed.path.strip("/").replace("/", "_"), max_chars=80)


def _read_small(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:10000]
    except OSError:
        return ""
