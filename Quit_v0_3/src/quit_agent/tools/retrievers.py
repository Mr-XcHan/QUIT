from __future__ import annotations

import json
import re
from json import JSONDecoder
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from quit_agent.config import RetrievalConfig
from quit_agent.schemas.enums import PaperStatus
from quit_agent.schemas.paper_card import PaperCard
from quit_agent.tools.retriever_interface import Retriever


class MockRetriever:
    def search(self, query: str, max_results: int) -> list[dict]:
        base = [
            ("Artifact-Centric Research Agents", "A system for research automation using artifacts, validators, and traceable state machines."),
            ("Evidence Grounded Idea Generation", "Methods for turning paper evidence into candidate research ideas with reviewer gates."),
            ("Local-First Agent Workflows", "A framework for local execution, mockable tools, and explicit artifact contracts."),
            ("Long Context Chat Agents", "A conversational agent relying on chat history for research assistance."),
            ("Budgeted Scientific Retrieval", "Lightweight ranking, top-k truncation, and stopping rules for paper retrieval."),
        ]
        results = []
        for idx, (title, abstract) in enumerate(base[:max_results]):
            results.append(
                {
                    "paper_id": f"mock-{abs(hash((query, title))) % 100000}",
                    "title": title,
                    "authors": ["A. Researcher", "B. Builder"],
                    "year": 2025 - idx,
                    "venue": "MockConf",
                    "abstract": abstract,
                    "source": "mock",
                    "paper_url": f"https://example.test/{idx}",
                    "pdf_url": f"https://example.test/{idx}.pdf",
                    "code_url": "",
                    "query_source": query,
                    "status": "found",
                    "local_pdf_path": "",
                    "retrieval_score": max(0.1, 1.0 - idx * 0.12),
                }
            )
        return results


class LocalPaperDatabaseRetriever:
    """Search a user-maintained paper database file or PDF directory.

    Supported inputs:
    - JSON/JSONL file with PaperCard-like records.
    - Directory containing JSON/JSONL metadata files.
    - Directory containing PDF files. PDF metadata is inferred from filenames.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def search(self, query: str, max_results: int) -> list[dict]:
        query_tokens = tokenize(query)
        candidates = []
        for raw in load_local_paper_records(self.database_path):
            item = normalize_candidate(raw, query=query, source="local")
            haystack = tokenize(" ".join([item["title"], item["abstract"], item["venue"], item["paper_id"]]))
            overlap = len(query_tokens & haystack)
            is_local_pdf = item["source"] == "local_pdf"
            if overlap == 0 and not is_local_pdf:
                continue
            item["retrieval_score"] = float(raw.get("retrieval_score") or overlap or 0.25)
            item["source"] = item["source"] or "local"
            candidates.append(item)
        candidates.sort(key=lambda item: item["retrieval_score"], reverse=True)
        return candidates[:max_results]


class ArxivRetriever:
    """Minimal arXiv API retriever using only the Python standard library."""

    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, max_results: int) -> list[dict]:
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all:{quote_plus(query)}&start=0&max_results={max_results}"
            "&sortBy=relevance&sortOrder=descending"
        )
        try:
            with urlopen(url, timeout=self.timeout_seconds) as response:
                payload = response.read()
        except OSError:
            return []
        root = ET.fromstring(payload)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in root.findall("atom:entry", ns):
            title = _xml_text(entry.find("atom:title", ns))
            abstract = _xml_text(entry.find("atom:summary", ns))
            paper_url = _xml_text(entry.find("atom:id", ns))
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            authors = [_xml_text(author.find("atom:name", ns)) for author in entry.findall("atom:author", ns)]
            published = _xml_text(entry.find("atom:published", ns))
            year = int(published[:4]) if published[:4].isdigit() else 1900
            results.append(
                normalize_candidate(
                    {
                        "paper_id": paper_url.rsplit("/", 1)[-1],
                        "title": " ".join(title.split()),
                        "authors": authors,
                        "year": year,
                        "venue": "arXiv",
                        "abstract": " ".join(abstract.split()),
                        "source": "arxiv",
                        "paper_url": paper_url,
                        "pdf_url": pdf_url,
                    },
                    query=query,
                    source="arxiv",
                )
            )
        return results


class OpenReviewRetriever:
    """Best-effort OpenReview search retriever."""

    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, max_results: int) -> list[dict]:
        url = f"https://api2.openreview.net/notes/search?term={quote_plus(query)}&limit={max_results}"
        try:
            with urlopen(url, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except OSError:
            return []
        notes = payload.get("notes") if isinstance(payload, dict) else None
        if not isinstance(notes, list):
            return []
        results = []
        for note in notes:
            content = note.get("content", {}) if isinstance(note, dict) else {}
            title = _content_value(content.get("title"))
            abstract = _content_value(content.get("abstract"))
            if not str(title).strip() and not str(abstract).strip():
                continue
            authors = _content_value(content.get("authors"))
            if isinstance(authors, str):
                authors = [authors]
            if not isinstance(authors, list):
                authors = []
            invitation = str(note.get("invitation") or "")
            note_id = str(note.get("id") or "")
            results.append(
                normalize_candidate(
                    {
                        "paper_id": note_id or title,
                        "title": title,
                        "authors": authors,
                        "year": _year_from_text(invitation),
                        "venue": _venue_from_invitation(invitation),
                        "abstract": abstract,
                        "source": "openreview",
                        "paper_url": f"https://openreview.net/forum?id={note_id}" if note_id else "",
                        "pdf_url": f"https://openreview.net/pdf?id={note_id}" if note_id else "",
                    },
                    query=query,
                    source="openreview",
                )
            )
        return results



class CompositeRetriever:
    """Fan out a query to multiple retrieval sources and combine candidates."""

    def __init__(self, retrievers: list[Retriever], per_source_results: int = 8) -> None:
        self.retrievers = retrievers
        self.per_source_results = per_source_results

    def search(self, query: str, max_results: int) -> list[dict]:
        batches: list[list[dict]] = []
        for retriever in self.retrievers:
            try:
                source_results = retriever.search(query, min(self.per_source_results, max_results))
            except Exception:
                source_results = []
            batches.append(source_results)

        # Round-robin keeps earlier sources as priority seeds without starving
        # later sources such as OpenReview when arXiv returns many records.
        results = []
        offset = 0
        while len(results) < max_results:
            added = False
            for batch in batches:
                if offset < len(batch):
                    results.append(batch[offset])
                    added = True
                    if len(results) >= max_results:
                        break
            if not added:
                break
            offset += 1
        return results[:max_results]


class PdfDownloader:
    """Download selected PDFs with deterministic local filenames."""

    def __init__(self, output_dir: str | Path, timeout_seconds: int = 30) -> None:
        self.output_dir = Path(output_dir)
        self.timeout_seconds = timeout_seconds

    def download(self, paper: PaperCard) -> PaperCard:
        if not paper.pdf_url:
            return paper
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = self.filename_for(paper)
        path = self.output_dir / filename
        try:
            with urlopen(paper.pdf_url, timeout=self.timeout_seconds) as response:
                path.write_bytes(response.read())
        except OSError:
            paper.status = PaperStatus.FAILED
            return paper
        paper.local_pdf_path = str(path)
        paper.status = PaperStatus.DOWNLOADED
        return paper

    def filename_for(self, paper: PaperCard) -> str:
        year = paper.year or 1900
        venue = safe_slug(paper.venue or paper.source, max_chars=24)
        author = first_author_slug(paper.authors)
        title = safe_slug(paper.title, max_chars=48)
        return f"{year}_{venue}_{author}_{title}.pdf"


def create_retriever(config: RetrievalConfig) -> Retriever:
    retrievers: list[Retriever] = []
    for source in config.sources:
        normalized = source.lower().strip()
        if normalized == "local":
            retrievers.append(LocalPaperDatabaseRetriever(config.local_database_path))
        elif normalized == "arxiv":
            retrievers.append(ArxivRetriever(timeout_seconds=config.timeout_seconds))
        elif normalized == "openreview":
            retrievers.append(OpenReviewRetriever(timeout_seconds=config.timeout_seconds))
        elif normalized == "mock":
            retrievers.append(MockRetriever())
    if not retrievers:
        retrievers.append(MockRetriever())
    return CompositeRetriever(retrievers, per_source_results=config.per_source_results)


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def safe_slug(value: str, *, max_chars: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:max_chars].strip("_") or "paper"


def first_author_slug(authors: list[str]) -> str:
    if not authors:
        return "unknown"
    parts = authors[0].split(",")[0].split()
    first = parts[-1] if parts else "unknown"
    return safe_slug(first, max_chars=24)


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict) and isinstance(value.get("papers"), list):
        return [item for item in value["papers"] if isinstance(item, dict)]
    return []


def load_local_paper_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_file():
        if path.suffix.lower() == ".pdf":
            return [record_from_pdf(path)]
        return load_json_or_jsonl(path)
    records: list[dict[str, Any]] = []
    for metadata_path in sorted(path.glob("**/*")):
        if metadata_path.is_file() and metadata_path.suffix.lower() in {".json", ".jsonl"}:
            records.extend(load_json_or_jsonl(metadata_path))
    metadata_by_stem = {Path(str(record.get("local_pdf_path", ""))).stem: record for record in records if record.get("local_pdf_path")}
    for pdf_path in sorted(path.glob("**/*.pdf")):
        if pdf_path.stem in metadata_by_stem:
            continue
        records.append(record_from_pdf(pdf_path))
    return records


def record_from_pdf(path: Path) -> dict[str, Any]:
    parsed = parse_pdf_filename(path)
    return {
        "paper_id": path.stem,
        "title": parsed["title"],
        "authors": [parsed["first_author"]] if parsed["first_author"] else [],
        "year": parsed["year"],
        "venue": parsed["venue"],
        "abstract": "",
        "source": "local_pdf",
        "paper_url": "",
        "pdf_url": "",
        "code_url": "",
        "status": "found",
        "local_pdf_path": str(path),
        "retrieval_score": 0.0,
    }


def parse_pdf_filename(path: Path) -> dict[str, Any]:
    stem = path.stem
    parts = stem.split("_")
    year = 1900
    venue = "local"
    first_author = ""
    title_parts = parts
    if parts and parts[0].isdigit() and len(parts[0]) == 4:
        year = int(parts[0])
        title_parts = parts[1:]
    if len(title_parts) >= 2:
        venue = title_parts[0]
        first_author = title_parts[1]
        title_parts = title_parts[2:]
    title = " ".join(title_parts).strip() or stem.replace("_", " ")
    return {
        "year": year,
        "venue": venue,
        "first_author": first_author,
        "title": title.replace("_", " ").strip().title(),
    }


def normalize_candidate(raw: dict[str, Any], *, query: str, source: str) -> dict[str, Any]:
    title = str(raw.get("title") or "").strip()
    paper_url = str(raw.get("paper_url") or raw.get("url") or "").strip()
    paper_id = str(raw.get("paper_id") or raw.get("id") or paper_url or safe_slug(title)).strip()
    year = raw.get("year") or 1900
    try:
        year = int(year)
    except (TypeError, ValueError):
        year = 1900
    authors = _normalize_authors(raw.get("authors") or [])
    return {
        "paper_id": paper_id,
        "title": title or "Untitled paper",
        "authors": authors,
        "year": year,
        "venue": str(raw.get("venue") or ""),
        "abstract": str(raw.get("abstract") or raw.get("summary") or ""),
        "source": str(raw.get("source") or source),
        "paper_url": paper_url,
        "pdf_url": str(raw.get("pdf_url") or ""),
        "code_url": str(raw.get("code_url") or ""),
        "query_source": str(raw.get("query_source") or query),
        "status": str(raw.get("status") or "found"),
        "local_pdf_path": str(raw.get("local_pdf_path") or ""),
        "retrieval_score": float(raw.get("retrieval_score") or 0.0),
    }


def _normalize_authors(authors: Any) -> list[str]:
    if isinstance(authors, str):
        return [item.strip() for item in authors.split(",") if item.strip()]
    if not isinstance(authors, list):
        return []
    normalized: list[str] = []
    for item in authors:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(
                item.get("fullname")
                or item.get("full_name")
                or item.get("name")
                or item.get("username")
                or ""
            ).strip()
        else:
            name = str(item).strip()
        if name:
            normalized.append(name)
    return normalized


def extract_json_array(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        decoder = JSONDecoder()
        for index, char in enumerate(raw):
            if char != "[":
                continue
            try:
                value, _ = decoder.raw_decode(raw[index:])
            except json.JSONDecodeError:
                continue
            return value
    return None


def _xml_text(element: ET.Element | None) -> str:
    return "" if element is None or element.text is None else element.text.strip()


def _content_value(value: object) -> object:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value or ""


def _year_from_text(value: str) -> int:
    for token in value.replace("/", " ").replace(".", " ").split():
        if token.isdigit() and len(token) == 4:
            return int(token)
    return 1900


def _venue_from_invitation(value: str) -> str:
    parts = value.replace("/", " ").replace(".", " ").split()
    for part in parts:
        upper = part.upper()
        if upper in {"ICLR", "ICML", "NEURIPS", "CVPR", "ACL"}:
            return upper
    return "OpenReview"
