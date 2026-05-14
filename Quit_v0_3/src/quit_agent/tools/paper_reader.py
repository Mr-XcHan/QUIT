from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from quit_agent.schemas.paper_card import PaperCard
from quit_agent.schemas.paper_text import PaperSections, PaperText


class PaperTextExtractor(Protocol):
    def extract(self, paper: PaperCard) -> PaperText:
        """Extract normalized text from a selected paper."""


class LocalPdfTextExtractor:
    """Best-effort local PDF text extractor with abstract fallback.

    The implementation intentionally treats PDF parsing as a replaceable tool.
    If pypdf is not installed or the PDF is unavailable, the workflow still
    produces a partial PaperText artifact from PaperCard metadata.
    """

    def __init__(self, *, max_chars: int = 120_000) -> None:
        self.max_chars = max_chars

    def extract(self, paper: PaperCard) -> PaperText:
        errors: list[str] = []
        source_path = paper.local_pdf_path
        full_text = ""

        if source_path:
            path = Path(source_path)
            if path.exists():
                full_text, errors = self._extract_pdf(path)
            else:
                errors.append(f"local_pdf_path does not exist: {source_path}")
        else:
            errors.append("no local_pdf_path available")

        if not full_text:
            full_text = "\n\n".join(part for part in [paper.title, paper.abstract] if part).strip()

        full_text = self._clean(full_text)[: self.max_chars]
        sections = self._sections(full_text, paper.abstract)
        status = "success" if source_path and full_text and not errors else "partial"
        if not full_text:
            status = "failed"
            errors.append("no text extracted")

        return PaperText(
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract or sections.abstract,
            full_text=full_text,
            sections=sections,
            source_path=source_path,
            extraction_status=status,
            errors=errors,
        )

    def _extract_pdf(self, path: Path) -> tuple[str, list[str]]:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError:
            return "", ["pypdf is not installed"]

        try:
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n\n".join(pages).strip(), []
        except Exception as exc:  # PDF parsing errors vary by backend/file.
            return "", [f"pdf extraction failed: {type(exc).__name__}: {exc}"]

    def _clean(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _sections(self, full_text: str, abstract: str) -> PaperSections:
        section_map = {
            "abstract": ["abstract"],
            "introduction": ["introduction", "background"],
            "method": ["method", "methods", "approach", "model", "algorithm"],
            "experiments": ["experiment", "experiments", "evaluation", "results"],
            "limitations": ["limitation", "limitations", "discussion"],
            "conclusion": ["conclusion", "conclusions"],
        }
        found: dict[str, str] = {}
        headings = self._heading_spans(full_text)
        for index, (name, start) in enumerate(headings):
            end = headings[index + 1][1] if index + 1 < len(headings) else len(full_text)
            canonical = self._canonical_heading(name, section_map)
            if canonical and canonical not in found:
                found[canonical] = full_text[start:end].strip()

        return PaperSections(
            abstract=found.get("abstract") or abstract,
            introduction=found.get("introduction", ""),
            method=found.get("method", ""),
            experiments=found.get("experiments", ""),
            limitations=found.get("limitations", ""),
            conclusion=found.get("conclusion", ""),
        )

    def _heading_spans(self, text: str) -> list[tuple[str, int]]:
        pattern = re.compile(
            r"(?im)^(?:\d+(?:\.\d+)*\s+)?"
            r"(abstract|introduction|background|method|methods|approach|model|algorithm|"
            r"experiment|experiments|evaluation|results|limitation|limitations|discussion|"
            r"conclusion|conclusions)\b.*$"
        )
        return [(match.group(1).lower(), match.start()) for match in pattern.finditer(text)]

    def _canonical_heading(self, heading: str, section_map: dict[str, list[str]]) -> str:
        for canonical, aliases in section_map.items():
            if heading in aliases:
                return canonical
        return ""
