from __future__ import annotations

from quit_agent.schemas.evidence_card import EvidenceCard, EvidenceValidationReport
from quit_agent.schemas.paper_card import PaperCard


class EvidenceValidator:
    def validate(self, papers: list[PaperCard], evidence: list[EvidenceCard]) -> EvidenceValidationReport:
        paper_ids = {paper.paper_id for paper in papers}
        evidence_ids = {card.paper_id for card in evidence}
        missing = sorted(paper_ids - evidence_ids)
        errors = [f"missing evidence for {paper_id}" for paper_id in missing]
        return EvidenceValidationReport(
            status="PASS" if not errors else "FAIL",
            errors=errors,
            missing_paper_ids=missing,
            malformed_count=0,
        )
