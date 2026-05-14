from __future__ import annotations

from quit_agent.schemas.evidence_card import EvidenceCard
from quit_agent.schemas.idea_card import IdeaCard, IdeaGenerationReport


class IdeaValidator:
    def validate(self, ideas: list[IdeaCard], evidence: list[EvidenceCard] | None = None) -> IdeaGenerationReport:
        errors = []
        evidence_ids = {card.evidence_id for card in evidence or []}
        for idea in ideas:
            if not idea.supporting_evidence_ids:
                errors.append(f"{idea.idea_id}: missing supporting evidence")
            if evidence_ids:
                missing = [item for item in idea.supporting_evidence_ids if item not in evidence_ids]
                if missing:
                    errors.append(f"{idea.idea_id}: unknown supporting evidence ids: {missing}")
        return IdeaGenerationReport(status="PASS" if ideas and not errors else "FAIL", idea_count=len(ideas), errors=errors)
