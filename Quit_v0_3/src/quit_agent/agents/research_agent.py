from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.evidence_card import EvidenceCard
from quit_agent.schemas.idea_card import IdeaCard
from quit_agent.schemas.paper_card import PaperCard
from quit_agent.schemas.paper_text import PaperSections, PaperText
from quit_agent.schemas.research_brief import ResearchBrief
from quit_agent.schemas.review_artifacts import IdeaDecision
from quit_agent.tools.llm_interface import LLMClient
from quit_agent.tools.paper_reader import LocalPdfTextExtractor, PaperTextExtractor
from quit_agent.tools.repo_tools import RepoManager
from quit_agent.tools.retrievers import PdfDownloader, extract_json_array
from quit_agent.tools.retriever_interface import Retriever
from quit_agent.validators.evidence_validator import EvidenceValidator
from quit_agent.validators.idea_validator import IdeaValidator


class ResearchAgent:
    def __init__(
        self,
        retriever: Retriever,
        artifacts: ArtifactManager,
        downloader: PdfDownloader | None = None,
        query_llm: LLMClient | None = None,
        read_llm: LLMClient | None = None,
        read_mode: str = "local_text",
        text_extractor: PaperTextExtractor | None = None,
        max_downloads: int = 20,
        min_downloads: int = 0,
        per_source_results: int = 8,
        relevance_selection_ratio: float = 0.7,
        repo_manager: RepoManager | None = None,
        max_direct_pdf_mb: float = 10.0,
        read_workers: int = 4,
    ) -> None:
        self.retriever = retriever
        self.artifacts = artifacts
        self.downloader = downloader
        self.query_llm = query_llm
        self.read_llm = read_llm
        self.read_mode = read_mode if read_mode in {"local_text", "direct_pdf"} else "local_text"
        self.text_extractor = text_extractor or LocalPdfTextExtractor()
        self.max_downloads = max(0, max_downloads)
        self.min_downloads = max(0, min_downloads)
        self.per_source_results = max(1, per_source_results)
        self.relevance_selection_ratio = min(1.0, max(0.0, relevance_selection_ratio))
        self.repo_manager = repo_manager
        self.max_direct_pdf_bytes = int(max_direct_pdf_mb * 1024 * 1024)
        self.read_workers = max(1, read_workers)

    def retrieve(self, brief: ResearchBrief) -> tuple[list[PaperCard], dict[str, Any], str, str]:
        """Skill: budgeted_retrieve_and_filter.

        Use when: ResearchBrief is valid and READ needs bounded paper inputs.
        Inputs: ResearchBrief.json.
        Output: PaperCards.jsonl.
        Failure mode: empty selected set is recorded and routed by orchestrator.
        Validation: candidate ranking, top-k truncation, and budget control happen here.
        """
        prompt = f"Retrieve papers for topic={brief.topic!r}, domains={brief.domain!r}"
        prev_report = self.artifacts.read_json("RetrievalReport.json") if self.artifacts.path("RetrievalReport.json").exists() else {}
        download_fail_count = int(prev_report.get("download_fail_count", 0))
        if prev_report.get("download_fail"):
            download_fail_count += 1
        else:
            download_fail_count = 0
        queries = self._queries(brief, download_fail_count=download_fail_count)[: brief.search_budget.max_queries]
        candidates: dict[str, PaperCard] = {}
        screened = 0
        no_new_signal_rounds = 0
        query_reports = []
        for query in queries:
            if screened >= brief.search_budget.max_papers_screened:
                break
            before = len(candidates)
            remaining = brief.search_budget.max_papers_screened - screened
            found_this_query = 0
            for raw in self.retriever.search(query, max_results=min(remaining, self.per_source_results)):
                screened += 1
                card = PaperCard.model_validate(raw)
                card.retrieval_score = self._score(card, brief)
                key = self._dedupe_key(card)
                if key not in candidates or card.retrieval_score > candidates[key].retrieval_score:
                    candidates[key] = card
                    found_this_query += 1
                if screened >= brief.search_budget.max_papers_screened:
                    break
            query_reports.append({"query": query, "new_unique": found_this_query, "screened_total": screened})
            if len(candidates) == before:
                no_new_signal_rounds += 1
            else:
                no_new_signal_rounds = 0
            if candidates and no_new_signal_rounds >= brief.search_budget.stop_if_no_new_signal_rounds:
                break
        selected = self._select_hybrid(
            list(candidates.values()),
            limit=brief.search_budget.max_papers_selected,
        )
        download_attempted = 0
        download_succeeded = 0
        if self.downloader is not None:
            downloaded = []
            for card in selected:
                should_download = bool(card.pdf_url) and download_attempted < self.max_downloads
                if should_download:
                    download_attempted += 1
                    card = self.downloader.download(card)
                    if card.local_pdf_path and str(card.status) != "failed":
                        download_succeeded += 1
                downloaded.append(card)
            selected = downloaded
        repo_report = {
            "status": "PASS",
            "repo_count": 0,
            "max_repos": brief.search_budget.max_repo_checked,
            "clone_stage": "BUILD_SPEC",
            "clone_attempted": 0,
            "clone_succeeded": 0,
            "env_file_count": 0,
        }
        repos = []
        if self.repo_manager is not None and brief.search_budget.max_repo_checked > 0:
            repos, repo_report = self.repo_manager.collect_from_papers(
                selected,
                max_repos=brief.search_budget.max_repo_checked,
                run_id=self.artifacts.paths.run_id,
            )
        minimum_selected = self._minimum_selected(brief.search_budget.max_papers_selected)
        download_fail = (
            self.downloader is not None
            and self.min_downloads > 0
            and len(candidates) > 0
            and download_succeeded < self.min_downloads
        )
        search_empty = len(candidates) == 0
        retrieval_status = "PASS" if (len(selected) >= minimum_selected and not download_fail) else "FAIL"
        self.artifacts.write_jsonl("PaperCards.jsonl", selected)
        self.artifacts.write_jsonl("RepoCards.jsonl", repos)
        self.artifacts.write_json("RepoRetrievalReport.json", repo_report)
        report = {
            "status": retrieval_status,
            "queries": queries,
            "query_reports": query_reports,
            "screened": screened,
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "minimum_selected": minimum_selected,
            "source_counts": self._source_counts(list(candidates.values())),
            "selected_source_counts": self._source_counts(selected),
            "download_pdfs": self.downloader is not None,
            "max_downloads": self.max_downloads,
            "min_downloads": self.min_downloads,
            "search_empty": search_empty,
            "download_fail": download_fail,
            "download_fail_count": download_fail_count,
            "selection_policy": {
                "mode": "hybrid_relevance_diversity",
                "relevance_selection_ratio": self.relevance_selection_ratio,
            },
            "download_attempted": download_attempted,
            "download_succeeded": download_succeeded,
            "repo_retrieval": repo_report,
            "budget": brief.search_budget.to_dict(),
            "stop_reason": self._stop_reason(
                screened=screened,
                selected=len(selected),
                query_count=len(query_reports),
                no_new_signal_rounds=no_new_signal_rounds,
                brief=brief,
            ),
        }
        self.artifacts.write_json("RetrievalReport.json", report)
        if retrieval_status != "PASS":
            self.artifacts.write_json("RetrievalFailure.json", report)
        return selected, report, prompt, repr(report)

    def read(self, brief: ResearchBrief, papers: list[PaperCard]) -> tuple[list[EvidenceCard], dict[str, Any], str, str]:
        """Skill: extract_structured_evidence.

        Use when: bounded PaperCards are ready for evidence extraction.
        Inputs: ResearchBrief.json and PaperCards.jsonl.
        Output: EvidenceCards.jsonl and EvidenceValidationReport.json.
        Failure mode: missing evidence is written as a validation report.
        """
        prompt = f"Read {len(papers)} selected papers and extract structured evidence for topic={brief.topic!r}"
        direct_pdf_attempted = 0
        direct_pdf_succeeded = 0
        direct_pdf_fallbacks: list[str] = []

        def _read_one(idx_paper: tuple[int, PaperCard]) -> tuple[int, PaperCard, EvidenceCard, PaperText, str | None]:
            """Process a single paper; returns (idx, paper, card, paper_text, failure_kind).
            failure_kind: None | "extraction" | "llm" | "direct_pdf_fallback"
            """
            idx, paper = idx_paper
            card = None
            failure_kind = None
            if self.read_mode == "direct_pdf":
                # Skip direct_pdf for oversized files to avoid slow uploads
                pdf_path = paper.local_pdf_path
                too_large = (
                    pdf_path
                    and Path(pdf_path).exists()
                    and Path(pdf_path).stat().st_size > self.max_direct_pdf_bytes
                )
                if not too_large:
                    card = self._direct_pdf_evidence(idx, brief, paper)
                if card is not None:
                    paper_text = self._direct_pdf_paper_text(paper)
                else:
                    failure_kind = "direct_pdf_fallback"
                    paper_text = self.text_extractor.extract(paper)
            else:
                paper_text = self.text_extractor.extract(paper)

            if paper_text.extraction_status == "failed" and failure_kind is None:
                failure_kind = "extraction"
            if card is None:
                card = self._llm_evidence(idx, brief, paper, paper_text)
            if card is None:
                if self.read_llm is not None and failure_kind is None:
                    failure_kind = "llm"
                card = self._fallback_evidence(idx, brief, paper, paper_text)
            return idx, paper, card, paper_text, failure_kind

        indexed_papers = list(enumerate(papers, start=1))
        results: list[tuple[int, PaperCard, EvidenceCard, PaperText, str | None]] = [None] * len(papers)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=self.read_workers) as pool:
            futures = {pool.submit(_read_one, ip): ip[0] - 1 for ip in indexed_papers}
            for future in as_completed(futures):
                slot = futures[future]
                try:
                    results[slot] = future.result()
                except Exception as exc:
                    idx, paper = indexed_papers[slot]
                    paper_text = self.text_extractor.extract(paper)
                    card = self._fallback_evidence(idx, brief, paper, paper_text)
                    results[slot] = (idx, paper, card, paper_text, f"exception:{exc}")

        paper_texts: list[PaperText] = []
        evidence: list[EvidenceCard] = []
        extraction_failures: list[str] = []
        llm_failures: list[str] = []
        for idx, paper, card, paper_text, failure_kind in results:
            paper_texts.append(paper_text)
            evidence.append(card)
            if failure_kind == "extraction":
                extraction_failures.append(paper.paper_id)
            elif failure_kind == "llm":
                llm_failures.append(paper.paper_id)
            elif failure_kind == "direct_pdf_fallback":
                direct_pdf_fallbacks.append(paper.paper_id)
            if self.read_mode == "direct_pdf":
                direct_pdf_attempted += 1
                if failure_kind != "direct_pdf_fallback":
                    direct_pdf_succeeded += 1
        report = EvidenceValidator().validate(papers, evidence)
        read_report = {
            "status": report.status,
            "paper_count": len(papers),
            "paper_text_count": len(paper_texts),
            "evidence_count": len(evidence),
            "text_extraction": {
                "success": sum(1 for item in paper_texts if item.extraction_status == "success"),
                "partial": sum(1 for item in paper_texts if item.extraction_status == "partial"),
                "failed": sum(1 for item in paper_texts if item.extraction_status == "failed"),
                "failed_paper_ids": extraction_failures,
            },
            "read_mode": self.read_mode,
            "direct_pdf": {
                "attempted": direct_pdf_attempted,
                "succeeded": direct_pdf_succeeded,
                "fallback_paper_ids": direct_pdf_fallbacks,
                "provider_supports_files": self._supports_direct_pdf_reading(),
            },
            "llm_reader_used": self.read_llm is not None,
            "llm_failed_paper_ids": llm_failures,
            "validation": report.to_dict(),
            "fallback_target": None if report.status == "PASS" else "RETRIEVE",
        }
        self.artifacts.write_jsonl("PaperTexts.jsonl", paper_texts)
        self.artifacts.write_jsonl("EvidenceCards.jsonl", evidence)
        self.artifacts.write_json("EvidenceValidationReport.json", report)
        self.artifacts.write_json("ReadReport.json", read_report)
        return evidence, read_report, prompt, repr(read_report)

    def ideate(
        self,
        brief: ResearchBrief,
        evidence: list[EvidenceCard],
        revision_decision: IdeaDecision | None = None,
    ) -> tuple[list[IdeaCard], dict[str, Any], str, str]:
        """Skill: evidence_grounded_ideation.

        Use when: evidence cards are available and candidate ideas are needed.
        Inputs: ResearchBrief.json and EvidenceCards.jsonl.
        Output: IdeaLibrary.jsonl and IdeaGenerationReport.json.
        Failure mode: empty or unsupported ideas are recorded by IdeaGenerationReport.
        """
        clusters = self._cluster_evidence(brief, evidence)
        prompt = self._ideation_prompt(brief, clusters, revision_decision)
        ideas = self._llm_ideas(clusters, prompt)
        used_fallback = False
        if not ideas:
            used_fallback = True
            ideas = self._fallback_ideas(brief, clusters)

        ideas = self._rank_ideas(ideas, evidence)
        report = IdeaValidator().validate(ideas, evidence)
        generation_report = {
            **report.to_dict(),
            "cluster_count": len(clusters),
            "llm_used": self.read_llm is not None,
            "fallback_used": used_fallback,
            "evidence_count": len(evidence),
            "top_idea_ids": [idea.idea_id for idea in ideas[:3]],
        }
        self.artifacts.write_json("IdeaClusters.json", {"clusters": [self._cluster_preview(cluster) for cluster in clusters]})
        self.artifacts.write_jsonl("IdeaLibrary.jsonl", ideas)
        self.artifacts.write_json("IdeaGenerationReport.json", generation_report)
        return ideas, generation_report, prompt, repr(generation_report)

    def _queries(self, brief: ResearchBrief, *, download_fail_count: int = 0) -> list[str]:
        if self.query_llm is not None:
            planned = self._llm_queries(brief)
            if planned:
                return planned
        topic = brief.topic
        keywords = [kw.strip() for kw in (brief.search_keywords or []) if kw.strip()]
        candidates = [
            topic,
            *keywords[:3],
            f"most influential {topic}".strip(),
            f"latest {topic}".strip(),
            f"{topic} survey benchmark".strip(),
        ]
        # After repeated download failures, broaden with keyword freshness variants.
        if download_fail_count >= 2:
            for kw in keywords[:3]:
                candidates.append(f"latest {kw}")

        seen = set()
        queries = []
        for query in candidates:
            normalized = " ".join(query.split())
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                queries.append(normalized)
        return queries

    def _cluster_evidence(self, brief: ResearchBrief, evidence: list[EvidenceCard]) -> list[dict[str, Any]]:
        buckets: dict[str, list[EvidenceCard]] = defaultdict(list)
        for card in evidence:
            buckets[self._cluster_key(brief, card)].append(card)
        return [
            {"cluster_id": f"cluster-{idx}", "theme": key, "evidence": cards}
            for idx, (key, cards) in enumerate(
                sorted(buckets.items(), key=lambda item: len(item[1]), reverse=True),
                start=1,
            )
        ]

    def _cluster_key(self, brief: ResearchBrief, card: EvidenceCard) -> str:
        text = " ".join(
            [
                brief.topic,
                " ".join(brief.domain),
                card.task,
                card.method,
                card.setting,
                " ".join(card.claims),
                " ".join(card.transferable_idea_seeds),
            ]
        )
        phrases = self._top_phrases([card.task, card.method, *card.transferable_idea_seeds], limit=2)
        if phrases:
            return "_".join(self._theme_token(phrase) for phrase in phrases if self._theme_token(phrase))[:80] or "evidence_grounded_theme"
        tokens = self._theme_terms(text, limit=3)
        return "_".join(tokens) if tokens else "evidence_grounded_theme"

    def _load_prior_extracts(self, topic: str) -> list[dict[str, Any]]:
        import re as _re
        normalized = _re.sub(r"[^\w\s-]", "", topic.lower()).strip()
        normalized = _re.sub(r"[\s_-]+", "_", normalized)[:80]
        if not normalized:
            return []
        extracts_file = self.artifacts.paths.extracts_dir / f"{normalized}.json"
        if not extracts_file.exists():
            return []
        try:
            data = json.loads(extracts_file.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _ideation_prompt(
        self,
        brief: ResearchBrief,
        clusters: list[dict[str, Any]],
        revision_decision: IdeaDecision | None = None,
    ) -> str:
        cluster_text = []
        for cluster in clusters[:8]:
            evidence_lines = []
            for card in cluster["evidence"][:5]:
                evidence_lines.append(
                    "\n".join(
                        [
                            f"- evidence_id: {card.evidence_id}",
                            f"  paper_id: {card.paper_id}",
                            f"  task: {card.task}",
                            f"  method: {card.method}",
                            f"  claims: {card.claims[:3]}",
                            f"  metrics: {card.metrics[:5]}",
                            f"  limitations: {[item for item in card.limitations[:3] if self._useful_limitation(item)]}",
                            f"  seeds: {card.transferable_idea_seeds[:3]}",
                        ]
                    )
                )
            cluster_text.append(
                f"Cluster {cluster['cluster_id']} theme={cluster['theme']}\n" + "\n".join(evidence_lines)
            )
        revision_text = ""
        if revision_decision is not None:
            revision_text = f"""

Previous IDEA_EVAL feedback to fix:
- reviewed_idea_id: {revision_decision.idea_id}
- reason: {revision_decision.reason}
- violations: {revision_decision.violations}
- required_changes: {revision_decision.required_changes}

Revision rules:
- Directly address every required_change.
- Do not repeat rejected target-task wording, placeholder text, or metric claims.
- Keep the new ideas aligned with the ResearchBrief topic and acceptance_criteria.
"""

        prior_extracts = self._load_prior_extracts(brief.topic)
        prior_text = ""
        if prior_extracts:
            lines = []
            for i, entry in enumerate(prior_extracts, start=1):
                lines.append(
                    f"[Prior run {i}]\n"
                    f"  contribution: {entry.get('contribution', '')}\n"
                    f"  method_novelty: {entry.get('method_novelty', '')}\n"
                    f"  key_results: {entry.get('key_results', '')}\n"
                    f"  limitations: {entry.get('limitations', [])}\n"
                    f"  future_directions: {entry.get('future_directions', [])}"
                )
            prior_text = "\n\nPrior research on this topic (avoid repeating; build on or address limitations):\n" + "\n".join(lines)

        return f"""\
You are generating candidate research ideas from structured evidence.

ResearchBrief:
- topic: {brief.topic}
- domain: {brief.domain}
- constraints: {brief.constraints}
- acceptance_criteria: {brief.acceptance_criteria}
- red_lines: {brief.red_lines}

Rules:
- Return a JSON array of 3 to 6 idea objects.
- Each idea must be grounded in supporting_evidence_ids from the evidence below.
- Do not invent papers or evidence ids.
- Do not propose ideas that violate red_lines.
- Prefer ideas that combine method seeds, limitations, and metrics across multiple papers.
- Target tasks must stay within the ResearchBrief topic/domain; do not append unrelated benchmark or policy-optimization suffixes.
- Expected gains must cite acceptance_criteria metrics when they are available.
- Return JSON only, no markdown.
{revision_text}{prior_text}

Idea object schema:
{{
  "idea_id": "idea-1",
  "target_task": "specific target task",
  "novelty_claim": "core innovation claim",
  "supporting_evidence_ids": ["ev-1", "ev-2"],
  "expected_gain": "expected metric or research benefit"
}}

Evidence clusters:
{chr(10).join(cluster_text)}
"""

    def _llm_ideas(
        self,
        clusters: list[dict[str, Any]],
        prompt: str,
    ) -> list[IdeaCard]:
        if self.read_llm is None:
            return []
        raw = self.read_llm.complete(prompt)
        value = extract_json_array(raw)
        if not isinstance(value, list):
            self.artifacts.write_markdown("IdeaGeneration.raw.txt", raw)
            return []
        valid_ids = {card.evidence_id for cluster in clusters for card in cluster["evidence"]}
        ideas = []
        for idx, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                continue
            item.setdefault("idea_id", f"idea-{idx}")
            support = [str(eid) for eid in item.get("supporting_evidence_ids", []) if str(eid) in valid_ids]
            item["supporting_evidence_ids"] = support
            item.setdefault("priority_score", float(len(support)))
            try:
                ideas.append(IdeaCard.model_validate(item))
            except Exception:
                continue
        if not ideas:
            self.artifacts.write_markdown("IdeaGeneration.raw.txt", raw)
        return ideas

    def _fallback_ideas(self, brief: ResearchBrief, clusters: list[dict[str, Any]]) -> list[IdeaCard]:
        ideas = []
        for idx, cluster in enumerate(clusters[:6], start=1):
            cards: list[EvidenceCard] = cluster["evidence"]
            support = [card.evidence_id for card in cards[:4]]
            methods = self._top_phrases([card.method for card in cards], limit=2)
            limitations = self._top_phrases(
                [item for card in cards for item in card.limitations if self._useful_limitation(item)],
                limit=2,
            )
            metrics = self._brief_metrics(brief) or sorted({metric for card in cards for metric in card.metrics})[:4]
            method_text = " + ".join(methods) if methods else cluster["theme"].replace("_", " ")
            limitation_text = "; ".join(limitations) if limitations else self._brief_constraint_summary(brief)
            metric_text = ", ".join(metrics) if metrics else "task quality and robustness metrics"
            ideas.append(
                IdeaCard(
                    idea_id=f"idea-{idx}",
                    target_task=self._fallback_target_task(brief, cluster["theme"]),
                    novelty_claim=(
                        f"Combine {method_text} to address {limitation_text} in a constrained "
                        f"{brief.topic} setting."
                    ),
                    supporting_evidence_ids=support,
                    expected_gain=f"Expected improvement in {metric_text} through evidence-backed method recombination.",
                    priority_score=float(len(support)),
                )
            )
        return ideas

    def _rank_ideas(self, ideas: list[IdeaCard], evidence: list[EvidenceCard]) -> list[IdeaCard]:
        evidence_by_id = {card.evidence_id: card for card in evidence}
        ranked = []
        for idea in ideas:
            metrics = {
                metric
                for evidence_id in idea.supporting_evidence_ids
                for metric in evidence_by_id.get(evidence_id, EvidenceCard(evidence_id="x", paper_id="x", task="x", method="x")).metrics
            }
            idea.priority_score = round(float(len(idea.supporting_evidence_ids)) + len(metrics) * 0.25, 4)
            ranked.append(idea)
        return sorted(ranked, key=lambda item: item.priority_score, reverse=True)

    def _top_phrases(self, phrases: list[str], *, limit: int) -> list[str]:
        seen = set()
        result = []
        for phrase in phrases:
            cleaned = self._short_text(phrase, fallback="", max_chars=120)
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                result.append(cleaned)
            if len(result) >= limit:
                break
        return result

    def _theme_terms(self, text: str, *, limit: int) -> list[str]:
        stopwords = {
            "and",
            "for",
            "from",
            "with",
            "that",
            "this",
            "into",
            "under",
            "using",
            "via",
            "the",
            "model",
            "method",
            "approach",
            "task",
            "paper",
        }
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", text.lower())
        seen = set()
        result = []
        for token in tokens:
            if token in stopwords or token in seen:
                continue
            seen.add(token)
            result.append(token)
            if len(result) >= limit:
                break
        return result

    def _theme_token(self, phrase: str) -> str:
        return "_".join(self._theme_terms(phrase, limit=4))

    def _useful_limitation(self, text: str) -> bool:
        cleaned = text.strip().lower()
        if not cleaned:
            return False
        rejected = [
            "no explicit limitations extracted",
            "no limitations extracted",
            "not available",
            "n/a",
            "none",
        ]
        return not any(marker in cleaned for marker in rejected)

    def _brief_metrics(self, brief: ResearchBrief) -> list[str]:
        candidates = [*brief.acceptance_criteria, *brief.constraints]
        metrics = []
        for item in candidates:
            text = item.strip()
            if not text:
                continue
            if any(char.isdigit() for char in text) or any(
                keyword in text.lower()
                for keyword in ["metric", "score", "latency", "lpips", "quality", "consistency", "accuracy", "error"]
            ):
                metrics.append(self._short_text(text, fallback="", max_chars=80))
            if len(metrics) >= 4:
                break
        return metrics

    def _brief_constraint_summary(self, brief: ResearchBrief) -> str:
        useful = [self._short_text(item, fallback="", max_chars=100) for item in brief.constraints[:2]]
        useful = [item for item in useful if item]
        return "; ".join(useful) if useful else "the brief's domain constraints"

    def _fallback_target_task(self, brief: ResearchBrief, theme: str) -> str:
        theme_text = theme.replace("_", " ").strip()
        topic = brief.topic.strip()
        if not theme_text or theme_text.lower() in topic.lower():
            return topic
        return f"{topic} with {theme_text}"

    def _cluster_preview(self, cluster: dict[str, Any]) -> dict[str, Any]:
        cards: list[EvidenceCard] = cluster["evidence"]
        return {
            "cluster_id": cluster["cluster_id"],
            "theme": cluster["theme"],
            "evidence_ids": [card.evidence_id for card in cards],
            "paper_ids": [card.paper_id for card in cards],
            "size": len(cards),
        }

    def _llm_queries(self, brief: ResearchBrief) -> list[str]:
        prompt = f"""\
Generate search queries for paper retrieval.

Topic: {brief.topic}
Domain: {brief.domain}
Constraints: {brief.constraints}

Return a JSON array of concise search query strings only.
Include a mix of:
- most relevant
- most influential
- latest
- benchmark/survey direction
Limit to {brief.search_budget.max_queries} queries.
"""
        raw = self.query_llm.complete(prompt)
        value = extract_json_array(raw)
        if not isinstance(value, list):
            self.artifacts.write_markdown("RetrieveQueryPlan.raw.txt", raw)
            return []
        queries = []
        seen = set()
        for item in value:
            query = " ".join(str(item).split())
            key = query.lower()
            if query and key not in seen:
                seen.add(key)
                queries.append(query)
        self.artifacts.write_json("RetrieveQueryPlan.json", {"queries": queries, "raw": raw})
        return queries[: brief.search_budget.max_queries]

    def _score(self, paper: PaperCard, brief: ResearchBrief) -> float:
        haystack = self._tokens(" ".join([paper.title, paper.abstract]))
        needles = self._tokens(" ".join([brief.topic, *brief.domain, *brief.constraints]))
        overlap = len(haystack & needles)
        return round(float(paper.retrieval_score) + overlap * 0.25, 4)

    def _direct_pdf_evidence(
        self,
        index: int,
        brief: ResearchBrief,
        paper: PaperCard,
    ) -> EvidenceCard | None:
        if self.read_llm is None or not self._supports_direct_pdf_reading() or not paper.local_pdf_path:
            return None
        if not Path(paper.local_pdf_path).exists():
            return None
        prompt = self._direct_pdf_prompt(brief, paper)
        try:
            raw = self.read_llm.complete_with_files(prompt, [paper.local_pdf_path])  # type: ignore[attr-defined]
        except Exception:
            return None
        value = self._extract_json_object(raw)
        if not isinstance(value, dict):
            self.artifacts.write_markdown(f"read_raw/{paper.paper_id}.txt", raw)
            return None
        value.setdefault("evidence_id", f"ev-{index}")
        value.setdefault("paper_id", paper.paper_id)
        try:
            return EvidenceCard.model_validate(value)
        except Exception:
            self.artifacts.write_markdown(f"read_raw/{paper.paper_id}.txt", raw)
            return None

    def _supports_direct_pdf_reading(self) -> bool:
        return self.read_llm is not None and callable(getattr(self.read_llm, "complete_with_files", None))

    def _direct_pdf_prompt(self, brief: ResearchBrief, paper: PaperCard) -> str:
        return f"""\
You are reading an attached PDF paper for an artifact-driven research workflow.

Research topic:
{brief.topic}

Paper metadata:
- paper_id: {paper.paper_id}
- title: {paper.title}
- venue: {paper.venue}
- year: {paper.year}

Return exactly one JSON object matching this schema:
{{
  "evidence_id": "ev-id",
  "paper_id": "{paper.paper_id}",
  "task": "specific task or problem",
  "method": "main method, module, training strategy, or algorithm",
  "setting": "datasets, task setup, baselines, or evaluation protocol",
  "claims": ["main claim", "experimental sub-claim"],
  "metrics": ["metric names"],
  "limitations": ["limitations or risks"],
  "transferable_idea_seeds": ["reusable module, training strategy, evaluation idea, or inductive bias"]
}}

Do not summarize freely. Extract evidence only.
"""

    def _direct_pdf_paper_text(self, paper: PaperCard) -> PaperText:
        return PaperText(
            paper_id=paper.paper_id,
            title=paper.title,
            abstract=paper.abstract,
            full_text="",
            sections=PaperSections(abstract=paper.abstract),
            source_path=paper.local_pdf_path,
            extraction_status="partial",
            errors=["direct_pdf mode used; full text was not extracted locally"],
        )

    def _llm_evidence(
        self,
        index: int,
        brief: ResearchBrief,
        paper: PaperCard,
        paper_text: PaperText,
    ) -> EvidenceCard | None:
        if self.read_llm is None:
            return None
        prompt = self._read_prompt(brief, paper, paper_text)
        raw = self.read_llm.complete(prompt)
        value = self._extract_json_object(raw)
        if not isinstance(value, dict):
            self.artifacts.write_markdown(f"read_raw/{paper.paper_id}.txt", raw)
            return None
        value.setdefault("evidence_id", f"ev-{index}")
        value.setdefault("paper_id", paper.paper_id)
        try:
            return EvidenceCard.model_validate(value)
        except Exception:
            self.artifacts.write_markdown(f"read_raw/{paper.paper_id}.txt", raw)
            return None

    def _read_prompt(self, brief: ResearchBrief, paper: PaperCard, paper_text: PaperText) -> str:
        sections = paper_text.sections
        text = "\n\n".join(
            [
                f"Title: {paper.title}",
                f"Abstract:\n{paper_text.abstract}",
                f"Introduction:\n{sections.introduction[:4000]}",
                f"Method:\n{sections.method[:5000]}",
                f"Experiments:\n{sections.experiments[:5000]}",
                f"Limitations:\n{sections.limitations[:3000]}",
                f"Conclusion:\n{sections.conclusion[:3000]}",
                f"Fallback full text excerpt:\n{paper_text.full_text[:7000]}",
            ]
        )
        return f"""\
You are reading a research paper for an artifact-driven research workflow.

Research topic:
{brief.topic}

Return exactly one JSON object matching this schema:
{{
  "evidence_id": "ev-id",
  "paper_id": "{paper.paper_id}",
  "task": "specific task or problem",
  "method": "main method, module, training strategy, or algorithm",
  "setting": "datasets, task setup, baselines, or evaluation protocol",
  "claims": ["main claim", "experimental sub-claim"],
  "metrics": ["metric names"],
  "limitations": ["limitations or risks"],
  "transferable_idea_seeds": ["reusable module, training strategy, evaluation idea, or inductive bias"]
}}

Do not summarize freely. Extract evidence only. Use empty lists when the paper text does not support a field.

Paper text:
{text}
"""

    def _fallback_evidence(
        self,
        index: int,
        brief: ResearchBrief,
        paper: PaperCard,
        paper_text: PaperText,
    ) -> EvidenceCard:
        sections = paper_text.sections
        claims = self._sentences(
            " ".join([paper_text.abstract, sections.experiments, sections.conclusion, paper.abstract]),
            limit=4,
        )
        metrics = self._metric_terms(paper_text.full_text)
        limitations = self._sentences(sections.limitations, limit=3) or [
            "No explicit limitations extracted from available text."
        ]
        seeds = self._idea_seeds(paper, paper_text)
        method_text = sections.method or paper.title
        setting_text = sections.experiments or paper.venue or paper.source
        return EvidenceCard(
            evidence_id=f"ev-{index}",
            paper_id=paper.paper_id,
            task=self._short_text(sections.introduction or paper_text.abstract or brief.topic, fallback=brief.topic),
            method=self._short_text(method_text, fallback=paper.title),
            setting=self._short_text(setting_text, fallback=paper.venue or paper.source),
            claims=claims or [paper.abstract[:240] or f"{paper.title} is relevant to {brief.topic}."],
            metrics=metrics,
            limitations=limitations,
            transferable_idea_seeds=seeds,
        )

    def _extract_json_object(self, raw: str) -> Any:
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(raw):
                if char != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(raw[index:])
                except json.JSONDecodeError:
                    continue
                return value if isinstance(value, dict) else None
        return None

    def _sentences(self, text: str, *, limit: int) -> list[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        return [self._short_text(part, fallback="") for part in parts if len(part) > 20][:limit]

    def _metric_terms(self, text: str) -> list[str]:
        patterns = [
            r"\baccuracy\b",
            r"\bsuccess rate\b",
            r"\breward\b",
            r"\breturn\b",
            r"\bcost\b",
            r"\bf1\b",
            r"\bprecision\b",
            r"\brecall\b",
            r"\bwall time\b",
            r"\bsample efficiency\b",
            r"\bregret\b",
        ]
        lowered = text.lower()
        found = []
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                found.append(match.group(0))
        return sorted(set(found))

    def _idea_seeds(self, paper: PaperCard, paper_text: PaperText) -> list[str]:
        seeds = []
        method = self._short_text(paper_text.sections.method, fallback="")
        experiments = self._short_text(paper_text.sections.experiments, fallback="")
        if method:
            seeds.append(f"Adapt method component from {paper.title}: {method}")
        if experiments:
            seeds.append(f"Reuse evaluation setup from {paper.title}: {experiments}")
        if not seeds:
            seeds.append(f"Use {paper.title} as a traceable evidence seed.")
        return seeds[:3]

    def _short_text(self, text: str, *, fallback: str, max_chars: int = 360) -> str:
        cleaned = " ".join(str(text or "").split())
        if not cleaned:
            cleaned = fallback
        return cleaned[:max_chars].strip()

    def _tokens(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}

    def _dedupe_key(self, card: PaperCard) -> str:
        return card.paper_url or card.title.lower()

    def _rank_by_relevance(self, cards: list[PaperCard]) -> list[PaperCard]:
        return sorted(cards, key=lambda card: card.retrieval_score, reverse=True)

    def _minimum_selected(self, max_papers_selected: int) -> int:
        if max_papers_selected <= 0:
            return 0
        return max(1, math.ceil(max_papers_selected * 0.2))

    def _select_hybrid(self, cards: list[PaperCard], *, limit: int) -> list[PaperCard]:
        if limit <= 0:
            return []
        ranked = self._rank_by_relevance(cards)
        relevance_count = min(len(ranked), math.ceil(limit * self.relevance_selection_ratio))
        selected = ranked[:relevance_count]
        remaining = [card for card in ranked if card not in selected]
        if len(selected) >= limit:
            return selected[:limit]

        by_source: dict[str, list[PaperCard]] = defaultdict(list)
        for card in remaining:
            by_source[card.source].append(card)

        source_order = sorted(
            by_source,
            key=lambda source: (
                self._source_counts(selected).get(source, 0),
                -by_source[source][0].retrieval_score,
            ),
        )
        offset = 0
        while len(selected) < limit and source_order:
            added = False
            for source in source_order:
                bucket = by_source[source]
                if offset < len(bucket):
                    selected.append(bucket[offset])
                    added = True
                    if len(selected) >= limit:
                        break
            if not added:
                break
            offset += 1
        return selected[:limit]

    def _source_counts(self, cards: list[PaperCard]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for card in cards:
            counts[card.source] = counts.get(card.source, 0) + 1
        return counts

    def _stop_reason(
        self,
        *,
        screened: int,
        selected: int,
        query_count: int,
        no_new_signal_rounds: int,
        brief: ResearchBrief,
    ) -> str:
        if screened >= brief.search_budget.max_papers_screened:
            return "max_papers_screened"
        if selected >= brief.search_budget.max_papers_selected:
            return "max_papers_selected"
        if query_count >= brief.search_budget.max_queries:
            return "max_queries"
        if no_new_signal_rounds >= brief.search_budget.stop_if_no_new_signal_rounds:
            return "no_new_signal_rounds"
        return "queries_exhausted"
