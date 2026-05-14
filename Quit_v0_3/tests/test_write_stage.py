from __future__ import annotations

from quit_agent.agents.builder_agent import BuilderAgent
from quit_agent.artifacts.manager import ArtifactManager
from quit_agent.schemas.build_spec import BuildSpec
from quit_agent.schemas.evidence_card import EvidenceCard
from quit_agent.schemas.paper_card import PaperCard


class LatexLLM:
    def complete(self, prompt: str) -> str:
        filler_abstract = " ".join(["abstractword"] * 160)
        filler_intro = " ".join(["introductionword"] * 710)
        filler_related = " ".join(["relatedword"] * 360)
        filler_method = " ".join(["methodword"] * 2010)
        filler_experiments = " ".join(["experimentword"] * 1210)
        filler_conclusion = " ".join(["conclusionword"] * 210)
        return rf"""
\documentclass{{article}}
\usepackage{{icml2026}}
\usepackage{{algorithm}}
\usepackage{{algorithmic}}
\usepackage{{amsmath}}
\begin{{document}}
\begin{{abstract}}
{filler_abstract}
\end{{abstract}}
\section{{Introduction}}
{filler_intro}
\cite{{paper_1,paper_2,paper_3}}
\section{{Related Work}}
{filler_related}
\section{{Method}}
{filler_method}
\begin{{equation}}
\mathcal{{L}}(\theta)=\mathcal{{L}}_{{task}}(\theta)
\end{{equation}}
\begin{{algorithm}}
\caption{{Evidence-guided method}}
\begin{{algorithmic}}
\STATE Train the proposed method.
\end{{algorithmic}}
\end{{algorithm}}
\begin{{proof}}
The claim follows from the stated construction.
\end{{proof}}
\section{{Experiments}}
{filler_experiments}
\section{{Conclusion}}
{filler_conclusion}
\bibliography{{references}}
\bibliographystyle{{icml2026}}
\appendix
\section{{Experimental Details}}
The appendix records dataset construction, environment settings, preprocessing, hyperparameters, seed choices, evaluation protocol, hardware device selection, and artifact paths for reproducibility.
\section{{Theoretical Proofs}}
\begin{{proof}}
Under the stated assumption, the proposition follows because the objective gives a bound on the update error and the convergence claim follows from monotone descent of the surrogate loss.
\end{{proof}}
\end{{document}}
"""


class HallucinatedCitationLLM(LatexLLM):
    def complete(self, prompt: str) -> str:
        return super().complete(prompt).replace("\\cite{paper_1,paper_2,paper_3}", "\\cite{paper_1,fake_key}")


class NoProofLatexLLM(LatexLLM):
    def complete(self, prompt: str) -> str:
        tex = super().complete(prompt)
        tex = tex.replace(
            "\\begin{proof}\nThe claim follows from the stated construction.\n\\end{proof}\n",
            "The method has a theoretical rationale based on regularized empirical risk, but this draft does not state a formal theorem.\n",
        )
        return tex.split("\\section{Theoretical Proofs}", 1)[0] + "\\end{document}\n"


class ShortLatexLLM:
    def complete(self, prompt: str) -> str:
        return r"""
\documentclass{article}
\usepackage{icml2026}
\begin{document}
\begin{abstract}
Too short.
\end{abstract}
\section{Introduction}
Too short.
\section{Related Work}
Too short.
\section{Method}
Too short.
\section{Experiments}
Too short.
\section{Conclusion}
Too short.
\bibliography{references}
\bibliographystyle{icml2026}
\end{document}
"""


class FailingWriterLLM:
    def complete(self, prompt: str) -> str:
        raise RuntimeError("HTTP provider request failed: 524 upstream timeout")


class TimeoutThenLatexLLM:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if self.calls == 1:
            raise RuntimeError("HTTP provider request failed: 524 upstream timeout")
        return LatexLLM().complete(prompt)


class BrokenThenFixedLatexLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        tex = LatexLLM().complete(prompt)
        if self.calls == 1:
            return tex.replace("\\end{document}", "\\BROKENCOMPILE\n\\end{document}")
        return tex


def test_latex_compile_uses_configured_timeout(tmp_path, monkeypatch):
    artifacts = ArtifactManager(tmp_path, "write-timeout")
    paper_dir = artifacts.path("paper_gene")
    paper_dir.mkdir(parents=True)
    (paper_dir / "main.tex").write_text("\\documentclass{article}\\begin{document}x\\end{document}\n", encoding="utf-8")
    builder = BuilderAgent(artifacts, latex_timeout_seconds=33)
    seen = {}

    monkeypatch.setattr("quit_agent.agents.builder_agent.shutil.which", lambda name: f"/usr/bin/{name}" if name == "pdflatex" else None)

    def fake_run(command, **kwargs):
        seen.setdefault("timeouts", []).append(kwargs.get("timeout"))
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("quit_agent.agents.builder_agent.subprocess.run", fake_run)

    report = builder._compile_latex(paper_dir)

    assert report["status"] == "PASS"
    assert seen["timeouts"] == [33, 33, 33]


def test_compact_experiment_results_uses_build_spec_metric_columns(tmp_path):
    artifacts = ArtifactManager(tmp_path, "write-demo")
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-1",
            idea_id="idea-1",
            target_task="offline RL",
            problem_statement="problem",
            method_summary="method",
            baselines=["CQL"],
            metrics=["1-Wasserstein distance", "Eval Reward"],
        ),
    )
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/results_table.csv").write_text(
        "method,source,seed,1-Wasserstein distance,Eval Reward\n"
        "PRFM,proposed,0,1.2,0.7\n"
        "PRFM,proposed,1,1.0,0.8\n"
        "CQL,baseline,0,1.6,0.5\n"
        "CQL,baseline,1,1.4,0.6\n",
        encoding="utf-8",
    )
    artifacts.path("results/sensitivity_results.csv").write_text(
        "method,1-Wasserstein distance\nPRFM,1.1\n",
        encoding="utf-8",
    )
    artifacts.path("results/ablations").mkdir()
    artifacts.path("results/ablations/ablation_results.csv").write_text(
        "method,1-Wasserstein distance\nPRFM-no-ensemble,1.3\n",
        encoding="utf-8",
    )

    summary = BuilderAgent(artifacts)._compact_experiment_results()

    assert summary["has_build_spec_metric_numeric_rows"] is True
    assert summary["primary_metric_name"] == "1-Wasserstein distance"
    assert summary["primary_metric_goal"] == "minimize"
    assert summary["best_result"]["method"] == "PRFM"
    assert summary["best_result"]["primary_metric_value"] == 1.1
    assert "score" not in summary["best_result"]
    assert {preview["path"] for preview in summary["csv_table_previews"]} == {
        "results/results_table.csv",
        "results/sensitivity_results.csv",
        "results/ablations/ablation_results.csv",
    }


def test_compact_experiment_results_does_not_fallback_to_score_without_build_spec_metrics(tmp_path):
    artifacts = ArtifactManager(tmp_path, "write-demo")
    artifacts.write_json(
        "BuildSpec.json",
        BuildSpec(
            build_id="build-1",
            idea_id="idea-1",
            target_task="generic task",
            problem_statement="problem",
            method_summary="method",
            metrics=[],
        ),
    )
    artifacts.path("results").mkdir(parents=True)
    artifacts.path("results/results_table.csv").write_text(
        "method,score\nproposed,1.0\nbaseline,0.5\n",
        encoding="utf-8",
    )

    summary = BuilderAgent(artifacts)._compact_experiment_results()

    assert summary["numeric_result_rows"] == []
    assert summary["has_build_spec_metric_numeric_rows"] is False
    assert summary["csv_table_previews"][0]["columns"] == ["method", "score"]


def test_write_stage_copies_template_and_generates_icml_artifacts(tmp_path):
    artifacts = ArtifactManager(tmp_path, "write-demo")
    builder = BuilderAgent(artifacts, llm=LatexLLM())
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="VAE representation learning",
        problem_statement="Improve latent representation quality under limited labels.",
        method_summary="Use an evidence-guided variational objective with stability regularization.",
        implementation_plan=["Implement encoder and decoder modules."],
        experiment_plan=["Compare against a standard VAE baseline."],
        baselines=["VAE"],
        metrics=["ELBO", "reconstruction loss"],
        success_criteria=["Lower validation reconstruction loss."],
        citations_required=["paper-1"],
    )
    evidence = [
        EvidenceCard(
            evidence_id="ev-1",
            paper_id="paper-1",
            task="latent variable modeling",
            method="variational autoencoder",
            claims=["regularization improves latent stability"],
            metrics=["ELBO"],
        )
    ]
    papers = [
        PaperCard(
            paper_id=f"paper-{index}",
            title=f"A Useful VAE Paper {index}",
            authors=["A. Author"],
            year=2025,
            source="arxiv",
        )
        for index in range(1, 16)
    ]

    report, prompt, response = builder.write(spec, evidence, papers)

    assert artifacts.path("paper_gene/main.tex").exists()
    assert artifacts.path("paper_gene/references.bib").exists()
    assert artifacts.path("WriteReport.json").exists()
    tex = artifacts.path("paper_gene/main.tex").read_text(encoding="utf-8")
    assert "\\usepackage[preprint]{icml2026}" in tex
    assert "\\icmlauthor{QUIT}{agent}" in tex
    assert "\\icmlauthor{QUIT Agent}{agent}" not in tex
    assert "\\icmlaffiliation{agent}{base model: unknown-model, builder: Xinchen}" in tex
    assert "\\icmlauthor{Xinchen Han}{builder}" not in tex
    assert "\\begin{algorithm}" in tex
    assert "\\begin{proof}" in tex
    assert "\\mathcal{L}" in tex
    assert "paper_gene/main.tex" in report["outputs"]
    # prompt is now the actual LLM skill prompt from write_from_build_spec.md
    assert "ICML 2026" in prompt
    assert "BuildSpec" in prompt
    assert report["used_llm"] is True
    assert report["length_validation"]["status"] == "PASS"
    assert report["reference_validation"]["status"] == "PASS"
    assert report["appendix_validation"]["status"] == "PASS"
    assert report["reference_validation"]["recommended_references"] == 15
    assert report["reference_validation"]["bib_entry_count"] >= 15
    assert response.strip().startswith("\\documentclass")


def test_write_stage_allows_missing_theoretical_appendix_when_no_proof_exists(tmp_path):
    artifacts = ArtifactManager(tmp_path, "write-no-proof")
    builder = BuilderAgent(artifacts, llm=NoProofLatexLLM())
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="VAE representation learning",
        problem_statement="Improve latent representation quality under limited labels.",
        method_summary="Use an evidence-guided variational objective with stability regularization.",
    )
    papers = [
        PaperCard(
            paper_id=f"paper-{index}",
            title=f"A Useful VAE Paper {index}",
            authors=["A. Author"],
            year=2025,
            source="arxiv",
        )
        for index in range(1, 16)
    ]

    report, _, _ = builder.write(spec, [], papers)

    assert report["appendix_validation"]["status"] == "PASS"
    assert report["appendix_validation"]["theoretical_proofs_required"] is False
    assert report["appendix_validation"]["theoretical_proofs_word_count"] == 0


def test_write_stage_rejects_missing_appendix_details_and_proofs(tmp_path):
    class NoAppendixLLM(LatexLLM):
        def complete(self, prompt: str) -> str:
            tex = super().complete(prompt)
            return tex.split("\\appendix", 1)[0] + "\\end{document}\n"

    artifacts = ArtifactManager(tmp_path, "write-no-appendix")
    builder = BuilderAgent(artifacts, llm=NoAppendixLLM())
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="VAE representation learning",
        problem_statement="Improve latent representation quality under limited labels.",
        method_summary="Use an evidence-guided variational objective with stability regularization.",
    )
    papers = [
        PaperCard(
            paper_id=f"paper-{index}",
            title=f"A Useful VAE Paper {index}",
            authors=["A. Author"],
            year=2025,
            source="arxiv",
        )
        for index in range(1, 16)
    ]

    report, _, _ = builder.write(spec, [], papers)

    assert report["appendix_validation"]["status"] == "FAIL"
    rules = {failure["rule"] for failure in report["appendix_validation"]["failures"]}
    assert "missing_appendix" in rules
    assert "missing_appendix_experimental_details" in rules
    assert "missing_appendix_theoretical_proofs" in rules


def test_write_stage_repairs_latex_compile_failure(tmp_path):
    artifacts = ArtifactManager(tmp_path, "write-repair")
    llm = BrokenThenFixedLatexLLM()
    builder = BuilderAgent(artifacts, llm=llm)
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="VAE representation learning",
        problem_statement="Improve latent representation quality under limited labels.",
        method_summary="Use an evidence-guided variational objective with stability regularization.",
    )
    papers = [
        PaperCard(
            paper_id=f"paper-{index}",
            title=f"A Useful VAE Paper {index}",
            authors=["A. Author"],
            year=2025,
            source="arxiv",
        )
        for index in range(1, 16)
    ]

    def fake_compile(paper_dir):
        tex = (paper_dir / "main.tex").read_text(encoding="utf-8")
        if "\\BROKENCOMPILE" in tex:
            return {
                "status": "FAIL",
                "engine": "fake",
                "runs": [
                    {
                        "command": "fake-latex main.tex",
                        "returncode": 1,
                        "stdout_tail": "! Undefined control sequence. \\BROKENCOMPILE",
                        "stderr_tail": "",
                    }
                ],
            }
        (paper_dir / "main.pdf").write_bytes(b"%PDF-1.4\n")
        return {
            "status": "PASS",
            "engine": "fake",
            "runs": [
                {
                    "command": "fake-latex main.tex",
                    "returncode": 0,
                    "stdout_tail": "Output written on main.pdf (8 pages).",
                    "stderr_tail": "",
                }
            ],
        }

    builder._compile_latex = fake_compile

    report, _, _ = builder.write(spec, [], papers)

    assert report["status"] == "PASS"
    assert report["compile"]["status"] == "PASS"
    assert report["latex_repair"]["attempted"] is True
    assert report["latex_repair"]["succeeded"] is True
    assert llm.calls == 2
    assert artifacts.path("paper_gene/main.pdf").exists()
    assert "\\BROKENCOMPILE" not in artifacts.path("paper_gene/main.tex").read_text(encoding="utf-8")


def test_write_stage_fails_without_writer_llm_instead_of_placeholder(tmp_path):
    artifacts = ArtifactManager(tmp_path, "write-no-llm")
    builder = BuilderAgent(artifacts)
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="VAE representation learning",
        problem_statement="Improve latent representation quality under limited labels.",
        method_summary="Use an evidence-guided variational objective with stability regularization.",
    )

    report, prompt, response = builder.write(spec, [], [])

    assert report["status"] == "FAIL"
    assert report["reason"] == "writer LLM is not configured"
    assert not artifacts.path("paper_gene/main.tex").exists()
    assert artifacts.path("paper_gene/references.bib").exists()
    assert "ICML 2026" in prompt
    assert response == ""


def test_write_stage_reports_llm_request_failure(tmp_path):
    artifacts = ArtifactManager(tmp_path, "write-llm-fail")
    builder = BuilderAgent(artifacts, llm=FailingWriterLLM())
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="VAE representation learning",
        problem_statement="Improve latent representation quality under limited labels.",
        method_summary="Use an evidence-guided variational objective with stability regularization.",
    )

    report, prompt, response = builder.write(spec, [], [])

    assert report["status"] == "FAIL"
    assert "524 upstream timeout" in report["reason"]
    assert report["used_llm"] is True
    assert not artifacts.path("paper_gene/main.tex").exists()
    assert artifacts.path("paper_gene/references.bib").exists()
    assert artifacts.path("WriteReport.json").exists()
    assert artifacts.path("WriteDraft.error.txt").exists()
    assert "ICML 2026" in prompt
    assert response == ""


def test_write_stage_retries_compact_prompt_after_524(tmp_path):
    artifacts = ArtifactManager(tmp_path, "write-retry")
    llm = TimeoutThenLatexLLM()
    builder = BuilderAgent(artifacts, llm=llm)
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="VAE representation learning",
        problem_statement="Improve latent representation quality under limited labels.",
        method_summary="Use an evidence-guided variational objective with stability regularization.",
    )
    papers = [
        PaperCard(
            paper_id=f"paper-{index}",
            title=f"A Useful VAE Paper {index}",
            authors=["A. Author"],
            year=2025,
            source="arxiv",
        )
        for index in range(1, 16)
    ]

    report, prompt, response = builder.write(spec, [], papers)

    assert llm.calls >= 2
    assert report["used_llm"] is True
    assert report["appendix_validation"]["status"] == "PASS"
    assert artifacts.path("paper_gene/main.tex").exists()
    assert artifacts.path("WriteDraft.retry_prompt.txt").exists()
    assert "compact artifacts" in llm.prompts[1]
    assert "compact artifacts" in prompt
    assert "524 upstream timeout" not in report.get("reason", "")
    assert response.strip().startswith("\\documentclass")


def test_write_stage_rejects_short_sections(tmp_path):
    artifacts = ArtifactManager(tmp_path, "write-short")
    builder = BuilderAgent(artifacts, llm=ShortLatexLLM())
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="VAE representation learning",
        problem_statement="Improve latent representation quality under limited labels.",
        method_summary="Use an evidence-guided variational objective with stability regularization.",
    )

    report, _, _ = builder.write(spec, [], [])

    assert report["status"] == "FAIL"
    assert report["length_validation"]["status"] == "FAIL"
    assert report["page_validation"]["expected_pages"] == 7
    assert report["page_validation"]["hard_fail"] is True
    assert {failure["section"] for failure in report["length_validation"]["failures"]} >= {
        "abstract",
        "introduction",
        "related work",
        "method",
        "experiments",
        "conclusion",
    }
    assert report["compile"]["status"] in {"PASS", "FAIL", "SKIPPED"}


def test_write_stage_rejects_hallucinated_citations(tmp_path):
    artifacts = ArtifactManager(tmp_path, "write-hallucinated-cite")
    builder = BuilderAgent(artifacts, llm=HallucinatedCitationLLM())
    spec = BuildSpec(
        build_id="build-idea-1",
        idea_id="idea-1",
        target_task="VAE representation learning",
        problem_statement="Improve latent representation quality under limited labels.",
        method_summary="Use an evidence-guided variational objective with stability regularization.",
    )
    papers = [
        PaperCard(
            paper_id=f"paper-{index}",
            title=f"A Useful VAE Paper {index}",
            authors=["A. Author"],
            year=2025,
            source="arxiv",
        )
        for index in range(1, 16)
    ]

    report, _, _ = builder.write(spec, [], papers)

    assert report["status"] == "PARTIAL"
    assert report["length_validation"]["status"] == "PASS"
    assert report["reference_validation"]["status"] == "FAIL"
    assert "fake_key" in report["reference_validation"]["hallucinated_keys"]
    assert report["compile"]["status"] in {"PASS", "FAIL", "SKIPPED"}
