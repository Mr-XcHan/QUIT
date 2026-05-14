# Skill: Write From BuildSpec

## Description
Generate an ICML-style paper artifact from `BuildSpec.json` after code and experiment audit artifacts are available.

This skill is used by the `WRITE` workflow state. It treats `BuildSpec.json` as the source of truth and only uses compact evidence artifacts for citations, style, and support. It must not depend on long chat history.

## Input Artifacts
Required:

```text
BuildSpec.json
```

Optional:

```text
EvidenceCards.jsonl
PaperCards.jsonl
results/metrics.json
results/results_table.csv
results/progress_log.jsonl
all results/*.png files, especially every path declared in BuildSpec.plots
code/EXPERIMENT_METRICS.md
```

## Output Artifacts
Required:

```text
paper_gene/
paper_gene/main.tex
paper_gene/references.bib
WriteReport.json
```

Best effort:

```text
paper_gene/main.pdf
```

## Runtime Policy
The writer must copy the ICML template directory into:

```text
runs/<run_id>/paper_gene/
```

The default template source is:

```text
/netfs/tsp/student/2022/xhan/code/Quit_agent/ICML2026
```

The generated paper must be a complete main-paper draft that passes the WRITE validation gates:

```text
compiled main paper has at least write.expected_main_pages pages
abstract has at least 150 words
introduction has at least 700 words
related work has at least 350 words
method has at least 1600 words
experiments has at least 150 words
conclusion has at least 200 words
references use only PaperCard-backed BibTeX keys
experiments include generated result tables, figures, and analysis
appendix includes concrete experimental details
appendix includes theoretical proofs only when the main paper states proof-like content
```

## Failure Mode
If LaTeX compilation fails or `pdflatex` is unavailable, keep all source files, write `WriteReport.json` with `status=PARTIAL`, and let `WRITE_EVAL` decide fallback. Do not silently mark a failed compile as a complete paper.

## Runtime Prompt Template

```text
You are writing an ICML 2026 formatted academic paper draft from structured research artifacts.

BuildSpec (source of truth for method, experiments, and structure):
{{build_spec}}

Supporting EvidenceCards (use for related work and citations):
{{evidence_cards}}

PaperCards (bibliographic metadata):
{{paper_cards}}

ExperimentResultSummary (must be used for Experiments tables, figures, and analysis):
{{experiment_results}}

Previous WRITE feedback, if any (fix these issues directly):
{{write_feedback}}

BibTeX citation key mapping (use these exact keys in \cite{}):
{{citation_keys}}

Configured WRITE page target:
{{expected_main_pages}} compiled ICML main-paper pages minimum, excluding appendix material.

Hard validation gates:
- The generated main paper MUST be long enough to pass these measured minimum word counts:
  - abstract: at least 150 words
  - Introduction: at least 700 words
  - Related Work: at least 350 words
  - Method: at least 1600 words
  - Experiments: at least 1200 words
  - Conclusion: at least 200 words
- The compiled main paper MUST be at least {{expected_main_pages}} pages under the ICML2026 template, excluding reference and appendix material.
- Do not use placeholders, TODOs, "pending", "omitted for brevity", "..." section bodies, or prose that says results will be added later.
- If previous WRITE feedback includes length_validation, page_validation, reference_validation, experiment_validation, or appendix_validation failures, directly repair those failures in this draft.
- Return ONLY raw LaTeX source. The first non-whitespace characters must be `\documentclass{article}`. Do not wrap the answer in markdown code fences.

Required LaTeX skeleton:
- Use EXACTLY the following structure and package block, but replace every placeholder with full paper text. Do not leave ellipses, bracketed placeholders, or empty sections.

\documentclass{article}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{hyperref}
\newcommand{\theHalgorithm}{\arabic{algorithm}}
\usepackage[preprint]{icml2026}
\renewcommand{\ttdefault}{cmtt}
\usepackage{amsmath,amssymb,mathtools,amsthm}
\usepackage{algorithm}
\usepackage{algorithmic}
\makeatletter
\renewcommand{\printAffiliationsAndNotice}[1]{\global\icml@noticeprintedtrue%
  \stepcounter{@affiliationcounter}%
  {\let\thefootnote\relax\footnotetext{\hspace*{-\footnotesep}\ificmlshowauthors #1\fi%
      \forloop{@affilnum}{1}{\value{@affilnum} < \value{@affiliationcounter}}{
        \textsuperscript{\arabic{@affilnum}}\ifcsname @affilname\the@affilnum\endcsname%
          \csname @affilname\the@affilnum\endcsname%
        \else
          {\bf AUTHORERR: Missing \textbackslash{}icmlaffiliation.}
        \fi
      }.%
      \ \\
      \Notice@String
    }
  }
}
\makeatother
\icmltitlerunning{<SHORT TITLE>}
\begin{document}
\twocolumn[
\icmltitle{<FULL TITLE>}
\begin{icmlauthorlist}
\icmlauthor{QUIT}{agent}
\end{icmlauthorlist}
\icmlaffiliation{agent}{base model: {{model_name}}, builder: Xinchen}
\icmlkeywords{<KEYWORDS>}
\vskip 0.3in
]
\printAffiliationsAndNotice{}
\begin{abstract}
<150+ word abstract>
\end{abstract}
\section{Introduction} <700+ words>
\section{Related Work} <350+ words with citations>
\section{Preliminaries} <definitions and notation as needed>
\section{Method} <1600+ words with objective, algorithmic details, and pseudocode>
\section{Experiments} <1200+ words with required subsections, tables, and figures>
\section{Conclusion} <200+ words>
\bibliography{references}
\bibliographystyle{icml2026}
\appendix
\section{Experimental Details} <reproducibility details>
\end{document}

- The title/author/keywords block MUST be inside `\twocolumn[...]` which comes immediately after `\begin{document}`. Never put `\icmltitle`, `\icmlauthorlist`, or `\icmlaffiliation` in the preamble (before `\begin{document}`).
- `\printAffiliationsAndNotice{}` must appear immediately after the closing `]` of `\twocolumn[...]`, before `\begin{abstract}`.
- Use the ICML abstract environment: `\begin{abstract}...\end{abstract}`. Do not write `\section{Abstract}`.
- Include sections in this exact order: Introduction, Related Work, Preliminaries, Method, Experiments, Conclusion, then `\bibliography{references}`, `\bibliographystyle{icml2026}`, and `\appendix`.
- The compiled main paper must meet the configured `write.expected_main_pages={{expected_main_pages}}` threshold under the ICML2026 template, excluding appendix material. Balance section lengths accordingly: WRITE_EVAL treats compiled PDFs with fewer pages than this threshold as direct FAIL and feeds that failure reason back to the next WRITE attempt.
- Method must be grounded in BuildSpec and include the objective/loss, implementation-level algorithm description, and a designed pseudocode algorithm. The pseudocode must describe the actual proposed method from BuildSpec, including inputs, the core computation loop (training, generation, or inference depending on the method type), evaluation output, and return value. Do not use generic placeholder pseudocode.
- If no defensible theorem exists, write theoretical discussion instead of inventing proof.
- If using `algorithmic`, use the older uppercase commands only: `\REQUIRE`, `\ENSURE`, `\STATE`, `\FOR`, `\ENDFOR`, `\IF`, `\ENDIF`. Do not use `\Return`; write `\STATE \textbf{return} ...`.
- Related Work must cite retrieved papers using \cite{key} from the citation key mapping above.
- Use only citation keys that appear in the BibTeX citation key mapping. Never invent citation keys.
- Every citation key used in \cite{} must appear exactly in the BibTeX citation key mapping. Citation hallucination is forbidden.

Experiment requirements:
- The Experiments section MUST use `ExperimentResultSummary`, not only BuildSpec.
- Treat BuildSpec `metrics`, `logging`, `plots`, and `baselines` as the source of truth for what CODE was asked to record and what WRITE must discuss. Do not introduce new primary metrics, baselines, datasets, plots, or claims that are not in BuildSpec or the generated artifacts.
- File names and paths are strict. Use the exact figure paths declared in BuildSpec `plots` and the exact available artifact paths from `ExperimentResultSummary.result_plot_paths`. Do not rename figures, invent aliases, change extensions, move figures, or refer to missing/generated-by-hand filenames.
- Every available PNG artifact under `results/` MUST appear in the LaTeX source as an `\includegraphics{...}` figure. This includes all BuildSpec-declared plot files and any additional generated `results/*.png` files listed by `ExperimentResultSummary.result_plot_paths`.
- Use the exact relative LaTeX path for figures from `paper_gene/main.tex`: `../results/<filename>.png`. For a BuildSpec plot path such as `results/secondary_metric_curve.png`, the LaTeX command must include `\includegraphics[width=\columnwidth]{../results/secondary_metric_curve.png}` exactly.
- Every available CSV result table summarized in `ExperimentResultSummary.csv_table_previews` MUST be represented in the paper as an editable LaTeX `table`/`tabular` or a compact appendix table. At minimum, convert `results/results_table.csv` into a real LaTeX table with the actual method names, BuildSpec metric column names, and numeric values. Do not include CSV as an image, verbatim block, attachment, or prose-only summary.
- The Experiments section must discuss the BuildSpec reporting targets when available: metrics in `metrics`, figures in `plots`, or both. Do not collapse all results into a single generic score.
- The main paper should present and discuss at least five total reporting artifacts or targets when available, counting LaTeX tables, result figures, and distinct BuildSpec metrics/plots. If fewer than five artifacts exist, discuss every available one and explain the limited artifact set.
- The Experiments section MUST contain these four required subsections in order:
  \subsection{Experimental Setup} — datasets/environments, evaluation protocol, hardware, seeds (~150 words)
  \subsection{Baselines} — one paragraph per baseline from BuildSpec `baselines` explaining what it is and why it is included (~200 words total)
  \subsection{Main Results} — the results table plus at least 200 words of analysis
  \subsection{Ablation Study} — at least one ablation or diagnostic with discussion (~150 words)
- Do NOT write a single-subsection or free-form Experiments section. Do NOT treat `\subsection{Training or Execution Dynamics}` or `\subsection{Discussion}` as mandatory subsections. The four subsections above are the required minimum structure; when BuildSpec and result artifacts justify more experimental material, you may add extra subsections after the required four. If the main paper already has enough pages or a table/figure is too wide, place the extra material in the appendix and summarize its conclusion in the main Experiments section.
- Present every table and figure available from the code/metrics artifacts. If `results/results_table.csv` contains rows, convert the actual rows into a LaTeX `table` with real method names and numbers. Do not write pending/TBD/N/A-only main results when numeric rows exist.
- Preserve CSV column names as table headers unless they are too long for ICML layout; if shortening is necessary, keep the meaning unambiguous and explain units in the caption.
- If `numeric_result_rows` is non-empty, build the main results table in \subsection{Main Results} from those rows. Rows are already aggregated across seeds. Use `method` as the row label and `primary_metric_value` as the primary numeric column, with `primary_metric_name` or `primary_metric_column` as the table header. Include `domain` as a column when non-empty. Include `std` as a ± column when non-empty. Use `ExperimentResultSummary.primary_metric_goal` to decide the best row: bold the minimum when the goal is `minimize`, and bold the maximum when the goal is `maximize`. Do not call this column a generic score unless BuildSpec actually names it score.
- If `numeric_result_rows` is empty but `results_table_available=true`, first inspect `ExperimentResultSummary.csv_table_previews`. Only write "—" or "n/a (smoke run)" when there are no numeric columns matching BuildSpec `metrics`. If the CSV previews contain BuildSpec metric columns or diagnostic numeric columns, convert those previews into truthful tables and explain that they are diagnostic or non-primary results rather than inventing missing primary metrics.
- Use `csv_table_previews` to cover auxiliary results such as sensitivity, ablation, robustness, diagnostic, or secondary-metric CSVs. Keep the main paper table focused on BuildSpec primary metrics. Put ablation/diagnostic tables in `\subsection{Ablation Study}` when compact; move wide auxiliary tables to the appendix when necessary and summarize their conclusions in the main Experiments section.
- General figure and table presentation requirements for the whole paper:
  - Prefer the figure paths and meanings declared in BuildSpec `plots`.
  - Include and discuss every available BuildSpec-declared plot file.
  - Include and discuss every additional `results/*.png` file listed in `ExperimentResultSummary.result_plot_paths`, even if it is not in BuildSpec `plots`.
  - Place each figure in the most relevant required subsection: final comparison figures normally belong in `\subsection{Main Results}`; training, sensitivity, ablation, robustness, diagnostic, or secondary-metric curves normally belong in `\subsection{Ablation Study}` or the appendix.
  - When a plot corresponds to a trainable model, discuss loss/validation dynamics only if those are the actual plotted quantities.
  - When a plot corresponds to simulation, optimization, scheduling, control, queueing, search, or analysis, discuss the plotted domain metric from BuildSpec `metrics` instead of inventing loss convergence.
  - If `progress_curve_available=true`: include exactly `\includegraphics[width=\columnwidth]{../results/progress_curve.png}` and caption it according to BuildSpec `plots` or the plotted metric.
  - If only `legacy_training_curve_available=true`, include exactly `\includegraphics[width=\columnwidth]{../results/training_curve.png}` and caption it according to BuildSpec `plots` or the plotted metric.
  - If `eval_curve_available=true`: include exactly `\includegraphics[width=\columnwidth]{../results/eval_curve.png}` and caption it according to BuildSpec `plots` or the plotted metric.
  - If `results/secondary_metric_curve.png` or any other PNG is available, include it with the exact `../results/<filename>.png` path.
  - Place each figure in its own `\begin{figure}[t]` block with a `\caption{}` and `\label{}`. Do not merge both plots into one figure block.
  - Analyze every included table and figure in prose near where it appears: identify the best/worst method when applicable, whether the proposed method improves or underperforms, what each curve shows, and any unavailable domains from `metrics.json`.

Appendix:
- After references, include `\appendix` and `\section{Experimental Details}`. This section is mandatory and must describe datasets/environments, preprocessing, hyperparameters, seeds, evaluation protocol, hardware/device, result artifact paths, and any unavailable benchmarks or failed comparisons.
- Include `\section{Theoretical Proofs}` only if the draft, BuildSpec, EvidenceCards, or prior artifacts contain a real proposition/lemma/theorem/proof sketch for the proposed algorithm. If such a proof exists, preserve and expand it here with assumptions, notation, and proof steps. If no proof is needed or supported, omit this section rather than inventing one.

```
