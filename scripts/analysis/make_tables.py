"""Write results/values.tex (numbers quoted in the paper) and results/tables/*.tex from the result CSVs.

Every value is looked up by key, so a reordered CSV cannot silently change a number.
Claims made in the paper text are asserted here.
"""

import pandas as pd

from radhalo.data.release import IMPRESSION_SOURCES
from radhalo.detectors import DETECTORS, FRONTIER_JUDGES
from radhalo.reporting.io import REPO, parse_args
from radhalo.reporting.latex import NUMBER_WORDS, Macros, fmt, lookup, thousands, write_table

RADCHECK = "radcheck-8b"
GEMINI_FLASH = "gemini-2.5-flash"  # both an impression source and a judge
TABLE1_SOURCE_ORDER = [
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-flash",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Ministral-3-3B-Instruct-2512",
    "Qwen/Qwen3-1.7B",
    "original_radiologist",
]
AGREEMENT_SOURCE_ORDER = TABLE1_SOURCE_ORDER[4:5] + TABLE1_SOURCE_ORDER[:4] + TABLE1_SOURCE_ORDER[5:]
THEMATIC_CLUSTERS = [
    ("reference to external information", "Reference to external information", "thematicexternalpct"),
    ("clinical judgment", "Clinical judgment", "thematicjudgmentpct"),
    ("unsupported information", "Unsupported information", "thematicunsupportedpct"),
    ("ambiguity", "Ambiguity", "thematicambiguitypct"),
]
TOP_MODALITY_ANATOMY = 7


def metric_value(df: pd.DataFrame, metric: str):
    return lookup(df, "metric", metric, "value")


def dataset_numbers(m: Macros, csv, tables) -> None:
    overall = csv("error_rates_overall")
    by_source = csv("error_rates_by_source")
    by_risk = csv("error_rates_by_risk_level")

    m.add("totalreports", int(metric_value(overall, "num_reports")))
    m.add("totalfindings", int(metric_value(overall, "findings_sentences")))
    m.add("totalimpression", int(metric_value(overall, "impression_sentences")))
    m.add("totalhallucinations", int(metric_value(overall, "hallucinations")))
    m.add("totalomissions", int(metric_value(overall, "omissions")))
    m.add("hallrate", metric_value(overall, "hallucination_rate"))
    m.add("omisrate", metric_value(overall, "omission_rate"))
    m.add("pctreportshall", metric_value(overall, "pct_reports_with_hallucination"))
    m.add("pctreportsomis", metric_value(overall, "pct_reports_with_omission"))

    def source_rate(source, column):
        return lookup(by_source, "impression_source", source, column)

    def risk_rate(level, column):
        return lookup(by_risk, "risk_level", level, column)

    m.add("refhallrate", source_rate("original_radiologist", "hallucination_rate"))
    m.add("refomisrate", source_rate("original_radiologist", "omission_rate"))
    m.add("omisratemin", fmt(by_source["omission_rate"].min(), 1))
    m.add("omisratemax", fmt(by_source["omission_rate"].max(), 1))
    for name, source in [
        ("llamahallratefmt", "meta-llama/Llama-3.1-8B-Instruct"),
        ("ministralhallratefmt", "mistralai/Ministral-3-3B-Instruct-2512"),
        ("geminiflashhallratefmt", "gemini-2.5-flash"),
        ("claudehallratefmt", "claude-sonnet-4-5-20250929"),
    ]:
        m.add(name, fmt(source_rate(source, "hallucination_rate"), 1))
    for name, level, column in [
        ("riskonehallratefmt", 1, "hallucination_rate"),
        ("riskoneomisratefmt", 1, "omission_rate"),
        ("riskfourhallratefmt", 4, "hallucination_rate"),
        ("riskfouromisratefmt", 4, "omission_rate"),
    ]:
        m.add(name, fmt(risk_rate(level, column), 1))

    def rates(hall, omis):
        return f"{fmt(hall, 1)}\\% & {fmt(omis, 1)}\\% \\\\"

    lines = [
        "\\begin{tabular}{@{}lrr@{}}",
        "\\toprule",
        "\\textbf{Category} & \\textbf{Hall. rate} & \\textbf{Omis. rate} \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\textit{Overall}} \\\\",
        "\\quad Sentence-level & "
        + rates(metric_value(overall, "hallucination_rate"), metric_value(overall, "omission_rate")),
        "\\quad Report-level & "
        + rates(
            metric_value(overall, "pct_reports_with_hallucination"),
            metric_value(overall, "pct_reports_with_omission"),
        ),
        "\\midrule",
        "\\multicolumn{3}{l}{\\textit{By impression source}} \\\\",
    ]
    for source in TABLE1_SOURCE_ORDER:
        label = "MIMIC-IV Ref." if source == "original_radiologist" else IMPRESSION_SOURCES[source]
        lines.append(
            f"\\quad {label} & "
            + rates(source_rate(source, "hallucination_rate"), source_rate(source, "omission_rate"))
        )
    lines += ["\\midrule", "\\multicolumn{3}{l}{\\textit{By MedVAL risk level}} \\\\"]
    for level in (1, 2, 3, 4):
        lines.append(
            f"\\quad Level {level} & "
            + rates(risk_rate(level, "hallucination_rate"), risk_rate(level, "omission_rate"))
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(tables / "error_by_stratum.tex", lines)


def modality_anatomy(m: Macros, csv) -> None:
    """Table 1 block, emitted as a macro: \\input cannot be used inside a tabular row scan."""
    counts = csv("modality_anatomy_counts").sort_values("num_reports", ascending=False, kind="stable")
    total = int(counts["num_reports"].sum())
    top = counts.head(TOP_MODALITY_ANATOMY)
    n_other = len(counts) - TOP_MODALITY_ANATOMY
    other = total - int(top["num_reports"].sum())
    rows = []
    for _, row in top.iterrows():
        modality, anatomy = row["modality_anatomy"].split(" ", 1)
        n = int(row["num_reports"])
        rows.append(f"\\quad {modality} -- {anatomy} & {n} ({fmt(100 * n / total, 1)}\\%) \\\\")
    rows.append(f"\\quad Other ({n_other} combinations) & {other} ({fmt(100 * other / total, 1)}\\%) \\\\")
    m.add("nummodalityanatomy", len(counts))
    m.add("numothermodalityanatomy", n_other)
    m.add("modalityanatomyrows", " ".join(rows))


def annotation_numbers(m: Macros, csv, tables) -> None:
    coverage = csv("prehighlight_coverage")
    m.add("prehighlightpct", fmt(metric_value(coverage, "pct_findings_sentences_highlighted"), 1))
    m.add("omissionsinhighlighted", int(metric_value(coverage, "omissions_inside_highlighted")))
    m.add("omissionsoutsidehighlighted", int(metric_value(coverage, "omissions_outside_highlighted")))

    trend = csv("risk_trend")
    for prefix, error_type in (("hall", "hallucination"), ("omis", "omission")):
        subset = trend[trend["error_type"] == error_type]
        p = float(metric_value(subset, "cochran_armitage_p"))
        if p >= 0.001:
            raise ValueError(f"paper states p < 0.001 for the {error_type} risk trend, got {p:.3g}")
        m.add(f"{prefix}riskz", fmt(metric_value(subset, "cochran_armitage_z"), 1))

    per_annotator = csv("annotator_error_rates")

    def annotator_value(i, column):
        return lookup(per_annotator, "annotator", f"annotator_{i}", column)

    for name, i, column in [
        ("annthreehallrate", 3, "hallucination_rate"),
        ("annfourhallrate", 4, "hallucination_rate"),
        ("annthreeomisrate", 3, "omission_rate"),
        ("annfouromisrate", 4, "omission_rate"),
    ]:
        m.add(name, fmt(annotator_value(i, column), 1))

    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "\\textbf{Annotator} & \\shortstack{\\textbf{Annotated}\\\\\\textbf{impr. sent.}} & "
        "\\textbf{Hall. rate} & \\shortstack{\\textbf{Annotated}\\\\\\textbf{find. sent.}} & "
        "\\textbf{Omis. rate} \\\\",
        "\\midrule",
    ]
    for i in range(1, 5):
        lines.append(
            f"Annotator {i} & {int(annotator_value(i, 'impression_sentences'))} & "
            f"{fmt(annotator_value(i, 'hallucination_rate'), 1)}\\% & "
            f"{int(annotator_value(i, 'findings_sentences'))} & "
            f"{fmt(annotator_value(i, 'omission_rate'), 1)}\\% \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(tables / "per_annotator_rates.tex", lines)

    summary = csv("adjudication_summary")

    def outcomes(prefix):
        return " & ".join(
            str(int(metric_value(summary, f"{prefix}_{o}")))
            for o in ("disputed_sentences", "adjudicated_as_error", "adjudicated_as_not_error")
        )

    write_table(
        tables / "adjudication_summary.tex",
        [
            "\\begin{tabular}{lrrr}",
            "\\toprule",
            "\\textbf{Dispute set} & \\textbf{Disputed} & \\textbf{Adjudicated error} & "
            "\\textbf{Adjudicated not error} \\\\",
            "\\midrule",
            f"Hallucination & {outcomes('hall')} \\\\",
            f"Omission & {outcomes('omis')} \\\\",
            "\\midrule",
            f"Total & {outcomes('total')} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )

    adjudicator = csv("adjudicator_agreement")
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "\\textbf{Annotator} & \\textbf{Hall. agreement} & \\textbf{Omis. agreement} & "
        "\\textbf{Overall agreement} \\\\",
        "\\midrule",
    ]
    for i in range(1, 5):
        values = [
            fmt(lookup(adjudicator, "annotator", f"annotator_{i}", column), 1)
            for column in ("hall_agreement_rate", "omis_agreement_rate", "overall_agreement_rate")
        ]
        lines.append(f"Annotator {i} & " + "\\% & ".join(values) + "\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(tables / "adjudicator_agreement.tex", lines)


def agreement_numbers(m: Macros, csv, tables) -> None:
    overall = csv("agreement_overall")
    by_source = csv("agreement_by_source")
    by_risk = csv("agreement_by_risk_level")

    m.add("hallalpha", lookup(overall, "error_type", "hallucination", "alpha"))
    m.add("hallpairwisef", lookup(overall, "error_type", "hallucination", "pairwise_f1"))
    m.add("omisalpha", lookup(overall, "error_type", "omission", "alpha"))
    m.add("omispairwisef", lookup(overall, "error_type", "omission", "pairwise_f1"))
    m.add(
        "llamahallalpha",
        fmt(lookup(by_source, "impression_source", "meta-llama/Llama-3.1-8B-Instruct", "hall_alpha"), 3),
    )
    m.add("qwenomisalpha", fmt(lookup(by_source, "impression_source", "Qwen/Qwen3-1.7B", "omis_alpha"), 3))
    for name, column in [
        ("riskfourhallalpha", "hall_alpha"),
        ("riskfourhallfone", "hall_f1"),
        ("riskfouromisalpha", "omis_alpha"),
        ("riskfouromisfone", "omis_f1"),
    ]:
        m.add(name, fmt(lookup(by_risk, "risk_level", 4, column), 3))

    columns = ("hall_alpha", "hall_f1", "omis_alpha", "omis_f1")
    lines = [
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "& \\multicolumn{2}{c}{\\textbf{Hallucination}} & \\multicolumn{2}{c}{\\textbf{Omission}} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "\\textbf{Stratum} & $\\alpha$ & \\textbf{F1} & $\\alpha$ & \\textbf{F1} \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\textit{By impression source}} \\\\",
    ]
    for source in AGREEMENT_SOURCE_ORDER:
        values = [fmt(lookup(by_source, "impression_source", source, c), 3) for c in columns]
        lines.append(f"\\quad {IMPRESSION_SOURCES[source]} & " + " & ".join(values) + " \\\\")
    # Risk level 1 is omitted: too few positive labels for a stable estimate.
    lines += ["\\midrule", "\\multicolumn{5}{l}{\\textit{By MedVAL risk level}} \\\\"]
    for level in (2, 3, 4):
        values = [fmt(lookup(by_risk, "risk_level", level, c), 3) for c in columns]
        lines.append(f"\\quad Level {level} & " + " & ".join(values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(tables / "iaa_by_stratum.tex", lines)


def detector_numbers(m: Macros, csv, tables) -> None:
    metrics = csv("detector_metrics")
    overall = csv("agreement_overall")
    if sorted(metrics["detector"]) != sorted(d.key for d in DETECTORS):
        raise KeyError("detector_metrics.csv does not contain exactly the paper's detectors")

    def value(detector, column):
        return lookup(metrics, "detector", detector, column)

    m.add("besthallfone", metrics["hall_f1"].max())
    m.add("radcheckhallfone", value(RADCHECK, "hall_f1"))
    m.add("radcheckomisfone", value(RADCHECK, "omis_f1"))
    hall_iaa = float(lookup(overall, "error_type", "hallucination", "pairwise_f1"))
    m.add("numdetectorsabovehalliaa", NUMBER_WORDS[int((metrics["hall_f1"] > hall_iaa).sum())])
    m.add("numdetectors", NUMBER_WORDS[len(DETECTORS)])
    for name, column in [
        ("radcheckhallauroc", "hall_auroc"),
        ("radcheckhallauprc", "hall_auprc"),
        ("radcheckomisauroc", "omis_auroc"),
        ("radcheckomisauprc", "omis_auprc"),
    ]:
        m.add(name, fmt(value(RADCHECK, column), 3))

    lines = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "& \\multicolumn{4}{c}{\\textbf{Hallucination}} & \\multicolumn{4}{c}{\\textbf{Omission}} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
        "\\textbf{Detector} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} & \\textbf{F1 95\\% CI} & "
        "\\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} & \\textbf{F1 95\\% CI} \\\\",
        "\\midrule",
    ]
    for detector in DETECTORS:
        cells = []
        for prefix in ("hall", "omis"):
            cells += [fmt(value(detector.key, f"{prefix}_{c}"), 3) for c in ("precision", "recall", "f1")]
            lower = fmt(value(detector.key, f"{prefix}_f1_ci_lower"), 3)
            upper = fmt(value(detector.key, f"{prefix}_f1_ci_upper"), 3)
            cells.append(f"[{lower}, {upper}]")
        lines.append(f"{detector.table_label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write_table(tables / "full_detector_metrics.tex", lines)

    # Limitations: Gemini 2.5 Flash judging its own impressions.
    flags = csv("judge_flags_by_source")
    own = flags[flags["impression_source"] == GEMINI_FLASH]
    for prefix, error_type in (("hall", "hallucination"), ("omis", "omission")):
        subset = own[own["error_type"] == error_type]
        positives = {int(lookup(subset, "judge", j, "positives")) for j in FRONTIER_JUDGES}
        if len(positives) != 1:
            raise ValueError(f"inconsistent {error_type} positive counts across judges")
        m.add(f"geminiflash{prefix}count", positives.pop())
        m.add(f"geminiself{prefix}recovered", int(lookup(subset, "judge", GEMINI_FLASH, "recovered")))
        self_rate = float(lookup(subset, "judge", GEMINI_FLASH, "flag_rate"))
        others = [
            float(lookup(subset, "judge", j, "flag_rate")) for j in FRONTIER_JUDGES if j != GEMINI_FLASH
        ]
        if error_type == "omission" and not self_rate > max(others):
            raise ValueError("paper states Gemini 2.5 Flash flags its own omissions most often")
        if error_type == "hallucination" and not min(others) <= self_rate <= max(others):
            raise ValueError(
                "paper states Gemini 2.5 Flash's own hallucination flag rate is within the others' range"
            )


def synthetic_numbers(m: Macros, tables) -> None:
    synthetic = pd.read_csv(REPO / "derived" / "synthetic_composition.csv")

    def count(metric):
        return int(metric_value(synthetic, metric))

    hall_pos, hall_neg = count("hallucination_positive"), count("hallucination_negative")
    omis_pos, omis_neg = count("omission_positive"), count("omission_negative")
    total = count("total_samples")
    if hall_pos + hall_neg + omis_pos + omis_neg != total:
        raise ValueError("synthetic composition does not sum to total_samples")

    def share(pos, n):
        return f"{fmt(100 * pos / n, 1)}\\%"

    def row(label, pos, neg):
        return (
            f"{label} & {thousands(pos)} & {thousands(neg)} & {thousands(pos + neg)} & "
            f"{share(pos, pos + neg)} \\\\"
        )

    write_table(
        tables / "synthetic_composition.tex",
        [
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "\\textbf{Error type} & \\textbf{Positive} & \\textbf{Negative} & \\textbf{Total} & "
            "\\textbf{Positive share} \\\\",
            "\\midrule",
            row("Hallucination", hall_pos, hall_neg),
            row("Omission", omis_pos, omis_neg),
            "\\midrule",
            row("Combined", hall_pos + omis_pos, hall_neg + omis_neg),
            "\\bottomrule",
            "\\end{tabular}",
        ],
    )
    m.add("synthtotalsamples", thousands(total))
    m.add("synthuniquereports", thousands(count("unique_reports")))
    m.add("synthhallposshare", fmt(100 * hall_pos / (hall_pos + hall_neg), 1))
    m.add("synthomisposshare", fmt(100 * omis_pos / (omis_pos + omis_neg), 1))


def thematic_numbers(m: Macros, tables) -> None:
    clusters = pd.read_csv(REPO / "derived" / "thematic_clusters.csv")
    total = int(clusters["count"].sum())
    m.add("thematichallcount", total)
    lines = [
        "\\begin{tabular}{@{}lrr@{}}",
        "\\toprule",
        "\\textbf{Cluster} & \\textbf{Count} & \\textbf{Share} \\\\",
        "\\midrule",
    ]
    for key, label, macro in THEMATIC_CLUSTERS:
        pct = fmt(lookup(clusters, "cluster", key, "percentage"), 1)
        lines.append(f"{label} & {int(lookup(clusters, 'cluster', key, 'count'))} & {pct}\\% \\\\")
        m.add(macro, pct)
    lines += [
        "\\midrule",
        f"Total & {total} & {fmt(clusters['percentage'].sum(), 1)}\\% \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    write_table(tables / "thematic_clusters.tex", lines)


def main() -> None:
    args = parse_args(__doc__)
    tables = args.results / "tables"

    def csv(name):
        return pd.read_csv(args.results / "csv" / f"{name}.csv")

    m = Macros()
    dataset_numbers(m, csv, tables)
    modality_anatomy(m, csv)
    annotation_numbers(m, csv, tables)
    agreement_numbers(m, csv, tables)
    detector_numbers(m, csv, tables)
    synthetic_numbers(m, tables)
    thematic_numbers(m, tables)
    m.write(args.results / "values.tex")


if __name__ == "__main__":
    main()
