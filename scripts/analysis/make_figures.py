"""Figure 3 (detector F1 vs. inter-annotator agreement) and the risk-grade distribution (App. A.7)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from radhalo.data.release import IMPRESSION_SOURCES  # noqa: E402
from radhalo.detectors import DETECTORS  # noqa: E402
from radhalo.reporting.io import REPO, parse_args  # noqa: E402

TITLE_SIZE, AXIS_LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 13, 11, 10, 10
LLM_COLOR, RADCHECK_COLOR, IAA_COLOR = "#4C78A8", "#59A14F", "#666666"
FIGURE3_GROUPS = [
    "Claude Opus 5.5",
    "Gemini 2.5 Flash",
    "Gemini 3.1 Pro",
    "GPT-6 Astra",
    "Qwen 3 8B",
    "RadCheck 8B",
]
SOURCE_COLORS = {
    "gemini-2.5-flash": "#377EB8",
    "claude-sonnet-4-5-20250929": "#FF7F00",
    "meta-llama/Llama-3.1-8B-Instruct": "#4DAF4A",
    "mistralai/Ministral-3-3B-Instruct-2512": "#E41A1C",
    "Qwen/Qwen3-1.7B": "#984EA3",
    "original_radiologist": "#999999",
}
PDF_METADATA = {"CreationDate": None}


def detector_performance(metrics: pd.DataFrame, agreement: pd.DataFrame) -> plt.Figure:
    order = [d for group in FIGURE3_GROUPS for d in DETECTORS if d.model == group]
    rows = [metrics.set_index("detector").loc[d.key] for d in order]
    y = np.arange(len(order))[::-1]
    llm = [i for i, d in enumerate(order) if not d.is_classifier]
    radcheck = [i for i, d in enumerate(order) if d.is_classifier]
    llm_faces = [LLM_COLOR if order[i].highest_reasoning else "white" for i in llm]

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.2), sharey=True)
    for ax, title, prefix, error_type in [
        (axes[0], "Hallucination", "hall", "hallucination"),
        (axes[1], "Omission", "omis", "omission"),
    ]:
        f1 = np.array([r[f"{prefix}_f1"] for r in rows])
        low = np.array([r[f"{prefix}_f1_ci_lower"] for r in rows])
        high = np.array([r[f"{prefix}_f1_ci_upper"] for r in rows])
        iaa = agreement.set_index("error_type").loc[error_type, "pairwise_f1"]

        ax.errorbar(
            f1,
            y,
            xerr=np.vstack([f1 - low, high - f1]),
            fmt="none",
            ecolor="#555555",
            elinewidth=1.0,
            capsize=3,
            zorder=3,
        )
        ax.scatter(
            f1[llm],
            y[llm],
            s=46,
            marker="o",
            facecolors=llm_faces,
            edgecolors=LLM_COLOR,
            linewidths=1.6,
            zorder=4,
        )
        ax.scatter(
            f1[radcheck],
            y[radcheck],
            s=52,
            marker="D",
            facecolors=RADCHECK_COLOR,
            edgecolors=RADCHECK_COLOR,
            linewidths=1.6,
            zorder=4,
        )
        ax.axvline(
            iaa, color=IAA_COLOR, linestyle="--", linewidth=1.4, zorder=2, label="Inter-annotator agreement"
        )
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_xlabel("F1 score", fontsize=AXIS_LABEL_SIZE)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.8, len(order) - 0.2)
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=TICK_SIZE)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([d.label for d in order], fontsize=TICK_SIZE)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor=LLM_COLOR,
            markeredgewidth=1.6,
            markersize=7,
            label="LLM, lowest reasoning",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=LLM_COLOR,
            markeredgecolor=LLM_COLOR,
            markersize=7,
            label="LLM, highest reasoning",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=RADCHECK_COLOR,
            markeredgecolor=RADCHECK_COLOR,
            markersize=6.5,
            label="Embedding classifier",
        ),
        plt.Line2D(
            [0], [0], color=IAA_COLOR, linestyle="--", linewidth=1.4, label="Inter-annotator agreement"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=LEGEND_SIZE,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24, wspace=0.18)
    return fig


def risk_grade_distribution(counts: pd.DataFrame) -> plt.Figure:
    """One histogram per impression source of the MedVAL grades of all generated impressions."""
    order = list(SOURCE_COLORS)
    per_source = {
        s: dict(sorted(zip(g["risk_level"], g["count"]))) for s, g in counts.groupby("impression_source")
    }
    global_max = max(c / sum(d.values()) * 100 for d in per_source.values() for c in d.values()) * 1.1

    fig, axes = plt.subplots(len(order), 1, figsize=(3, 1.8 * len(order)), sharex=True)
    for i, (ax, source) in enumerate(zip(axes, order)):
        levels = list(per_source[source])
        values = list(per_source[source].values())
        total = sum(values)
        percentages = [v / total * 100 for v in values]
        mean_risk = sum(lv * c for lv, c in zip(levels, values)) / total

        bars = ax.bar(
            levels,
            percentages,
            color=SOURCE_COLORS[source],
            alpha=0.8,
            width=1.0,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axvline(mean_risk, color="#333333", linestyle="--", linewidth=1.5, alpha=0.7, zorder=5)
        ax.text(
            mean_risk - 0.1,
            global_max * 0.90,
            f"μ={mean_risk:.1f}",
            fontsize=TICK_SIZE - 1,
            color="#333333",
            va="center",
            ha="right",
        )
        for bar, p in zip(bars, percentages):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{p:.0f}%",
                ha="center",
                va="bottom",
                fontsize=TICK_SIZE,
            )
        ax.set_title(f"{IMPRESSION_SOURCES[source]} (n = {total})", fontsize=TITLE_SIZE)
        ax.set_ylim(0, global_max)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks(levels)
        ax.tick_params(axis="both", labelsize=TICK_SIZE)
        if i == len(order) - 1:
            ax.set_xlabel("MedVAL risk level", fontsize=AXIS_LABEL_SIZE)
            ax.set_ylabel("Percentage (%)", fontsize=AXIS_LABEL_SIZE)
        else:
            ax.tick_params(axis="x", labelbottom=False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.45)
    return fig


def main() -> None:
    args = parse_args(__doc__)
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.results / "csv" / "detector_metrics.csv")
    agreement = pd.read_csv(args.results / "csv" / "agreement_overall.csv")
    detector_performance(metrics, agreement).savefig(
        out / "detector_performance.pdf", bbox_inches="tight", metadata=PDF_METADATA
    )
    counts = pd.read_csv(REPO / "derived" / "risk_grade_counts.csv")
    risk_grade_distribution(counts).savefig(
        out / "risk_grade_distribution.pdf", bbox_inches="tight", metadata=PDF_METADATA
    )


if __name__ == "__main__":
    main()
