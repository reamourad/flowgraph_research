"""
plots_multimodel.py — Multi-model figures and LaTeX table for RQ1 + RQ3
========================================================================
Loads results for all 4 models and produces:

  fig_pass_at_1.pdf/png   — Pass@1: Direct vs Mermaid, all 4 models
  fig_codebleu.pdf/png    — CodeBLEU: Direct vs Mermaid, all 4 models
  fig_tokens.pdf/png      — Token overhead per model (RQ3)
  table_multimodel.tex    — LaTeX booktabs table (paste into Overleaf)
  table_multimodel.pdf/png — PNG preview of the table
  summary_multimodel.csv  — Raw numbers

Usage:
    python plots_multimodel.py --out figures_multimodel/
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILES = {
    "GPT-OSS 120B":  "results_multimodel/results_gpt-oss-120b.jsonl",
    "Llama 3 70B":   "results_multimodel/results_llama3-70b.jsonl",
    "Llama 4 Scout": "results_multimodel/results_llama4-scout.jsonl",
    "Qwen3 32B":     "results_multimodel/results_qwen3-32b.jsonl",
}

MODEL_SHORT = {
    "GPT-OSS 120B":  "GPT-OSS\n120B",
    "Llama 3 70B":   "Llama 3\n70B",
    "Llama 4 Scout": "Llama 4\nScout",
    "Qwen3 32B":     "Qwen3\n32B",
}

APPROACH_PALETTE = {
    "direct":  "#2196F3",
    "mermaid": "#FF5722",
}
APPROACH_LABELS = {
    "direct":  "Direct (baseline)",
    "mermaid": "Mermaid (proposed)",
}

MODEL_ORDER = list(MODEL_FILES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all() -> pd.DataFrame:
    frames = []
    for model_name, path in MODEL_FILES.items():
        records = [json.loads(l) for l in open(path)]
        df = pd.json_normalize(records)
        df.columns = [
            c.replace("metrics.", "").replace("token_usage.", "")
            for c in df.columns
        ]
        if "total_tokens" not in df.columns:
            pt = df.get("prompt_tokens", pd.Series(0, index=df.index))
            ct = df.get("completion_tokens", pd.Series(0, index=df.index))
            df["total_tokens"] = pt + ct
        df["total_tokens"] = df["total_tokens"].fillna(0)
        df["model_name"] = model_name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined[combined["approach"].isin(["direct", "mermaid"])].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pass_at_1(df: pd.DataFrame) -> float:
    rates = []
    for _, grp in df.groupby("task_id"):
        passes = grp["passed"].tolist()
        n, c = len(passes), sum(passes)
        k = 1
        if n < k:
            rates.append(float(c > 0))
        elif n - c < k:
            rates.append(1.0)
        else:
            rates.append(1.0 - math.comb(n - c, k) / math.comb(n, k))
    return float(np.mean(rates))


def bootstrap_ci(values, n_boot=2000, ci=0.95):
    rng = np.random.default_rng(42)
    boot = [np.mean(rng.choice(values, size=len(values), replace=True))
            for _ in range(n_boot)]
    lo = np.percentile(boot, (1 - ci) / 2 * 100)
    hi = np.percentile(boot, (1 + ci) / 2 * 100)
    return lo, hi


def set_style():
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "figure.dpi":        150,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
    })


def stars(pval: float) -> str:
    if pval < 0.001: return "***"
    if pval < 0.01:  return "**"
    if pval < 0.05:  return "*"
    return "ns"


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Pass@1 grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_pass_at_1(df: pd.DataFrame, out: Path):
    approaches = ["direct", "mermaid"]
    n_models   = len(MODEL_ORDER)
    bar_w      = 0.35
    gap        = 0.1
    group_w    = len(approaches) * bar_w + gap
    x_centers  = np.arange(n_models) * (group_w + 0.3)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for j, ap in enumerate(approaches):
        vals, lows, highs = [], [], []
        for model in MODEL_ORDER:
            sub = df[(df["model_name"] == model) & (df["approach"] == ap)]
            p1 = pass_at_1(sub)
            task_rates = sub.groupby("task_id")["passed"].mean().values.astype(float)
            lo, hi = bootstrap_ci(task_rates)
            vals.append(p1)
            lows.append(p1 - lo)
            highs.append(hi - p1)

        xpos = x_centers + j * bar_w - bar_w / 2 * (len(approaches) - 1)
        bars = ax.bar(xpos, vals, width=bar_w,
                      color=APPROACH_PALETTE[ap], label=APPROACH_LABELS[ap],
                      zorder=3, alpha=0.9)
        ax.errorbar(xpos, vals,
                    yerr=[lows, highs],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
        for x, v in zip(xpos, vals):
            ax.text(x, v + 0.025, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
    ax.set_ylabel("Pass@1")
    ax.set_ylim(0, 1.12)
    ax.set_title("Pass@1 by Model and Approach (HumanEval, 164 tasks)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out / "fig_pass_at_1.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_pass_at_1.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Saved fig_pass_at_1")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — CodeBLEU grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_codebleu(df: pd.DataFrame, out: Path):
    approaches = ["direct", "mermaid"]
    n_models   = len(MODEL_ORDER)
    bar_w      = 0.35
    gap        = 0.1
    group_w    = len(approaches) * bar_w + gap
    x_centers  = np.arange(n_models) * (group_w + 0.3)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for j, ap in enumerate(approaches):
        vals, lows, highs, sig_labels = [], [], [], []
        for model in MODEL_ORDER:
            sub = df[(df["model_name"] == model) & (df["approach"] == ap)]
            v = sub["code_bleu"].mean()
            lo, hi = bootstrap_ci(sub["code_bleu"].values)
            vals.append(v)
            lows.append(v - lo)
            highs.append(hi - v)

            if ap == "mermaid":
                direct_sub = df[(df["model_name"] == model) & (df["approach"] == "direct")]
                _, p = stats.mannwhitneyu(
                    sub["code_bleu"].values,
                    direct_sub["code_bleu"].values,
                    alternative="two-sided"
                )
                sig_labels.append(stars(p))
            else:
                sig_labels.append("")

        xpos = x_centers + j * bar_w - bar_w / 2 * (len(approaches) - 1)
        ax.bar(xpos, vals, width=bar_w,
               color=APPROACH_PALETTE[ap], label=APPROACH_LABELS[ap],
               zorder=3, alpha=0.9)
        ax.errorbar(xpos, vals,
                    yerr=[lows, highs],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
        for x, v, sig in zip(xpos, vals, sig_labels):
            ax.text(x, v + 0.015, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
            if sig and sig != "ns":
                ax.text(x, v + 0.065, sig,
                        ha="center", va="bottom", fontsize=9, color="#BF360C")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
    ax.set_ylabel("CodeBLEU")
    ax.set_ylim(0, 1.05)
    ax.set_title("CodeBLEU by Model and Approach  (* p<0.05, ** p<0.01, *** p<0.001 vs Direct)")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out / "fig_codebleu.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_codebleu.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Saved fig_codebleu")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Token usage per model (RQ3)
# ─────────────────────────────────────────────────────────────────────────────

def fig_tokens(df: pd.DataFrame, out: Path):
    approaches = ["direct", "mermaid"]
    n_models   = len(MODEL_ORDER)
    bar_w      = 0.35
    gap        = 0.1
    group_w    = len(approaches) * bar_w + gap
    x_centers  = np.arange(n_models) * (group_w + 0.3)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    all_means = {}
    for j, ap in enumerate(approaches):
        vals, lows, highs = [], [], []
        for model in MODEL_ORDER:
            sub = df[(df["model_name"] == model) & (df["approach"] == ap)]
            v = sub["total_tokens"].values.astype(float).mean()
            lo, hi = bootstrap_ci(sub["total_tokens"].values.astype(float))
            vals.append(v)
            lows.append(v - lo)
            highs.append(hi - v)
        all_means[ap] = dict(zip(MODEL_ORDER, vals))

        xpos = x_centers + j * bar_w - bar_w / 2 * (len(approaches) - 1)
        ax.bar(xpos, vals, width=bar_w,
               color=APPROACH_PALETTE[ap], label=APPROACH_LABELS[ap],
               zorder=3, alpha=0.9)
        ax.errorbar(xpos, vals,
                    yerr=[lows, highs],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
        for x, v in zip(xpos, vals):
            ax.text(x, v + 15, f"{int(v)}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    for i, model in enumerate(MODEL_ORDER):
        d = all_means["direct"][model]
        m = all_means["mermaid"][model]
        if d > 0:
            mult = m / d
            ax.text(x_centers[i], max(d, m) + max(d, m) * 0.12,
                    f"×{mult:.1f}", ha="center", fontsize=9,
                    color="gray", style="italic")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER])
    ax.set_ylabel("Avg tokens per task")
    ax.set_title("Token Overhead by Model (RQ3)  — ×N = Mermaid / Direct ratio")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out / "fig_tokens.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_tokens.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Saved fig_tokens")


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV + stats
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for ap in ["direct", "mermaid"]:
            sub = df[(df["model_name"] == model) & (df["approach"] == ap)]
            p1 = pass_at_1(sub)
            rows.append({
                "model":       model,
                "approach":    ap,
                "pass_at_1":   round(p1, 3),
                "codebleu_mean": round(sub["code_bleu"].mean(), 3),
                "codebleu_std":  round(sub["code_bleu"].std(), 3),
                "cc_delta_mean": round(sub["cc_delta"].mean(), 3),
                "cc_delta_std":  round(sub["cc_delta"].std(), 3),
                "loc_delta_mean": round(sub["loc_delta"].mean(), 3),
                "loc_delta_std":  round(sub["loc_delta"].std(), 3),
                "avg_tokens":    int(sub["total_tokens"].mean()),
                "n_tasks":       sub["task_id"].nunique(),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table — multi-model comparison
# ─────────────────────────────────────────────────────────────────────────────

def _tex(s: str) -> str:
    return (
        s.replace("&",  r"\&")
         .replace("%",  r"\%")
         .replace("±",  r"$\pm$")
         .replace("–",  "--")
         .replace("***", r"$^{***}$")
         .replace("**",  r"$^{**}$")
         .replace("* ",  r"$^{*}$ ")
         .replace("*\n", r"$^{*}$\n")
         .replace("ns",  r"\textit{ns}")
    )


def save_latex_table(df: pd.DataFrame, out: Path):
    L = [
        r"% Requires: \usepackage{booktabs,multirow,xcolor} in preamble",
        r"",
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \small",
        (r"  \caption{Multi-model HumanEval Round-Trip Results (164 tasks). "
         r"Best value per metric pair in \textbf{bold}. "
         r"Mermaid vs.\ Direct significance (Mann--Whitney $U$, two-sided): "
         r"$^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}"),
        r"  \label{tab:multimodel_results}",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{ll r r r r r}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Approach} & \textbf{Pass@1} & \textbf{CodeBLEU} & \textbf{CC $\Delta$} & \textbf{LOC $\Delta$} & \textbf{Avg Tokens} \\",
        r"    \midrule",
    ]

    for model in MODEL_ORDER:
        first = True
        for ap in ["direct", "mermaid"]:
            sub = df[(df["model_name"] == model) & (df["approach"] == ap)]
            p1 = pass_at_1(sub)
            cbleu_m = sub["code_bleu"].mean()
            cbleu_s = sub["code_bleu"].std()
            cc_m    = sub["cc_delta"].mean()
            cc_s    = sub["cc_delta"].std()
            loc_m   = sub["loc_delta"].mean()
            loc_s   = sub["loc_delta"].std()
            tokens  = sub["total_tokens"].mean()

            sig_bleu = sig_cc = sig_loc = ""
            if ap == "mermaid":
                d_sub = df[(df["model_name"] == model) & (df["approach"] == "direct")]
                _, p_b = stats.mannwhitneyu(sub["code_bleu"].values, d_sub["code_bleu"].values, alternative="two-sided")
                _, p_c = stats.mannwhitneyu(sub["cc_delta"].values,  d_sub["cc_delta"].values,  alternative="two-sided")
                _, p_l = stats.mannwhitneyu(sub["loc_delta"].values,  d_sub["loc_delta"].values, alternative="two-sided")
                sig_bleu = stars(p_b)
                sig_cc   = stars(p_c)
                sig_loc  = stars(p_l)

            d_sub2 = df[(df["model_name"] == model) & (df["approach"] == "direct")]
            m_sub2 = df[(df["model_name"] == model) & (df["approach"] == "mermaid")]
            d_p1   = pass_at_1(d_sub2)
            m_p1   = pass_at_1(m_sub2)
            d_bleu = d_sub2["code_bleu"].mean()
            m_bleu = m_sub2["code_bleu"].mean()
            d_cc   = abs(d_sub2["cc_delta"].mean())
            m_cc   = abs(m_sub2["cc_delta"].mean())
            d_loc  = abs(d_sub2["loc_delta"].mean())
            m_loc  = abs(m_sub2["loc_delta"].mean())
            d_tok  = d_sub2["total_tokens"].mean()
            m_tok  = m_sub2["total_tokens"].mean()

            def maybe_bold(val_str, is_best):
                return f"\\textbf{{{val_str}}}" if is_best else val_str

            is_best_p1    = (ap == "mermaid") == (m_p1 > d_p1)
            is_best_bleu  = (ap == "mermaid") == (m_bleu > d_bleu)
            is_best_cc    = (ap == "mermaid") == (m_cc < d_cc)
            is_best_loc   = (ap == "mermaid") == (m_loc < d_loc)
            is_best_tok   = (ap == "direct")  # lower is better for tokens

            p1_str    = maybe_bold(f"{p1:.3f}", is_best_p1)
            bleu_cell = f"{cbleu_m:.3f}$\\pm${cbleu_s:.3f}"
            if sig_bleu and sig_bleu != "ns":
                bleu_cell += f"$^{{{sig_bleu.replace('*', r'*')}}}$"
            bleu_str  = maybe_bold(bleu_cell, is_best_bleu)
            cc_cell   = f"{cc_m:+.2f}$\\pm${cc_s:.2f}"
            if sig_cc and sig_cc != "ns":
                cc_cell += f"$^{{{sig_cc.replace('*', r'*')}}}$"
            cc_str    = maybe_bold(cc_cell, is_best_cc)
            loc_cell  = f"{loc_m:+.1f}$\\pm${loc_s:.1f}"
            if sig_loc and sig_loc != "ns":
                loc_cell += f"$^{{{sig_loc.replace('*', r'*')}}}$"
            loc_str   = maybe_bold(loc_cell, is_best_loc)
            tok_str   = maybe_bold(f"{int(tokens)}", is_best_tok)

            ap_label = "Direct" if ap == "direct" else "Mermaid"
            if first:
                model_cell = f"\\multirow{{2}}{{*}}{{\\textbf{{{model}}}}}"
                first = False
            else:
                model_cell = ""

            L.append(f"    {model_cell} & {ap_label} & {p1_str} & {bleu_str} & {cc_str} & {loc_str} & {tok_str} \\\\")

        L.append(r"    \addlinespace")

    L += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    tex = "\n".join(L)
    tex_path = out / "table_multimodel.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print("  Saved table_multimodel.tex")
    print()
    print("─" * 72)
    print(tex)
    print("─" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# PNG preview table
# ─────────────────────────────────────────────────────────────────────────────

def save_png_table(summary: pd.DataFrame, out: Path):
    rows = []
    for model in MODEL_ORDER:
        for ap in ["direct", "mermaid"]:
            r = summary[(summary["model"] == model) & (summary["approach"] == ap)].iloc[0]
            rows.append([
                model if ap == "direct" else "",
                ap.capitalize(),
                f"{r.pass_at_1:.3f}",
                f"{r.codebleu_mean:.3f} ±{r.codebleu_std:.3f}",
                f"{r.cc_delta_mean:+.2f} ±{r.cc_delta_std:.2f}",
                f"{r.loc_delta_mean:+.1f} ±{r.loc_delta_std:.1f}",
                str(r.avg_tokens),
            ])

    col_labels = ["Model", "Approach", "Pass@1", "CodeBLEU", "CC Δ", "LOC Δ", "Avg Tokens"]
    fig, ax = plt.subplots(figsize=(14, len(rows) * 0.52 + 1.2))
    ax.axis("off")

    row_colors = []
    for i, row in enumerate(rows):
        ap = row[1].lower()
        base = "#E3F2FD" if ap == "direct" else "#FBE9E7"
        row_colors.append([base] * len(col_labels))

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=row_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Multi-Model HumanEval Round-Trip Results\n"
        "(shaded blue = Direct, shaded orange = Mermaid)",
        fontsize=10, pad=8
    )
    fig.tight_layout()
    fig.savefig(out / "table_multimodel.pdf", bbox_inches="tight")
    fig.savefig(out / "table_multimodel.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  Saved table_multimodel.pdf / .png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures_multimodel")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("\nLoading all model result files...")
    df = load_all()
    print(f"  {len(df)} records | {df['model_name'].nunique()} models | "
          f"{df['task_id'].nunique()} tasks per model\n")

    set_style()

    print("Generating figures...")
    fig_pass_at_1(df, out)
    fig_codebleu(df, out)
    fig_tokens(df, out)

    print("\nBuilding summary table...")
    summary = build_summary(df)
    summary.to_csv(out / "summary_multimodel.csv", index=False)
    print("  Saved summary_multimodel.csv")

    save_png_table(summary, out)

    print("\nGenerating LaTeX table...")
    save_latex_table(df, out)

    print(f"\nAll outputs saved to ./{args.out}/")
    print("  Recommended for report:")
    print("    fig_pass_at_1.pdf  — RQ1 correctness")
    print("    fig_codebleu.pdf   — RQ1 structural fidelity")
    print("    fig_tokens.pdf     — RQ3 overhead")
    print("    table_multimodel.tex — paste into Overleaf")


if __name__ == "__main__":
    main()
