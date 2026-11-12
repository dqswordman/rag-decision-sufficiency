"""Materialize paper tables and the observed regret distribution from complete results."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS, SOURCE, FIGURES = (ROOT / p for p in ("results", "source", "figures"))
ORDER = ["Fixed source", "Source ensemble", "BO9", "P1", "P1-check", "Adaptive", "BO-adaptive",
         "Adaptive-check", "Random pairs", "Hashed subsets", "All-pairs", "Truncated Walsh", "Oracle"]
LABEL = {"BO9": "Best measured (base)", "P1": "Additive (P1)", "P1-check": "P1 + one check",
         "BO-adaptive": "Best measured (pairs)", "Adaptive": "Adaptive fit", "Adaptive-check": "Adaptive + one check",
         "All-pairs": "All-pairs fit", "Oracle": "Exact utility optimum"}


def tex_rows(name, rows):
    macro = "Paper" + name.removeprefix("table_").removesuffix(".tex").capitalize() + "Rows"
    body = "\n".join(" & ".join(map(str, row)) + r" \\%" for row in rows)
    (SOURCE / name).write_text(f"\\newcommand{{\\{macro}}}{{%\n{body}\n}}\n", encoding="ascii")


def percentage(x):
    return f"{100*x:.1f}"


def ci(row, precision=3):
    return rf"${row['mean']:+.{precision}f}$ $[{row.ci_lo:+.3f},\,{row.ci_hi:+.3f}]$"


def materialize_tables():
    selection = pd.read_csv(RESULTS / "selection_overall.csv").set_index(["budget", "method"])
    gen = pd.read_csv(RESULTS / "generation_overall.csv").set_index("method")
    rows = []
    for method in ORDER:
        a, b = selection.loc[(3, method)], selection.loc[(4, method)]
        assert a.n == b.n == 1440
        f1, em = (f"{gen.loc[method].token_f1:.3f}", f"{gen.loc[method].exact_match:.3f}") if method in gen.index else ("--", "--")
        probes = f"{a.mean_probes:.2f}" if method != "Oracle" else "--"
        rows.append([LABEL.get(method, method), probes, f"{a.mean_regret:.3f}", f"{a.median_regret:.3f}",
                     f"{a.p90_regret:.3f}", percentage(a.epsilon_rate), percentage(a.exact_set_rate),
                     f"{a.mean_size:.2f}", f1, em, f"{b.mean_regret:.3f}"])
    tex_rows("table_primary.tex", rows)

    gen_pairs = pd.read_csv(RESULTS / "generation_paired.csv").set_index(["left", "right", "metric"])
    score_pairs = pd.read_csv(RESULTS / "selection_paired.csv").query("budget == 3").set_index(["left", "right", "metric"])
    focus = [("P1", "BO9"), ("P1-check", "BO9"), ("P1-check", "P1"), ("Adaptive", "BO-adaptive"),
             ("Adaptive", "P1"), ("P1-check", "Adaptive"), ("Adaptive-check", "Adaptive"),
             ("P1", "All-pairs"), ("P1", "Source ensemble")]
    rows = []
    for left, right in focus:
        name = f"{LABEL.get(left, left)} vs. {LABEL.get(right, right)}"
        rows.append([name, ci(score_pairs.loc[(left, right, "normalized_regret")], 5 if right == "All-pairs" else 3),
                     ci(gen_pairs.loc[(left, right, "token_f1")]), ci(gen_pairs.loc[(left, right, "exact_match")])])
    tex_rows("table_paired.tex", rows)

    geometry = pd.read_csv(RESULTS / "geometry_overall.csv").set_index(["domain_budget", "metric"])
    tex_rows("table_geometry.tex", [[label, *[percentage(geometry.loc[(d, measure), "mean"])
                                              for measure in ("material_negative_marginal_rate", "material_positive_hessian_rate")]]
                                    for d, label in [(3, "$b=3$"), (4, "$b=4$"), (8, "Full cube")]])

    nested = pd.read_csv(RESULTS / "nested_summary.csv").set_index(["m", "method"])
    expanded = pd.read_csv(RESULTS / "extension_summary.csv").set_index(["m", "method"])
    assert len(expanded) == 18 and (expanded.n == 40).all()
    methods = ["BO9", "P1", "P1-check", "Adaptive", "BO-adaptive", "All-pairs", "Retrieval top-b"]
    rows = []
    for method in methods:
        values = [LABEL.get(method, method)]
        for table, m in [(nested, 4), (nested, 6), (nested, 8), (expanded, 8), (expanded, 12)]:
            if (m, method) in table.index:
                row = table.loc[(m, method)]
                values.append(f"{row.mean_regret:.3f} ({row.mean_probes:.1f})")
            else:
                values.append("--")
        rows.append(values)
    tex_rows("table_pool.tex", rows)


def plot_regret_distribution():
    results = pd.read_csv(RESULTS / "m8_decisions.csv")
    figure, ax = plt.subplots(figsize=(3.35, 2.35), layout="constrained")
    curves = [("P1", "Additive", "#915122", "--"), ("BO9", "Best measured (base)", "#2b5082", "-"),
              ("P1-check", "P1 + one check", "#14715f", "-"), ("BO-adaptive", "Best measured (pairs)", "#6a4d8b", "-.")]
    for name, label, color, style in curves:
        values = results[(results.budget == 3) & (results.method == name)].normalized_regret.to_numpy()
        values.sort()
        end = np.searchsorted(values, 1, side="right")
        ax.step([*values[:end], 1], [*(np.arange(1, end + 1) / len(values)), end / len(values)], where="post",
                label=label, color=color, linestyle=style, linewidth=1.35)
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Normalized utility regret", ylabel="Fraction of choices")
    ax.axvline(.05, color="#a4aab1", linestyle=":", linewidth=.95)
    ax.grid(axis="y", alpha=.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8, length=3)
    ax.xaxis.label.set_size(8.5)
    ax.yaxis.label.set_size(8.5)
    ax.legend(loc="lower right", fontsize=7.4, frameon=True, framealpha=.96, edgecolor="#e1e4e7", handlelength=2.25)
    figure.savefig(FIGURES / "regret_distribution.pdf")
    figure.savefig(FIGURES / "regret_distribution.png", dpi=240)
    plt.close(figure)


if __name__ == "__main__":
    materialize_tables()
    plot_regret_distribution()
