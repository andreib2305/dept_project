import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FEATURES, PLOTS_DIR, STATE_COLORS, STATE_NAMES
from .fuzzy_model import plot_membership_functions


def make_all_plots(df: pd.DataFrame, show: bool = True):
    PLOTS_DIR.mkdir(exist_ok=True)
    saved = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cnt = df["state"].value_counts().sort_index()
    ax.bar([STATE_NAMES[i] for i in cnt.index], cnt.values,
           color=[STATE_COLORS[i] for i in cnt.index])
    for i, v in enumerate(cnt.values):
        ax.text(i, v + 5, str(v), ha="center")
    ax.set_title("Распределение состояний подразделений")
    ax.set_ylabel("Число наблюдений")
    saved.append(_save(fig, "01_state_distribution.png"))

    fig, ax = plt.subplots(figsize=(8, 6))
    m = df.groupby("department")["profitability"].mean().sort_values()
    ax.barh(m.index, m.values, color=["#d62728" if v < 8 else "#ff9f1c" if v < 20 else "#2ca02c" for v in m.values])
    ax.set_title("Средняя рентабельность по подразделениям, %")
    ax.set_xlabel("%")
    ax.grid(axis="x", alpha=0.3)
    saved.append(_save(fig, "02_profitability_by_department.png"))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    g = df.groupby("period").agg(p=("profitability", "mean"), plan=("plan_percent", "mean"))
    ax.plot(g.index, g["p"], color="#1f77b4", lw=2, label="Рентабельность, %")
    ax.set_ylabel("Рентабельность, %")
    ax2 = ax.twinx()
    ax2.plot(g.index, g["plan"], color="#9467bd", lw=2, ls="--", label="Выполнение плана, %")
    ax2.set_ylabel("Выполнение плана, %")
    ax.set_xticks(range(0, len(g), 6))
    ax.set_xticklabels(g.index[::6], rotation=45)
    ax.set_title("Динамика средних показателей по месяцам")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")
    ax.grid(alpha=0.3)
    saved.append(_save(fig, "03_monthly_dynamics.png"))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k in sorted(df["state"].unique()):
        s = df[df["state"] == k]
        ax.scatter(s["plan_percent"], s["profitability"], s=14, alpha=0.6,
                   color=STATE_COLORS[k], label=STATE_NAMES[k])
    ax.set_xlabel("Выполнение плана, %")
    ax.set_ylabel("Рентабельность, %")
    ax.set_title("Рентабельность и выполнение плана по состояниям")
    ax.legend()
    ax.grid(alpha=0.3)
    saved.append(_save(fig, "04_scatter_plan_profitability.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, col, title in zip(axes, ["overdue_orders", "defects"],
                              ["Просроченные заказы", "Дефекты / рекламации"]):
        data = [df.loc[df["state"] == k, col] for k in (0, 1, 2)]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=[STATE_NAMES[k] for k in (0, 1, 2)])
        for patch, k in zip(bp["boxes"], (0, 1, 2)):
            patch.set_facecolor(STATE_COLORS[k])
            patch.set_alpha(0.6)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    saved.append(_save(fig, "05_boxplots_risks.png"))

    fig, ax = plt.subplots(figsize=(8, 7))
    corr = df[FEATURES].corr().to_numpy()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(FEATURES, fontsize=8)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Матрица корреляций признаков")
    saved.append(_save(fig, "06_correlation_heatmap.png"))

    fig = plot_membership_functions(save=True)
    saved.append(str(PLOTS_DIR / "07_fuzzy_membership.png"))

    if show:
        plt.show()
    plt.close("all")
    return saved


def _save(fig, name):
    fig.tight_layout()
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=130)
    return str(path)
