import numpy as np
import pandas as pd

from .config import PLOTS_DIR, STATE_NAMES


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if b <= a else (x - a) / (b - a)
    right = np.ones_like(x) if d <= c else (d - x) / (d - c)
    return np.clip(np.minimum(left, right), 0, 1)


INPUT_SETS = {
    "profitability": {
        "low": (-100, -100, 3, 12), "mid": (5, 15, 15, 25), "high": (18, 28, 100, 100)},
    "plan_percent": {
        "low": (0, 0, 85, 95), "mid": (88, 98, 98, 108), "high": (100, 110, 200, 200)},
    "overdue_orders": {
        "low": (0, 0, 5, 12), "mid": (8, 18, 18, 28), "high": (22, 30, 100, 100)},
    "defects": {
        "low": (0, 0, 3, 7), "mid": (4, 10, 10, 16), "high": (12, 18, 100, 100)},
}
LABELS = {
    "profitability": ("рентабельность", {"low": "низкая", "mid": "средняя", "high": "высокая"}),
    "plan_percent": ("выполнение плана", {"low": "низкое", "mid": "нормальное", "high": "высокое"}),
    "overdue_orders": ("просрочки", {"low": "мало", "mid": "средне", "high": "много"}),
    "defects": ("дефекты", {"low": "мало", "mid": "средне", "high": "много"}),
}

OUTPUT_SETS = {
    "problem": (0, 0, 0, 1), "attention": (0, 1, 1, 2), "stable": (1, 2, 2, 2)}
OUT_CODE = {"problem": 0, "attention": 1, "stable": 2}

RULES = [
    ({"profitability": "high", "plan_percent": "high", "overdue_orders": "low"}, "stable"),
    ({"profitability": "high", "plan_percent": "mid", "overdue_orders": "low"}, "stable"),
    ({"profitability": "high", "defects": "low", "overdue_orders": "low"}, "stable"),
    ({"profitability": "mid", "plan_percent": "mid", "overdue_orders": "low", "defects": "low"}, "stable"),
    ({"profitability": "mid", "plan_percent": "high", "overdue_orders": "low"}, "stable"),
    ({"profitability": "mid", "overdue_orders": "mid"}, "attention"),
    ({"profitability": "high", "overdue_orders": "high"}, "attention"),
    ({"profitability": "high", "defects": "high"}, "attention"),
    ({"profitability": "mid", "plan_percent": "low"}, "attention"),
    ({"profitability": "low", "plan_percent": "mid"}, "attention"),
    ({"overdue_orders": "mid", "defects": "mid"}, "attention"),
    ({"profitability": "high", "plan_percent": "low"}, "attention"),
    ({"profitability": "low", "overdue_orders": "high"}, "problem"),
    ({"profitability": "low", "plan_percent": "low"}, "problem"),
    ({"plan_percent": "low", "overdue_orders": "high"}, "problem"),
    ({"defects": "high", "overdue_orders": "high"}, "problem"),
    ({"profitability": "low", "defects": "high"}, "problem"),
    ({"profitability": "mid", "plan_percent": "low", "defects": "high"}, "problem"),
]
OUT_RU = {"problem": "проблемное", "attention": "требует внимания", "stable": "устойчивое"}


def rule_text(i):
    conds, out = RULES[i]
    parts = [f"{LABELS[v][0]} {LABELS[v][1][s]}" for v, s in conds.items()]
    return f"R{i + 1}: ЕСЛИ " + " И ".join(parts) + f" ТО {OUT_RU[out]}"


class FuzzyStateModel:
    def __init__(self, low_thr=0.95, high_thr=1.15):
        self.low_thr, self.high_thr = low_thr, high_thr
        self.u = np.linspace(0, 2, 201)
        self.out_mf = {k: trapmf(self.u, *p) for k, p in OUTPUT_SETS.items()}

    def _firing(self, df):
        mem = {v: {s: trapmf(df[v].to_numpy(float), *p) for s, p in sets.items()}
               for v, sets in INPUT_SETS.items()}
        return np.array([np.minimum.reduce([mem[v][s] for v, s in conds.items()])
                         for conds, _ in RULES])

    def score(self, df):
        F = self._firing(df)
        n = F.shape[1]
        strength = {k: np.zeros(n) for k in OUTPUT_SETS}
        for f, (_, out) in zip(F, RULES):
            strength[out] = np.maximum(strength[out], f)
        agg = np.max([np.minimum(strength[k][:, None], self.out_mf[k][None, :])
                      for k in OUTPUT_SETS], axis=0)
        area = agg.sum(axis=1)
        centroid = np.where(area > 1e-9, (agg * self.u).sum(axis=1) / np.maximum(area, 1e-9), 1.0)
        return centroid, strength

    def to_category(self, score):
        score = np.asarray(score)
        return np.where(score < self.low_thr, 0, np.where(score > self.high_thr, 2, 1))

    def predict(self, df):
        return self.to_category(self.score(df)[0])

    def evaluate_one(self, df_row):
        score, strength = self.score(df_row)
        F = self._firing(df_row)[:, 0]
        fired = sorted(((F[i], rule_text(i)) for i in range(len(RULES)) if F[i] > 0.01),
                       reverse=True)
        return {"score": float(score[0]), "category": int(self.to_category(score)[0]),
                "strength": {k: float(v[0]) for k, v in strength.items()}, "fired_rules": fired}


def plot_membership_functions(save=True):
    import matplotlib.pyplot as plt
    ranges = {"profitability": (-20, 50), "plan_percent": (70, 125),
              "overdue_orders": (0, 40), "defects": (0, 25)}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (var, sets) in zip(axes.ravel(), INPUT_SETS.items()):
        lo, hi = ranges[var]
        x = np.linspace(lo, hi, 400)
        for s, p in sets.items():
            ax.plot(x, trapmf(x, *p), lw=2, label=LABELS[var][1][s])
        ax.set_title(LABELS[var][0].capitalize())
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Функции принадлежности входных переменных")
    fig.tight_layout()
    if save:
        PLOTS_DIR.mkdir(exist_ok=True)
        fig.savefig(PLOTS_DIR / "07_fuzzy_membership.png", dpi=130)
    return fig
