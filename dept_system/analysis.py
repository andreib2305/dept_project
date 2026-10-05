import numpy as np
import pandas as pd

from .config import FEATURES, STATE_NAMES


def basic_report(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Размер набора: {df.shape[0]} строк x {df.shape[1]} столбцов")
    lines.append(f"Период: {df['period'].min()} ... {df['period'].max()}; "
                 f"подразделений: {df['department'].nunique()}")
    lines.append(f"Пропусков всего: {int(df.isna().sum().sum())}")
    lines.append("\nПервые 5 строк:\n" + df.head().to_string(index=False))
    lines.append("\nОписательная статистика:\n"
                 + df[FEATURES].describe().T[["mean", "std", "min", "50%", "max"]].round(2).to_string())

    dist = df["state"].value_counts().sort_index()
    lines.append("\nРаспределение классов состояния:")
    for k, v in dist.items():
        lines.append(f"  {k} - {STATE_NAMES[k]:<18} {v:>5} ({v / len(df) * 100:.1f}%)")

    lines.append("\nСредние показатели по классам состояния:\n"
                 + df.groupby("state")[["profitability", "plan_percent", "overdue_orders",
                                        "defects", "revenue_per_employee"]].mean().round(2).to_string())

    by_dep = df.groupby("department").agg(
        profitability=("profitability", "mean"), plan=("plan_percent", "mean"),
        overdue=("overdue_orders", "mean"), stable_share=("state", lambda s: (s == 2).mean() * 100))
    lines.append("\nЛучшие 3 подразделения по рентабельности:\n"
                 + by_dep.sort_values("profitability", ascending=False).head(3).round(2).to_string())
    lines.append("\nХудшие 3 подразделения по рентабельности:\n"
                 + by_dep.sort_values("profitability").head(3).round(2).to_string())

    corr = np.corrcoef(df["profitability"].to_numpy(), df["plan_percent"].to_numpy())[0, 1]
    corr2 = np.corrcoef(df["overdue_orders"].to_numpy(), df["defects"].to_numpy())[0, 1]
    p10, p50, p90 = np.percentile(df["profitability"].to_numpy(), [10, 50, 90])
    lines.append(f"\nNumPy: corr(рентабельность, план) = {corr:.3f}; corr(просрочки, дефекты) = {corr2:.3f}")
    lines.append(f"NumPy: процентили рентабельности P10/P50/P90 = {p10:.1f} / {p50:.1f} / {p90:.1f} %")
    loss = int((df["profitability"] < 0).sum())
    lines.append(f"Убыточных наблюдений (рентабельность < 0): {loss}")
    return "\n".join(lines)
