import numpy as np
import pandas as pd

from .config import (DATA_PATH, N_DEPARTMENTS, N_MONTHS, SEED, START_PERIOD)

COLUMNS = [
    "period", "department", "revenue", "costs", "employees", "productivity",
    "overdue_orders", "defects", "plan_percent", "profitability",
    "revenue_per_employee", "state",
]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["profitability"] = ((df["revenue"] - df["costs"]) / df["revenue"] * 100).round(2)
    df["revenue_per_employee"] = (df["revenue"] / df["employees"]).round(4)
    return df


def _points(values, good, ok, lower_is_better=False):
    v = np.asarray(values, dtype=float)
    if lower_is_better:
        return np.where(v <= good, 2, np.where(v <= ok, 1, 0))
    return np.where(v >= good, 2, np.where(v >= ok, 1, 0))


def compute_state(df: pd.DataFrame) -> pd.Series:
    total = (
        _points(df["profitability"], 20, 8)
        + _points(df["plan_percent"], 100, 90)
        + _points(df["overdue_orders"], 8, 20, lower_is_better=True)
        + _points(df["defects"], 5, 12, lower_is_better=True)
    )
    state = np.where(total >= 6, 2, np.where(total <= 3, 0, 1))
    state = np.where(df["profitability"] < 0, 0, state)
    return pd.Series(state, index=df.index, dtype=int)


def generate_dataset(n_departments=N_DEPARTMENTS, n_months=N_MONTHS,
                     start=START_PERIOD, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods = pd.period_range(start, periods=n_months, freq="M").astype(str)
    quality = rng.permutation(np.linspace(0.18, 0.92, n_departments))
    base_emp = rng.integers(25, 250, n_departments)
    trend = rng.normal(0, 0.12, n_departments)
    phase = rng.uniform(0, 2 * np.pi, n_departments)

    t = np.arange(n_months)
    frames = []
    for d in range(n_departments):
        h = (quality[d] + trend[d] * (t / n_months - 0.5)
             + 0.08 * np.sin(2 * np.pi * t / 12 + phase[d])
             + rng.normal(0, 0.10, n_months))
        h = np.clip(h, 0.02, 0.98)
        emp = np.clip(np.round(base_emp[d] * (1 + rng.normal(0, 0.03, n_months))), 10, 250)
        prod = np.clip(40 + 140 * h + rng.normal(0, 8, n_months), 40, 180)
        revenue = np.clip(emp * prod * 0.0006 * rng.normal(1, 0.10, n_months), 1, 20)
        margin = -0.12 + 0.48 * h + rng.normal(0, 0.05, n_months)
        costs = np.clip(revenue * (1 - margin), 0.6, 18)
        overdue = np.clip(np.round(40 * (1 - h) ** 1.3 * rng.uniform(0.6, 1.3, n_months)), 0, 40)
        defects = np.clip(np.round(25 * (1 - h) ** 1.5 * rng.uniform(0.5, 1.4, n_months)), 0, 25)
        plan = np.clip(70 + 55 * h + rng.normal(0, 5, n_months), 70, 125)
        frames.append(pd.DataFrame({
            "period": periods,
            "department": f"Цех {d + 1:02d}",
            "revenue": revenue.round(2), "costs": costs.round(2),
            "employees": emp.astype(int), "productivity": prod.round(1),
            "overdue_orders": overdue.astype(int), "defects": defects.astype(int),
            "plan_percent": plan.round(1),
        }))
    df = add_derived(pd.concat(frames, ignore_index=True))
    df["state"] = compute_state(df)
    df = df.sort_values(["period", "department"]).reset_index(drop=True)
    return df[COLUMNS]


def save_dataset(df: pd.DataFrame, path=DATA_PATH):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")
