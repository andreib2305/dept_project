import numpy as np
import pandas as pd

from .config import FEATURES, STATE_NAMES
from .data_generator import add_derived, compute_state
from .fuzzy_model import FuzzyStateModel
from .neural_model import predict_proba

FIELDS = [
    ("revenue", "Выручка, млн (1-20)", 9.0),
    ("costs", "Затраты, млн (0.6-18)", 7.8),
    ("employees", "Численность, чел. (10-250)", 80),
    ("productivity", "Производительность (40-180)", 110.0),
    ("overdue_orders", "Просроченные заказы (0-40)", 10),
    ("defects", "Дефекты / рекламации (0-25)", 6),
    ("plan_percent", "Выполнение плана, % (70-125)", 97.0),
]
EXAMPLES = {
    "Устойчивое (пример)": dict(revenue=14.0, costs=10.0, employees=120, productivity=150,
                                overdue_orders=3, defects=2, plan_percent=108),
    "Требует внимания (пример)": dict(revenue=9.0, costs=7.8, employees=80, productivity=110,
                                      overdue_orders=14, defects=8, plan_percent=96),
    "Проблемное (пример)": dict(revenue=4.0, costs=4.6, employees=60, productivity=60,
                                overdue_orders=30, defects=18, plan_percent=79),
}


def build_row(**v) -> pd.DataFrame:
    row = pd.DataFrame([{k: float(v[k]) for k, _, _ in FIELDS}])
    if row.at[0, "revenue"] <= 0 or row.at[0, "employees"] <= 0:
        raise ValueError("Выручка и численность должны быть больше нуля.")
    return add_derived(row)


def assess(values: dict, bundle: dict, fuzzy: FuzzyStateModel) -> dict:
    row = build_row(**values)
    fz = fuzzy.evaluate_one(row)
    proba = predict_proba(bundle, row[FEATURES].to_numpy(float)[0])
    nn_cls = int(np.argmax(proba))
    ref = int(compute_state(row).iloc[0])
    return {"row": row, "fuzzy": fz, "nn_class": nn_cls, "proba": proba, "formula": ref,
            "comment": _comment(fz["category"], nn_cls)}


def _comment(f, n):
    if f == n:
        return (f"Модели согласны: «{STATE_NAMES[f]}». Вывод можно считать надежным, "
                "решение принимать по рекомендациям для этой категории.")
    if abs(f - n) == 1:
        return (f"Небольшое расхождение: нечеткая модель — «{STATE_NAMES[f]}», нейросеть — "
                f"«{STATE_NAMES[n]}». Подразделение близко к границе классов; "
                "рекомендуется проверить динамику за несколько месяцев.")
    return (f"Существенное расхождение: нечеткая — «{STATE_NAMES[f]}», нейросеть — "
            f"«{STATE_NAMES[n]}». Нужна ручная экспертная проверка исходных данных.")


def format_assessment(res: dict) -> str:
    r = res["row"].iloc[0]
    fz = res["fuzzy"]
    lines = [
        "Рассчитанные показатели:",
        f"  рентабельность       = {r['profitability']:.2f} %",
        f"  выручка на сотрудника = {r['revenue_per_employee']:.4f} млн/чел.",
        "",
        f"Нечеткая модель : {STATE_NAMES[fz['category']]} (балл {fz['score']:.2f} из 2)",
        "  сработавшие правила (степень):",
    ]
    for deg, text in fz["fired_rules"][:4]:
        lines.append(f"    {deg:.2f}  {text}")
    lines += [
        f"Нейронная сеть  : {STATE_NAMES[res['nn_class']]}",
        "  вероятности: " + "; ".join(f"{STATE_NAMES[i]} = {p:.1%}" for i, p in enumerate(res["proba"])),
        f"Эталон по учебной формуле: {STATE_NAMES[res['formula']]}",
        "",
        "Комментарий: " + res["comment"],
    ]
    return "\n".join(lines)


def ask_user() -> dict:
    print("Введите данные подразделения (Enter — значение по умолчанию в скобках):")
    values = {}
    for key, label, default in FIELDS:
        while True:
            raw = input(f"  {label} [{default}]: ").strip().replace(",", ".")
            try:
                val = float(raw) if raw else float(default)
                if val < 0:
                    raise ValueError
                values[key] = val
                break
            except ValueError:
                print("   Нужно неотрицательное число.")
    return values
