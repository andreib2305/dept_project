import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import STATE_NAMES


def compare_methods(df, bundle, fuzzy, nn_pred) -> dict:
    test = df.loc[bundle["test_index"]]
    y = bundle["y_test"]
    fz_pred = fuzzy.predict(test)
    res = {
        "n": len(y),
        "fuzzy_acc": accuracy_score(y, fz_pred), "fuzzy_f1": f1_score(y, fz_pred, average="macro"),
        "nn_acc": accuracy_score(y, nn_pred), "nn_f1": f1_score(y, nn_pred, average="macro"),
        "agree": float(np.mean(fz_pred == nn_pred)),
        "cross": pd.crosstab(pd.Series(fz_pred, name="нечеткая"), pd.Series(nn_pred, name="нейросеть")),
    }
    return res


def format_comparison(c: dict) -> str:
    names = {i: STATE_NAMES[i] for i in range(3)}
    cross = c["cross"].rename(index=names, columns=names)
    both = "Нейросеть точнее воспроизводит учебную разметку" if c["nn_acc"] > c["fuzzy_acc"] \
        else "Нечеткая модель не уступает нейросети"
    return (f"Тестовая выборка: {c['n']} наблюдений\n"
            f"Нечеткая модель : accuracy = {c['fuzzy_acc']:.3f}, F1 (macro) = {c['fuzzy_f1']:.3f}\n"
            f"Нейронная сеть  : accuracy = {c['nn_acc']:.3f}, F1 (macro) = {c['nn_f1']:.3f}\n"
            f"Доля совпадения ответов двух моделей: {c['agree']:.1%}\n\n"
            f"Таблица сопряженности (строки — нечеткая, столбцы — нейросеть):\n{cross.to_string()}\n\n"
            f"Вывод: {both}.\n"
            "Нечеткая модель прозрачна (каждое решение объясняется правилами) и не требует обучения, "
            "но зависит от экспертных порогов. Нейросеть обучается на данных и точнее, но работает как "
            "«черный ящик». Практически целесообразно использовать обе: расхождение моделей — "
            "сигнал для ручной проверки подразделения.")
