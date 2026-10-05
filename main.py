import sys

import matplotlib

if "--demo" in sys.argv:
    matplotlib.use("Agg")

from dept_system import analysis, viewer, compare, data_generator as dg, new_department as nd, plots, report
from dept_system import neural_model as nn
from dept_system.config import DATA_PATH, OUTPUT_DIR
from dept_system.fuzzy_model import FuzzyStateModel, RULES, rule_text
from dept_system.config import STATE_NAMES
import pandas as pd


class App:
    def __init__(self, show_plots=True, window = True):
        self.window = window
        self.df = None
        self.bundle = None
        self.metrics = None
        self.fuzzy = FuzzyStateModel()
        self.show = show_plots
        OUTPUT_DIR.mkdir(exist_ok=True)

    def ensure_data(self):
        if self.df is None:
            if DATA_PATH.exists():
                self.df = dg.load_dataset()
            else:
                self.generate()
        return self.df

    def ensure_model(self):
        self.ensure_data()
        if self.bundle is None:
            self.bundle = nn.load_model()
            if self.bundle is None:
                self.train()
        if self.metrics is None:
            self.metrics = nn.evaluate_model(self.bundle)
        return self.bundle

    def out(self, title, text):
        viewer.show_text(title, text, self.window)

    def generate(self):
        self.df = dg.generate_dataset()
        dg.save_dataset(self.df)
        self.bundle = self.metrics = None
        print(f"Сформировано {len(self.df)} строк, файл сохранен: {DATA_PATH}")

    def load_or_generate(self):
        if DATA_PATH.exists() and input("data.csv найден. Загрузить (Enter) или сгенерировать заново (g)? ").strip().lower() != "g":
            self.df = dg.load_dataset()
            print(f"Загружено {len(self.df)} строк из {DATA_PATH.name}")
        else:
            self.generate()

    def show_stats(self):
        text = analysis.basic_report(self.ensure_data())
        (OUTPUT_DIR / "basic_stats.txt").write_text(text, encoding="utf-8")
        self.out("Основные показатели", text)

    def show_plots(self):
        paths = plots.make_all_plots(self.ensure_data(), show=self.show)
        print("Графики сохранены:\n  " + "\n  ".join(paths))

    def check_fuzzy(self):
        df = self.ensure_data()
        pred = self.fuzzy.predict(df)
        acc = (pred == df["state"].to_numpy()).mean()
        L = [f"Правил в базе: {len(RULES)}"] + ["  " + rule_text(i) for i in range(len(RULES))]
        L.append(f"\nСовпадение нечеткой модели с учебной формулой на всех данных: {acc:.1%}")
        cross = pd.crosstab(df["state"].map(STATE_NAMES).rename("формула"),
                            pd.Series(pred, index=df.index).map(STATE_NAMES).rename("нечеткая"))
        L.append(cross.to_string())
        L.append("\nПримеры:")
        for name, vals in nd.EXAMPLES.items():
            r = self.fuzzy.evaluate_one(nd.build_row(**vals))
            L.append(f"  {name}: балл {r['score']:.2f} -> {STATE_NAMES[r['category']]}")
        self.out("Нечеткая модель", "\n".join(L))

    def train(self):
        self.ensure_data()
        self.bundle = nn.train_model(self.df)
        self.metrics = nn.evaluate_model(self.bundle)
        print(f"Нейросеть обучена. Train = {self.bundle['n_train']}, test = {self.bundle['n_test']}. "
              f"Accuracy на тесте: {self.metrics['accuracy']:.3f}")

    def show_metrics(self):
        self.ensure_model()
        text = nn.format_metrics(self.bundle, self.metrics)
        (OUTPUT_DIR / "metrics.txt").write_text(text, encoding="utf-8")
        img = nn.plot_confusion_matrix(self.metrics, show=False)
        self.out("Метрики нейронной сети", text + f"\n\nГрафик матрицы ошибок: {img}")
        if self.show:
            nn.plot_confusion_matrix(self.metrics, show=True)

    def new_department(self, values=None):
        self.ensure_model()
        if values is None:
            values = nd.ask_user()
        res = nd.assess(values, self.bundle, self.fuzzy)
        text = nd.format_assessment(res)
        self.out("Оценка нового подразделения", text)
        return text

    def compare(self):
        self.ensure_model()
        c = compare.compare_methods(self.df, self.bundle, self.fuzzy, self.metrics["y_pred"])
        text = compare.format_comparison(c)
        (OUTPUT_DIR / "comparison.txt").write_text(text, encoding="utf-8")
        self.out("Сравнение методов", text)
        return c

    def conclusion(self, example_text=None):
        self.ensure_model()
        c = compare.compare_methods(self.df, self.bundle, self.fuzzy, self.metrics["y_pred"])
        if example_text is None:
            vals = nd.EXAMPLES["Требует внимания (пример)"]
            example_text = nd.format_assessment(nd.assess(vals, self.bundle, self.fuzzy))
        path = report.write_conclusion(self.df, self.metrics, c, example_text)
        print("Итоговый вывод сохранен:", path)


MENU = """
---Оценка состояния производственных подразделений---
1. Сгенерировать / загрузить данные
2. Показать основные показатели
3. Показать графики
4. Проверить нечеткую модель
5. Обучить нейронную сеть
6. Показать метрики и матрицу ошибок
7. Оценить новое подразделение
8. Сравнить два метода
0. Выход
"""


def menu():
    app = App()
    actions = {"1": app.load_or_generate, "2": app.show_stats, "3": app.show_plots,
               "4": app.check_fuzzy, "5": app.train, "6": app.show_metrics,
               "7": app.new_department, "8": app.compare}
    while True:
        print(MENU)
        choice = input("Выберите пункт: ").strip()
        if choice == "0":
            print("Выход.")
            return
        action = actions.get(choice)
        if action is None:
            print("Нет такого пункта.")
            continue
        try:
            action()
        except (KeyboardInterrupt, EOFError):
            print("\nОперация прервана.")
        except Exception as exc:
            print(f"Ошибка: {exc}")


def demo():
    app = App(show_plots=False)
    app.generate()
    app.show_stats()
    app.show_plots()
    app.check_fuzzy()
    app.train()
    app.show_metrics()
    out = []
    for name, vals in nd.EXAMPLES.items():
        print(f"\n--- {name} ---")
        out.append(f"=== {name} ===\n" + app.new_department(vals))
    (OUTPUT_DIR / "new_department_example.txt").write_text("\n\n".join(out), encoding="utf-8")
    print()
    app.compare()
    app.conclusion()


if __name__ == "__main__":
    demo() if "--demo" in sys.argv else menu()
