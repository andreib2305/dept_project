import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .config import FEATURES, MODEL_PATH, OUTPUT_DIR, PLOTS_DIR, SEED, STATE_NAMES


def train_model(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    X = df[FEATURES].to_numpy(float)
    y = df["state"].to_numpy(int)
    idx = df.index.to_numpy()
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, idx, test_size=test_size, random_state=SEED, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                          alpha=1e-3, max_iter=1000, random_state=SEED)
    model.fit(X_tr_s, y_tr)
    bundle = {"model": model, "scaler": scaler, "X_test": X_te_s, "y_test": y_te,
              "test_index": idx_te, "n_train": len(y_tr), "n_test": len(y_te)}
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def evaluate_model(bundle: dict) -> dict:
    y_true = bundle["y_test"]
    y_pred = bundle["model"].predict(bundle["X_test"])
    labels = [0, 1, 2]
    names = [STATE_NAMES[i] for i in labels]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=names,
                                        zero_division=0),
        "y_pred": y_pred,
    }


def format_metrics(bundle: dict, m: dict) -> str:
    cm = pd.DataFrame(m["confusion"],
                      index=[f"факт: {STATE_NAMES[i]}" for i in range(3)],
                      columns=[f"прогноз: {STATE_NAMES[i]}" for i in range(3)])
    return (f"MLPClassifier(hidden_layer_sizes=(32, 16)); train = {bundle['n_train']}, "
            f"test = {bundle['n_test']}\n"
            f"Accuracy          : {m['accuracy']:.4f}\n"
            f"Precision (macro) : {m['precision']:.4f}\n"
            f"Recall    (macro) : {m['recall']:.4f}\n"
            f"F1-score  (macro) : {m['f1']:.4f}\n\n"
            f"Отчет по классам:\n{m['report']}\n"
            f"Матрица ошибок:\n{cm.to_string()}")


def plot_confusion_matrix(m: dict, show: bool = False):
    import matplotlib.pyplot as plt
    PLOTS_DIR.mkdir(exist_ok=True)
    cm = m["confusion"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    ticks = [STATE_NAMES[i] for i in range(3)]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(ticks, rotation=20)
    ax.set_yticklabels(ticks)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xlabel("Прогноз")
    ax.set_ylabel("Факт")
    ax.set_title("Матрица ошибок MLPClassifier (тест)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = PLOTS_DIR / "08_confusion_matrix.png"
    fig.savefig(path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return str(path)


def predict_proba(bundle: dict, X_row: np.ndarray):
    Xs = bundle["scaler"].transform(np.asarray(X_row, float).reshape(1, -1))
    proba = bundle["model"].predict_proba(Xs)[0]
    full = np.zeros(3)
    for cls, p in zip(bundle["model"].classes_, proba):
        full[int(cls)] = p
    return full
