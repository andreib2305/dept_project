from pathlib import Path

SEED = 42

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data.csv"
PLOTS_DIR = BASE_DIR / "plots"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_PATH = OUTPUT_DIR / "mlp_model.joblib"

N_DEPARTMENTS = 20
START_PERIOD = "2022-01"
N_MONTHS = 60

FEATURES = [
    "revenue", "costs", "employees", "productivity", "overdue_orders",
    "defects", "plan_percent", "profitability", "revenue_per_employee",
]

STATE_NAMES = {0: "Проблемное", 1: "Требует внимания", 2: "Устойчивое"}
STATE_COLORS = {0: "#d62728", 1: "#ff9f1c", 2: "#2ca02c"}
