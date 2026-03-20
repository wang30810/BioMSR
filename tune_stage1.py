import os
import json
import optuna
import subprocess

GRAPH_PATH = "data/final/final_graph_data.pt"
OUT_ROOT = "artifacts/tuning_stage1"
os.makedirs(OUT_ROOT, exist_ok=True)

# -------- 分工在这里改 --------
# 你用这个搜索空间：
LR_LOW, LR_HIGH = 3e-4, 1e-3
# 队友改成：
# LR_LOW, LR_HIGH = 1e-3, 5e-3
# ------------------------------

results = []

def objective(trial):
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    lr = trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)

    run_name = f"trial_{trial.number}_hd{hidden_dim}_lr{lr:.2e}_dp{dropout:.2f}"
    out_dir = os.path.join(OUT_ROOT, run_name)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "train_model.py",
        "--graph", GRAPH_PATH,
        "--epochs", "30",
        "--hidden-dim", str(hidden_dim),
        "--lr", str(lr),
        "--dropout", str(dropout),
        "--out-dir", out_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise optuna.exceptions.TrialPruned()

    summary_path = os.path.join(out_dir, "summary.json")
    summary = {}

    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    auc = summary.get("best_val", {}).get("auc", 0)
    aupr = summary.get("best_val", {}).get("aupr", 0)

    results.append({
        "run": run_name,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "dropout": dropout,
        "auc": auc,
        "aupr": aupr
    })

    return aupr  # 优化目标

study = optuna.create_study(
    direction="maximize",
    study_name="stage1_yours",
    storage=f"sqlite:///{OUT_ROOT}/study.db",
    load_if_exists=True
)

study.optimize(objective, n_trials=20)

# 保存所有 trial 结果
with open(os.path.join(OUT_ROOT, "tuning_results.json"), "w") as f:
    json.dump(results, f, indent=4)

print("\n===== Stage1 Done =====")
print("Optimal parameters:", study.best_params)
print(f"Best AUPR: {study.best_value:.4f}")
