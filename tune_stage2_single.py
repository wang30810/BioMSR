import os  
import json  
import subprocess  
import argparse  
  
parser = argparse.ArgumentParser()  
parser.add_argument("--config-index", type=int, default=0)  
args = parser.parse_args()  
  
GRAPH_PATH = "data/final/final_graph_data.pt"  
STAGE1_TOP = "top2_global.json"  
OUT_ROOT = "artifacts/tuning_stage2"  
  
os.makedirs(OUT_ROOT, exist_ok=True)  
  
# 读取 Top2  
with open(STAGE1_TOP, "r") as f:  
    top_configs = json.load(f)  
  
# 选择一个 config  
cfg = top_configs[args.config_index]  
  
hd = cfg["hidden_dim"]  
lr = cfg["lr"]  
dp = cfg["dropout"]  
  
run_name = f"final_hd{hd}_lr{lr}_dp{dp}"  
out_dir = os.path.join(OUT_ROOT, run_name)  
  
cmd = [  
    "python", "train_model.py",  
    "--graph", GRAPH_PATH,  
    "--epochs", "100",   # Stage2：精训  
    "--hidden-dim", str(hd),  
    "--lr", str(lr),  
    "--dropout", str(dp),  
    "--out-dir", out_dir  
]  
  
print(f"\n🚀 Running Stage2 Config {args.config_index}")  
print(cfg)  
  
subprocess.run(cmd)  
  
summary_path = os.path.join(out_dir, "summary.json")  
summary = {} # 初始化

if os.path.exists(summary_path):  
    with open(summary_path, "r") as f:  
        summary = json.load(f)  
  
result = {  
    "run": run_name,  
    "hidden_dim": hd,  
    "lr": lr,  
    "dropout": dp,  
    "auc": summary.get("best_val", {}).get("auc", 0),   # ✅ 修改处
    "aupr": summary.get("best_val", {}).get("aupr", 0)  # ✅ 修改处
}  
  
print("\n🏆 RESULT:")  
print(result)  
  
with open(os.path.join(out_dir, "result.json"), "w") as f:  
    json.dump(result, f, indent=4)
