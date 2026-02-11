import json
import csv
from pathlib import Path
from collections import defaultdict

# 定義要統計的 Mode 組合
# (Mode, CtxType) -> 顯示名稱
METRICS_TO_SHOW = [
    # 基準線 (No Context)
    ("no_context", "true", "NoCtx-True"),
    
    # 標準 RAG (Baseline)
    ("baseline", "true",  "Base-True"),
    ("baseline", "false", "Base-False"),
    
    # 思維鏈 RAG (CoT)
    ("cot", "false", "CoT-False"),
]

def analyze_folder(folder_path):
    stats = {}
    
    # 掃描該資料夾下所有 tagged 檔案
    for f in folder_path.glob("*_tagged.jsonl"):
        # 解析檔名: preds_{mode}_{ctx}_tagged.jsonl
        name_parts = f.name.replace("preds_", "").replace("_tagged.jsonl", "").split("_")
        # 處理 mode 有底線的情況 (如 no_context)
        if "no_context" in f.name:
            mode = "no_context"
            ctx = name_parts[-1]
        else:
            mode = name_parts[0]
            ctx = name_parts[1]
            
        key = (mode, ctx)
        
        labels = []
        with open(f, "r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    try:
                        item = json.loads(line)
                        labels.append(item.get("label", "Unknown"))
                    except: pass
        
        total = len(labels)
        if total == 0: continue
        
        counts = defaultdict(int)
        for l in labels:
            counts[l] += 1
            
        stats[key] = {
            "N": total,
            "Stick": counts["Stick"] / total * 100,
            "Follow": counts["Follow"] / total * 100,
            "Reject": counts["Reject"] / total * 100,
            "Wrong": counts["Wrong"] / total * 100,
            # 對於 True Context，Stick 就是 Accuracy
            "Acc": counts["Stick"] / total * 100
        }
    return stats

def main():
    root = Path("runs")
    if not root.exists():
        root = Path("../runs")
    results = []
    
    # 遍歷所有模型資料夾
    folders = sorted(list(root.glob("*-known")) + list(root.glob("*-unknown")))
    
    for folder in folders:
        if not folder.is_dir(): continue
        
        # 解析 Model 和 Split
        # 假設資料夾格式: {model_name}-{split}
        name_parts = folder.name.rsplit("-", 1)
        if len(name_parts) != 2:
            print(f"Skipping weird folder: {folder.name}")
            continue
            
        model_name = name_parts[0]
        split = name_parts[1] # known / unknown
        
        print(f"Processing {model_name} [{split}]...")
        stats = analyze_folder(folder)
        
        row = {
            "Model": model_name,
            "Split": split
        }
        
        # 填入數據
        for mode, ctx, label_prefix in METRICS_TO_SHOW:
            s = stats.get((mode, ctx), {})
            if not s:
                continue
                
            # 根據不同情境選擇顯示的欄位
            if ctx == "true":
                # True Context 主要看 Accuracy (Stick) 和 Reject
                row[f"{label_prefix} Acc"] = f"{s['Acc']:.1f}%"
                row[f"{label_prefix} Rej"] = f"{s['Reject']:.1f}%"
            else:
                # False Context 主要看 Stick (抗干擾), Follow (被騙), Reject
                row[f"{label_prefix} Stick"] = f"{s['Stick']:.1f}%"
                row[f"{label_prefix} Follow"] = f"{s['Follow']:.1f}%"
                row[f"{label_prefix} Rej"] = f"{s['Reject']:.1f}%"

        results.append(row)

    # 輸出 CSV
    if not results:
        print("No results found!")
        return

    # 動態決定所有欄位名稱
    fieldnames = ["Model", "Split"]
    # 預先定義順序
    for _, _, label in METRICS_TO_SHOW:
        if "True" in label:
            fieldnames.extend([f"{label} Acc", f"{label} Rej"])
        else:
            fieldnames.extend([f"{label} Stick", f"{label} Follow", f"{label} Rej"])
            
    # 只保留有資料的欄位 (Optional) - 這裡我們保留全部以求整齊
    
    out_csv = "experiment_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n✅ Summary saved to {out_csv}")
    
    # Print preview
    print("-" * 100)
    print(f"{'Model':<20} {'Split':<10} {'Base-False Stick':<20} {'Base-False Follow':<20}")
    print("-" * 100)
    for r in results:
        print(f"{r.get('Model', ''):<20} {r.get('Split', ''):<10} {r.get('Base-False Stick', '-'):<20} {r.get('Base-False Follow', '-'):<20}")

if __name__ == "__main__":
    main()
