import json
from pathlib import Path
from collections import defaultdict

def final_audit():
    # Updated to look relative to project root if run from scripts/
    root = Path("runs")
    if not root.exists():
        root = Path("../runs")
    
    report = []
    
    # 遍歷所有 模型-分組 資料夾
    dirs = sorted(list(root.glob("*-known")) + list(root.glob("*-unknown")))
    
    for d in dirs:
        model_name = d.name
        
        # 定義我們關心的主要檔案
        required_files = [
            "preds_baseline_true.jsonl",
            "preds_baseline_false.jsonl",
            "preds_baseline_insuf.jsonl",
            "preds_cot_true.jsonl",
            "preds_cot_false.jsonl",
            "preds_cot_insuf.jsonl",
            "preds_no_context_true.jsonl"
        ]
        
        dir_ok = True
        
        print(f"\nChecking {model_name}...")
        for fname in required_files:
            fpath = d / fname
            if not fpath.exists():
                print(f"  ❌ Missing: {fname}")
                dir_ok = False
                continue
                
            # 計算行數與有效回答
            lines = 0
            bad_lines = 0
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            lines += 1
                            if "SYSTEM_ERROR" in line:
                                bad_lines += 1
            except:
                print(f"  ❌ Corrupt: {fname}")
                dir_ok = False
                continue
            
            status = "✅"
            if lines != 270:
                status = f"⚠️ ({lines})"
                dir_ok = False
            if bad_lines > 0:
                status += f" [BAD: {bad_lines}]"
                dir_ok = False
                
            print(f"  {status} {fname}")

        # 檢查 Tagged File 是否存在 (這才是最終能用的數據)
        tagged_missing = 0
        for fname in required_files:
            tagged_name = fname.replace(".jsonl", "_tagged.jsonl")
            if not (d / tagged_name).exists():
                # No Context 的 tagged 檔可能還沒生出來 (如果 run_all 剛跑完還沒來得及 tag)
                # 或者 tagger 腳本有問題
                tagged_missing += 1
        
        if tagged_missing > 0:
            print(f"  ⚠️ Missing {tagged_missing} tagged files (run tagging script again?)")
        
        if dir_ok:
            report.append(f"✅ {model_name}: All Clean")
        else:
            report.append(f"❌ {model_name}: Issues Found")

    print("\n" + "="*40)
    print("FINAL AUDIT REPORT")
    print("="*40)
    for line in report:
        print(line)

if __name__ == "__main__":
    final_audit()
