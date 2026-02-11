import json
import random
from pathlib import Path
import shutil

# Config
KNOWN_PATH = Path("data/en_us_known.jsonl")
UNKNOWN_PATH = Path("data/en_us_unknown.jsonl")
TARGET_COUNT = 270
SEED = 42

random.seed(SEED)

def read_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: data.append(json.loads(line))
                except: pass
    return data

def write_jsonl(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # 1. Process Unknown (Deduplicate)
    unknown_raw = read_jsonl(UNKNOWN_PATH)
    unknown_unique = {}
    for item in unknown_raw:
        # Deduplicate by question text
        unknown_unique[item['question']] = item
    
    unknown_final = list(unknown_unique.values())
    print(f"Unknown: {len(unknown_raw)} -> {len(unknown_final)} unique items.")
    
    # 2. Process Known (Random Sample to match 270)
    known_raw = read_jsonl(KNOWN_PATH)
    # Ensure raw is unique first?
    known_unique_map = {item['question']: item for item in known_raw}
    known_unique = list(known_unique_map.values())
    
    print(f"Known (Raw): {len(known_raw)} -> {len(known_unique)} unique items.")
    
    # Sample down to 270
    if len(known_unique) > TARGET_COUNT:
        known_final = random.sample(known_unique, TARGET_COUNT)
    else:
        known_final = known_unique
    
    print(f"Known (Sampled): {len(known_final)} items.")
    
    # Save back to data files
    write_jsonl(UNKNOWN_PATH, unknown_final)
    write_jsonl(KNOWN_PATH, known_final)
    print("✅ Modified data files.")

    # 3. Filter existing runs to match the new Known list
    # Use a set of allowed questions
    allowed_known_questions = {item['question'] for item in known_final}
    
    # Scan runs/*-known/*.jsonl
    runs_dir = Path("runs")
    for known_dir in runs_dir.glob("*-known"):
        if not known_dir.is_dir(): continue
        
        print(f"Filtering results in {known_dir}...")
        for preds_file in known_dir.glob("*.jsonl"):
            # skip tagged files? No, filter them too if they exist.
            # But better to just filter preds (tagged will be re-generated if we run naming script again)
            # Actually, filter everything so consistency is kept.
            
            original_rows = read_jsonl(preds_file)
            filtered_rows = [r for r in original_rows if r.get('question') in allowed_known_questions]
            
            if len(original_rows) != len(filtered_rows):
                print(f"  - {preds_file.name}: {len(original_rows)} -> {len(filtered_rows)}")
                write_jsonl(preds_file, filtered_rows)
                
    print("✅ Done! All data and results aligned to 270 items.")

if __name__ == "__main__":
    main()
