import json
import random
from pathlib import Path

# Config
INPUT_FILE = "data/en_us_full_v2.jsonl"
OUT_KNOWN = "data/en_us_known.jsonl"
OUT_UNKNOWN = "data/en_us_unknown.jsonl"
SEED = 42

random.seed(SEED)

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_data(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} items to {path}")

def main():
    data = load_data(INPUT_FILE)
    
    # Split
    # Known: Year < 2024 or (Year == 2024 and Month < 11)
    # Actually based on distribution: All dates <= 2024-04 are Known. All dates >= 2024-12 are Unknown.
    # So we can simply split by year/month logic.
    known = []
    unknown = []
    
    for item in data:
        y, m = item['year'], item['month']
        if y > 2024 or (y == 2024 and m >= 11):
            unknown.append(item)
        else:
            known.append(item)
            
    print(f"Original Split: Known={len(known)}, Unknown={len(unknown)}")
    
    # Balance
    target_count = min(len(known), len(unknown))
    print(f"Balancing to {target_count} items each.")
    
    # User requested same size. We truncate the larger one randomly.
    if len(known) > target_count:
        known = random.sample(known, target_count)
    if len(unknown) > target_count:
        unknown = random.sample(unknown, target_count)
        
    save_data(OUT_KNOWN, known)
    save_data(OUT_UNKNOWN, unknown)

if __name__ == "__main__":
    main()
