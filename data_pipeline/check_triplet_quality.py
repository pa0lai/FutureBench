import pandas as pd
import json
import glob
import os
import time
from openai import OpenAI, RateLimitError, APIError
from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm.notebook import tqdm # Use notebook tqdm for better progress bar in notebooks
from tqdm import tqdm # Use standard tqdm if running as a script
from collections import defaultdict


# --- 1. 設定 ---
TRIPLETS_DIR = "data/" # 你存放 _triplets.jsonl 的資料夾
OUTPUT_CSV = "llm_quality_report_v2.csv" # 輸出的報告檔名 (v2)
LLM_MODEL = "gpt-4o-mini" # 使用的模型 (確保你有權限)
MAX_WORKERS = 10 # 同時呼叫 API 的執行緒數量 (可依你的速率限制調整)
MAX_RETRIES = 3 # API 呼叫失敗時的重試次數
INITIAL_DELAY = 1 # 重試前的初始延遲 (秒)

# 限制處理的項目數量 (-1 表示處理全部，建議先設為 10 測試)
MAX_ITEMS_TO_PROCESS = -1
# --- 設定結束 ---

# --- OpenAI Client ---
try:
    client = OpenAI()
    print("OpenAI client initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY environment variable is set correctly.")
    exit()

# --- LLM 判斷函式 ---
def check_context_with_llm(question, answer_gold, context, context_type):
    """使用 LLM 判斷單一上下文的品質 (v2 加入 is_valid_insuf)"""
    
    # --- v2: 修改 Prompt ---
    prompt = f"""
    You are an expert data quality checker for evaluating contexts used in Retrieval-Augmented Generation (RAG) benchmarks.
    Your task is to analyze the provided context based on the question and the gold answer.

    Question: "{question}"
    Gold Answer: "{answer_gold}"
    Context Type: {context_type}
    Context: "{context}"

    Please evaluate the context based on the following criteria and respond ONLY with a valid JSON object containing boolean values (true/false) or null for each key:

    1.  "contains_answer": Does the Context *directly* contain the information stated in the Gold Answer? (Focus on semantic meaning, ignore minor wording differences).
    2.  "is_relevant": Is the main topic of the Context *directly relevant* to the main topic of the Question? (Ignore boilerplate/noise).
    3.  "has_noise": Does the Context contain significant amounts of non-informative text like HTML tags, website navigation elements, error messages (e.g., 404, expired), copyright notices, login prompts, or excessive ads/promotions?
    4.  "is_contradictory": (Only evaluate if Context Type is 'ctx_false') Does the Context state information that *directly contradicts* the Gold Answer? If Context Type is not 'ctx_false', set this to null.
    5.  "is_valid_insuf": (Only evaluate if Context Type is 'ctx_insuf') Is the Context BOTH relevant ("is_relevant": true) AND does NOT contain the answer ("contains_answer": false)? If Context Type is not 'ctx_insuf', set this to null.

    Example JSON Response Format (for ctx_insuf):
    {{
      "contains_answer": false,
      "is_relevant": true,
      "has_noise": false,
      "is_contradictory": null,
      "is_valid_insuf": true 
    }}
    Example JSON Response Format (for ctx_true or ctx_false):
     {{
      "contains_answer": boolean,
      "is_relevant": boolean,
      "has_noise": boolean,
      "is_contradictory": boolean_or_null,
      "is_valid_insuf": null 
    }}

    Respond only with the JSON object.
    """
    
    delay = INITIAL_DELAY
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"} # Use JSON mode
            )
            
            result_json_str = response.choices[0].message.content
            result_json = json.loads(result_json_str)
            
            # --- v2: 修改 JSON 驗證邏輯 ---
            required_keys = {"contains_answer", "is_relevant", "has_noise", "is_contradictory", "is_valid_insuf"}
            missing_keys = required_keys - set(result_json.keys())
            if missing_keys:
                raise ValueError(f"LLM response missing keys: {missing_keys}")

            # 根據 context_type 強制設定 is_contradictory 和 is_valid_insuf
            if context_type != 'ctx_false':
                if result_json.get("is_contradictory") is not None:
                   # print(f"Warning: Overriding is_contradictory to null for {context_type}")
                   result_json["is_contradictory"] = None 
            elif result_json.get("is_contradictory") is None: # ctx_false 但 is_contradictory 是 null
                 raise ValueError("LLM response 'is_contradictory' should not be null for ctx_false")

            if context_type != 'ctx_insuf':
                 if result_json.get("is_valid_insuf") is not None:
                    # print(f"Warning: Overriding is_valid_insuf to null for {context_type}")
                    result_json["is_valid_insuf"] = None
            elif result_json.get("is_valid_insuf") is None: # ctx_insuf 但 is_valid_insuf 是 null
                 raise ValueError("LLM response 'is_valid_insuf' should not be null for ctx_insuf")
            # (可選) 驗證 is_valid_insuf 的邏輯
            # elif result_json["is_valid_insuf"] != (result_json.get("is_relevant", False) and not result_json.get("contains_answer", True)):
            #     print(f"Warning: LLM's is_valid_insuf logic seems inconsistent for ctx_insuf.")
            #     # 可以選擇相信 LLM 的判斷，或者強制根據 is_relevant 和 contains_answer 重新計算
            #     # result_json["is_valid_insuf"] = (result_json.get("is_relevant", False) and not result_json.get("contains_answer", True))
                 
            return result_json
            
        except RateLimitError as e:
            last_error = f"Rate limit hit ({e})"
            print(f"{last_error}, retrying in {delay}s... (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2 
        except APIError as e:
            last_error = f"API error ({e})"
            print(f"{last_error}, retrying in {delay}s... (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2
        except (json.JSONDecodeError, ValueError) as e:
             last_error = f"Error parsing LLM response ({e})"
             print(f"{last_error}. Response: {result_json_str if 'result_json_str' in locals() else 'N/A'}. Retrying...")
             time.sleep(delay) 
             delay *=2
        except Exception as e:
            last_error = f"Unexpected error ({e})"
            print(f"{last_error}. Retrying...")
            time.sleep(delay)
            delay *= 2
            
    print(f"Failed to get valid response after {MAX_RETRIES} retries for question: {question[:50]}... Last error: {last_error}")
    return { # 返回預設錯誤值
        "contains_answer": None, 
        "is_relevant": None, 
        "has_noise": None, 
        "is_contradictory": None,
        "is_valid_insuf": None, # <--- v2 新增
        "error": f"Max retries exceeded: {last_error}"
    }

# --- 讀取資料 (同 v6) ---
def load_triplets_v6():
    files = glob.glob(os.path.join(TRIPLETS_DIR, "*.jsonl"))
    if not files:
        print(f"錯誤: 在 '{TRIPLETS_DIR}' 中找不到任何 .jsonl 檔案。")
        return None
    
    all_items_data = []
    print(f"找到 {len(files)} 個 triplet 檔案。正在讀取...")
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file):
                    try:
                        item = json.loads(line)
                        item['unique_id'] = item.get('item_id', f"{os.path.basename(f)}_{line_num}")
                        required = ["question", "answer_gold", "ctx_true", "ctx_false", "ctx_insuf", "unique_id"]
                        if all(k in item for k in required):
                            all_items_data.append(item)
                        # else: # 減少囉嗦的警告
                        #     print(f"  警告: 檔案 {f} 第 {line_num+1} 行缺少必要欄位。跳過。")
                    except json.JSONDecodeError:
                        print(f"    警告: 檔案 {f} 第 {line_num+1} 行 JSON 解碼失敗。")
        except Exception as e:
            print(f"    錯誤: 讀取檔案 {f} 失敗: {e}")
            
    if not all_items_data:
        print("錯誤: 沒有讀取到任何有效的 triplet 資料。")
        return None
        
    df = pd.DataFrame(all_items_data)
    print(f"成功讀取 {len(df)} 筆資料。")
    return df

# --- 主執行流程 ---
def run_llm_checks(df):
    
    results_list = []
    items_to_process = df.head(MAX_ITEMS_TO_PROCESS) if MAX_ITEMS_TO_PROCESS > 0 else df
    total_items = len(items_to_process)
    print(f"開始使用 LLM ({LLM_MODEL}) 檢查 {total_items} 筆資料...")

    processed_count = 0
    temp_results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_info = {}
        for index, row in items_to_process.iterrows():
            item_id = row['unique_id']
            q = row['question']
            a = row['answer_gold']
            for ctx_type_short in ['true', 'false', 'insuf']:
                ctx_key = f"ctx_{ctx_type_short}"
                ctx_content = str(row.get(ctx_key, ""))
                if not ctx_content: # 跳過空的上下文，節省 API call
                    print(f"  警告: item {item_id}, {ctx_key} 是空的，跳過 LLM 檢查。")
                    temp_results.append({
                        "item_id": item_id,
                        "ctx_type": f"ctx_{ctx_type_short}",
                        "error": "Empty context"
                    })
                    continue
                    
                future = executor.submit(check_context_with_llm, q, a, ctx_content, f"ctx_{ctx_type_short}")
                future_to_info[future] = {"id": item_id, "type": f"ctx_{ctx_type_short}"}

        for future in tqdm(as_completed(future_to_info), total=len(future_to_info), desc="Processing items"):
            info = future_to_info[future]
            item_id = info["id"]
            ctx_type = info["type"]
            try:
                result_dict = future.result()
                result_dict['item_id'] = item_id
                result_dict['ctx_type'] = ctx_type
                temp_results.append(result_dict)
                # processed_count += 1 # 這個計數器不准確，因為是基於上下文數量
            except Exception as e:
                print(f"Error retrieving result for {item_id} ({ctx_type}): {e}")
                temp_results.append({
                    "item_id": item_id,
                    "ctx_type": ctx_type,
                    "error": str(e)
                })

    if not temp_results:
        print("沒有成功處理任何項目。")
        return

    llm_results_df = pd.DataFrame(temp_results)
    
    # 計算成功處理了多少 item_id
    successful_item_ids = llm_results_df[llm_results_df['error'].isna()]['item_id'].nunique()

    # 將結果與原始資料合併
    # --- v2: 新增 is_valid_insuf 欄位 ---
    llm_pivot = llm_results_df.pivot(index='item_id', columns='ctx_type', 
                                     values=['contains_answer', 'is_relevant', 'has_noise', 'is_contradictory', 'is_valid_insuf', 'error'])
    llm_pivot.columns = ['_'.join(col).strip('_') for col in llm_pivot.columns.values]
    
    report_df = pd.merge(items_to_process[['unique_id', 'question', 'answer_gold']], 
                         llm_pivot, 
                         left_on='unique_id', 
                         right_index=True, 
                         how='left')
    report_df = report_df.rename(columns={'unique_id': 'item_id'})

    # --- 輸出報告 ---
    print(f"\n檢查完成。成功處理 {successful_item_ids} / {total_items} 個項目。正在將報告寫入 {OUTPUT_CSV}...")
    report_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"詳細報告已儲存至 {OUTPUT_CSV}")
    
    # --- v2: 修改摘要統計 ---
    print("\n--- LLM 品質檢查摘要 ---")
    total_checked_items = successful_item_ids # 使用成功處理的 item 數量
    if total_checked_items == 0:
        print("沒有可供分析的結果。")
        return
        
    print(f"基於成功處理的 {total_checked_items} 個項目進行統計:")
    
    issues_summary = defaultdict(int)
    # 確保只統計沒有 API 錯誤的項目
    valid_rows = report_df[report_df['error_ctx_true'].isna() & 
                           report_df['error_ctx_false'].isna() & 
                           report_df['error_ctx_insuf'].isna()]
    
    if valid_rows.empty:
         print("沒有完整處理成功的項目可供統計。")
         return
         
    actual_stats_count = len(valid_rows)
    print(f"(共 {actual_stats_count} 個項目完整處理成功)")
    
    for index, row in valid_rows.iterrows():
        # ctx_true checks
        if row.get('contains_answer_ctx_true') == False: issues_summary['ctx_true: !contains_answer'] += 1
        if row.get('is_relevant_ctx_true') == False: issues_summary['ctx_true: !is_relevant'] += 1
        if row.get('has_noise_ctx_true') == True: issues_summary['ctx_true: has_noise'] += 1
            
        # ctx_false checks
        if row.get('contains_answer_ctx_false') == True: issues_summary['ctx_false: contains_answer'] += 1
        if row.get('is_relevant_ctx_false') == False: issues_summary['ctx_false: !is_relevant'] += 1
        if row.get('has_noise_ctx_false') == True: issues_summary['ctx_false: has_noise'] += 1
        if row.get('is_contradictory_ctx_false') == False: issues_summary['ctx_false: !is_contradictory'] += 1

        # ctx_insuf checks
        if row.get('contains_answer_ctx_insuf') == True: issues_summary['ctx_insuf: contains_answer'] += 1
        if row.get('is_relevant_ctx_insuf') == False: issues_summary['ctx_insuf: !is_relevant'] += 1
        if row.get('has_noise_ctx_insuf') == True: issues_summary['ctx_insuf: has_noise'] += 1
        # v2: 新增 is_valid_insuf 的統計
        if row.get('is_valid_insuf_ctx_insuf') == False: issues_summary['ctx_insuf: !is_valid (not relevant or contains answer)'] += 1
        elif row.get('is_valid_insuf_ctx_insuf') == True: issues_summary['ctx_insuf: is_valid (relevant & no answer)'] += 1


    if not issues_summary:
        print("所有檢查的項目看起來都符合預期。")
    else:
        print("\n潛在問題與有效性統計 (基於 LLM 判斷):")
        sorted_issues = sorted(issues_summary.items(), key=lambda item: item[0]) # 按名稱排序
        for issue, count in sorted_issues:
             print(f"  - {issue}: {count} 次 ({count/actual_stats_count:.1%})")

# --- 主程式 ---
if __name__ == "__main__":
    df_triplets = load_triplets_v6()
    if df_triplets is not None:
        run_llm_checks(df_triplets)