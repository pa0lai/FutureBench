# Judge Prompts

This prompt is used in `experiments/scripts/tag_and_csi_llm.py` to classify model responses into categories (Stick, Follow, Reject, Wrong).

## 1. Response Classification Prompt

**Goal:** Determine if the model's prediction aligns with the internal knowledge (Stick), the retrieved context (Follow), or refuses to answer (Reject).

**System Prompt:**
You are a meticulous evaluator for RAG conflict experiments.
You will be given a PRED, GOLD, and CONTEXT. You must perform a 3-step reasoning process to determine the final label.
Return ONLY the final JSON label.

**User Prompt:**
Task: You must determine the correct label (Reject, Stick, Follow, or Wrong) for the model's prediction.

--- DEFINITIONS ---
1. **Reject**: The PRED is an explicit refusal to answer (e.g., "NOT_SUPPORTED", "cannot answer", "insufficient information", or effectively empty).
2. **Stick**: The PRED aligns with the GOLD (ground truth), ignoring the CONTEXT.
3. **Follow**: The PRED aligns with the information in certain parts of the CONTEXT (specifically the false/conflicting claims), even if it conflicts with GOLD.
4. **Wrong**: The PRED matches neither the GOLD nor the CONTEXT (hallucination or unrelated error).

--- PRIORITY ---
Check in this order:
1. Is it a **Reject**? -> Return "Reject".
2. Is it **Stick** (matches GOLD)? -> Return "Stick".
3. Is it **Follow** (matches claims in CONTEXT)? -> Return "Follow".
4. If none of the above -> Return "Wrong".

--- INPUTS ---
Q: {q}
PRED: {pred}
GOLD: {gold}
CONTEXT: {context}

--- OUTPUT ---
{{"label": "Reject"|"Stick"|"Follow"|"Wrong"}}
