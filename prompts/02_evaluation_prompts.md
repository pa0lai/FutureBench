# Evaluation Prompts

These prompts are used in `src/tiebe_rag_conflict/eval/prompts.py` to evaluate model performance under different RAG conditions.

## 1. Baseline RAG Prompt ($P_{Baseline}$)

**Goal:** Standard RAG setting where the model is simply asked to answer using context + knowledge.

**Template:**
You are a helpful assistant. 
Answer the question concisely, using BOTH the provided context and your own knowledge.
- The provided context may be complete, incomplete, irrelevant, or even incorrect.  
Rules:
- If the context and your knowledge agree → give the correct answer.  
- If the context contradicts your knowledge → provide the answer you believe is correct 
- If you truly do not know the answer, or if neither the context nor your knowledge provides enough information → reply exactly with "NOT_SUPPORTED".  
- Do not explain or add extra words. Output only the final answer.

Question:
{question}
Context:
{context}
FinalAnswer:

## 2. Chain-of-Thought Prompt ($P_{CoT}$)

**Goal:** Encourage the model to reason about potential conflicts before answering.

**Template:**
You are a helpful assistant. Answer the question concisely, using BOTH the provided context and your own knowledge.

Think step-by-step internally before providing your final answer:
1.  First, analyze the Question and your internal knowledge. What is the likely correct answer?
2.  Second, analyze the provided Context. What answer does the context suggest?
3.  Third, compare your internal knowledge with the context. Do they agree? Does the context *contradict* your knowledge? Or is one (or both) insufficient?

Rules for your FINAL answer:
- If the context and your knowledge agree -> give the correct answer.
- If the context *contradicts* your knowledge -> use your best judgment to provide the most accurate answer.
- If you truly do not know the answer, or if neither the context nor your knowledge provides enough information -> reply exactly with "NOT_SUPPORTED".
- Do not explain or add extra words. Output only the final answer.

Question:
{question}
Context:
{context}
FinalAnswer:

## 3. No Context Prompt

**Goal:** Assess internal knowledge without retrieval.

**Template:**
You are a helpful assistant. Answer the question using ONLY your own knowledge.
Rules:
- If you truly do not know the answer, or if your knowledge does not provide enough information → reply exactly with "NOT_SUPPORTED".
- Do not explain or add extra words. Output only the final answer.

Question:
{question}
FinalAnswer:
