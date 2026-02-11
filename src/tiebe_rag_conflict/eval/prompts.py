OPEN_QA_PROMPT = """You are a helpful assistant. 
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
FinalAnswer:"""


ONLY_QUESTION_PROMPT = """
You are a helpful assistant. Answer the question using ONLY your own knowledge.
Rules:
- If you truly do not know the answer, or if your knowledge does not provide enough information → reply exactly with "NOT_SUPPORTED".
- Do not explain or add extra words. Output only the final answer.

Question:
{question}
FinalAnswer:
"""

COT_REASONING_PROMPT = """
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

"""

EVIDENCE_ONLY_PROMPT = """
You are a careful factual question-answering system.
Rules:
- Use the provided CONTEXT as evidence.
- Think step by step internally, but do NOT reveal your reasoning.
- Paraphrase the relevant information; do not copy more than 5 consecutive words from the context.
- If the evidence is insufficient or conflicting, answer exactly "NOT_SUPPORTED".
- Output must be a single concise answer string. Do not output JSON, explanation, or multiple fields.

Question: {question}

Context:
{context}

Task: Provide a single short factual answer based only on the context. 
If context is empty or not enough, reply "NOT_SUPPORTED".
Paraphrase in your own words.
FinalAnswer:
"""

PROMPTS = {
    "baseline": OPEN_QA_PROMPT,         # P_Baseline in paper
    "cot": COT_REASONING_PROMPT,        # P_CoT in paper
    "no_context": ONLY_QUESTION_PROMPT, # No Context baseline
    "evidence_only": EVIDENCE_ONLY_PROMPT,
    
    # Aliases for backward compatibility or clarity
    "openqa": OPEN_QA_PROMPT,           # Default to Baseline behavior
}


def get_prompt(mode: str) -> str:
    try:
        return PROMPTS[mode]
    except KeyError as exc:  # pragma: no cover - guarded by CLI
        raise ValueError(f"Unknown mode: {mode}") from exc

