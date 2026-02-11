# Data Generation Prompts

This document outlines the exact LLM prompts used in the data generation pipeline (`data_pipeline/data_generation_pipeline.py`) to create high-quality evidence triplets.

## 1. False Context Generation ($C_F$)

**Function:** `generate_contradictory_context_llm`
**Goal:** Generate a highly plausible but factually incorrect news snippet that contradicts the ground truth answer.

**Prompt Template:**
```text
Given the following Question, Gold Answer, and a True Context snippet from a news report:

Question: "{question}"
Gold Answer: "{answer_gold}"
True Context: "{ctx_true}"

Your task is to generate:
1. A *new*, detailed news report snippet that *directly contradicts* the Gold Answer (creating a False Context). 
2. The specific "False Answer" that claims to be true in your new snippet.

Constraints:
- **LENGTH REQUIREMENT:** The generated snippet MUST be roughly the same length (in words) as the "True Context" provided above. If the True Context is long, your False Context must also be long and detailed.
- The snippet MUST be directly relevant to the Question.
- The snippet MUST NOT contain the Gold Answer.
- Avoid simple negations like "not [Gold Answer]" if possible; modify names, numbers, dates, or outcomes.
- The snippet should sound authoritative and detailed, like a real news article.
- The "False Answer" must be concise (like the Gold Answer).

Output match ONLY this JSON format:
{{
    "context": "The generated contradictory text snippet...",
    "answer": "The specific false answer string"
}}
```

## 2. Insufficient Context Generation ($C_I$)

**Function:** `generate_insufficient_context_llm`
**Goal:** Generate a snippet that discusses the relevant topic/entity but *omits* the specific answer to the question.

**Prompt Template:**
```text
Given the following Question, Gold Answer, and a True Context snippet:

Question: "{question}"
Gold Answer: "{answer_gold}"
True Context: "{ctx_true}"

Your task is to generate a *new* news report snippet (Insufficient Context) that is:
1. **Topically Relevant** to the Question (discusses the same entities/events).
2. **Does NOT contain the answer**. It should provide background info but fail to answer the specific question.

Constraints:
- **LENGTH REQUIREMENT:** The generated snippet MUST be roughly the same length (in words) as the "True Context" provided above. 
- The snippet MUST be authoritative and detailed (news style).
- **CRITICAL:** Do NOT include the Gold Answer or any information that directly implies it. The context must be "insufficient" to answer the question.

Output ONLY the generated JSON:
{{
    "context": "The generated insufficient text snippet..."
}}
```
