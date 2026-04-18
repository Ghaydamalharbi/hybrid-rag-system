import time
import re
import requests
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder



def detect_hallucination(answer, context):
    answer_words = set(re.findall(r'\w+', answer.lower()))
    context_words = set(re.findall(r'\w+', context.lower()))

    if not answer_words:
        return 100

    match_ratio = len(answer_words & context_words) / len(answer_words)
    return int((1 - match_ratio) * 100)



def clean_answer(answer):
    if not answer:
        return answer

    forbidden = [
        "this document",
        "provided context",
        "refers to",
        "in general",
        "typically",
        "as described",
        "according to the context"
    ]

    for f in forbidden:
        if f in answer.lower():
            return "NOT FOUND IN DOCUMENT"

    return answer



def extract_best_sentence(context, question):
    sentences = re.split(r'[.\n]', context)

    best_sentence = ""
    best_score = 0

    q_words = set(question.lower().split())

    for s in sentences:
        s_words = set(s.lower().split())
        score = len(q_words & s_words)

        if score > best_score and len(s.split()) > 6:
            best_score = score
            best_sentence = s.strip()

    return best_sentence


def run_model(model_name, prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=60
        )

        if res.status_code == 200:
            return res.json().get("response", "").strip()

        return None
    except:
        return None



def run_models_parallel(models, prompt):

    results = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(func, prompt): name
            for name, func in models
        }

        for future in futures:
            name = futures[future]
            try:
                result = future.result(timeout=60)
                results.append((name, result))
            except:
                results.append((name, None))

    return results



def hybrid_retrieve(vectorstore, question):

    docs_and_scores = vectorstore.similarity_search_with_score(question, k=8)
    docs = [doc for doc, _ in docs_and_scores]

    corpus = [d.page_content for d in docs]
    tokenized = [c.split() for c in corpus]

    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(question.split())

    combined = list(zip(docs, bm25_scores))
    combined = sorted(combined, key=lambda x: x[1], reverse=True)

    docs = [d for d, _ in combined[:5]]

    return docs



def rerank(question, docs):

    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    pairs = [(question, d.page_content) for d in docs]

    scores = model.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [d for d, _ in ranked[:3]]



def answer_question(vectorstore, question: str):

    start_time = time.time()
    logs = []

   
    docs = hybrid_retrieve(vectorstore, question)

    if not docs:
        return {"error": "NO DOCUMENTS FOUND"}

    docs = rerank(question, docs)

   
    context = "\n\n".join([
        d.page_content.strip()[:500]
        for d in docs if d.page_content.strip()
    ])

    pages = [d.metadata.get("page") for d in docs]

    if len(context) < 100:
        return {"error": "LOW CONTEXT"}

   
    extracted = extract_best_sentence(context, question)

    if extracted and len(extracted.split()) > 8:
        return {
            "answer": extracted,
            "model": "extraction",
            "pages": pages,
            "performance": {
                "latency": round(time.time() - start_time, 2),
                "hallucination": 0,
                "context_length": len(context),
                "docs_used": len(docs)
            },
            "logs": ["direct extraction used"]
        }

   
    def is_definition_question(question):
        keywords = ["what is", "define", "meaning", "definition"]
        return any(k in question.lower() for k in keywords)

    if is_definition_question(question):
        prompt = f"""
You are a strict document QA system.

RULES:
- Use ONLY the context
- DO NOT add generic phrases
- DO NOT say: "this document", "provided context"
- DO NOT generalize
- Answer using ONLY words from context
- If not found → NOT FOUND IN DOCUMENT

Context:
{context}

Question:
{question}

Answer:
Provide a precise definition in 1-2 sentences.
"""
    else:
        prompt = f"""
You are an extraction system.

RULES:
- Copy ONLY exact sentences from context
- DO NOT rephrase
- DO NOT add explanations
- If not found → NOT FOUND IN DOCUMENT

Context:
{context}

Question:
{question}

Answer:
"""

   
    models = [
        ("mistral", lambda p: run_model("mistral", p)),
        ("llama3", lambda p: run_model("llama3", p)),
    ]

    results = run_models_parallel(models, prompt)

    best_answer = None
    best_model = None
    best_hallucination = 100

  
    for model_name, answer in results:

        if not answer:
            continue

        hallucination = detect_hallucination(answer, context)

        logs.append(f"{model_name} hallucination={hallucination}")

        if hallucination < best_hallucination:
            best_hallucination = hallucination
            best_answer = answer
            best_model = model_name

  
    best_answer = clean_answer(best_answer)

    if not best_answer:
        best_answer = "NOT FOUND IN DOCUMENT"

    if best_hallucination > 35:
        best_answer = "NOT FOUND IN DOCUMENT"

  
    end_time = time.time()

    return {
        "answer": best_answer,
        "model": best_model,
        "pages": pages,
        "performance": {
            "latency": round(end_time - start_time, 2),
            "hallucination": best_hallucination,
            "context_length": len(context),
            "docs_used": len(docs)
        },
        "logs": logs
    }
