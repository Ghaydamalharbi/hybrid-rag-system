# Hybrid RAG System — Controlled Document Intelligence Engine

## Abstract

This project presents a Retrieval-Augmented Generation (RAG) system designed for **high-reliability document understanding** under constrained context.
The system focuses on minimizing hallucination through structured retrieval, ranking, and strict answer filtering.

It implements a hybrid pipeline combining semantic search, keyword matching, cross-encoder re-ranking, and multi-model validation.

---

## Problem Statement

Large Language Models tend to generate plausible but unsupported answers when operating without grounded context.
This becomes critical in domains such as legal, regulatory, and enterprise documents where **precision and traceability** are required.

The core problem addressed:

* Uncontrolled generation outside document scope
* Weak retrieval relevance
* Lack of verification between model outputs and source context

---

## System Objective

Design a system that:

* Anchors responses strictly to document content
* Reduces hallucination through measurable signals

* Improves retrieval precision using hybrid techniques
* Provides transparent performance metrics

---

## System Architecture

The system follows a layered pipeline:

User Interface → API Layer → Retrieval → Ranking → Context Construction → Generation → Validation

### Flow Description

1. User uploads document via Streamlit interface
2. FastAPI processes ingestion and indexing
3. Hybrid retrieval combines:

   * Vector similarity (semantic)
   * BM25 (keyword relevance)
4. Cross-encoder re-ranking refines document selection
5. Context compression extracts high-signal segments
6. Multiple LLMs generate candidate answers
7. Hallucination scoring filters unreliable outputs
8. Best answer is returned with performance metadata

---
<img width="1666" height="6463" alt="untitled@2x (1)" src="https://github.com/user-attachments/assets/ab3a8485-a216-421f-8c98-99c11836451e" />

<img width="7032" height="776" alt="untitled@2x (2)" src="https://github.com/user-attachments/assets/d13e8fb5-4613-43bc-9db7-bb67b2c0fe21" />
## Retrieval Strategy

### Hybrid Retrieval

The system integrates:

* Dense retrieval using embeddings
* Sparse retrieval using BM25

This improves recall and relevance across different query types.
---


### Re-ranking

A cross-encoder model evaluates:

* Query–context semantic alignment

Only top-ranked chunks are passed forward.

---

## Generation Strategy

Multi-model execution is used:

* Mistral
* LLaMA3

Each model generates an answer independently.

---

## Hallucination Control

A deterministic scoring function is applied:

* Token overlap between answer and context
* Penalization for unsupported terms

Threshold-based filtering ensures:

* High-risk answers are rejected
* Fallback response: `"NOT FOUND IN DOCUMENT"`

---

## Context Compression

To improve efficiency and reduce noise:

* Only top-ranked chunks are selected
* Each chunk is truncated to high-signal segments

This reduces latency and improves model focus.

---

## Performance Metrics

Each response includes:

* Latency (execution time)
* Hallucination score
* Context length
* Number of documents used

This enables system observability and iterative improvement.

---

## Technology Stack

* FastAPI (API layer)
* Streamlit (UI)
* LangChain (orchestration)
* ChromaDB (vector storage)
* Sentence Transformers (embeddings + re-ranking)
* Ollama (local LLM execution)

---

## System Characteristics

* Local-first architecture (no external API dependency)
* Deterministic filtering layer
* Modular pipeline design
* Extensible retrieval and ranking components

---

## Example Use Cases

* Legal document analysis
* Policy and compliance systems
* Enterprise knowledge retrieval
* Internal document Q&A systems




---
### Results
<img width="1863" height="870" alt="Screenshot 2026-04-15 214346" src="https://github.com/user-attachments/assets/b574e773-bd8a-4714-8e59-8c04545fc0e6" />
<img width="1832" height="837" alt="Screenshot 2026-04-15 214330" src="https://github.com/user-attachments/assets/353b66ab-23fb-4a6d-8914-03343d21c74a" />


---

## Project Structure

```
rag-system/
│
├── api.py
├── app.py
├── rag/
│   ├── ingestion/
│   ├── processing/
│   ├── retrieval/
│   └── engine/
│
├── assets/
│   ├── architecture.png
│   ├── result.png
│
└── README.md
```

---

## Running the System

```
uvicorn api:app --reload
streamlit run app.py
```

---

## Author

AI Engineer focused on building reliable RAG systems with controlled generation and measurable performance.
