# 📡 SignalDesk — Telecom Customer Care RAG

> A retrieval-augmented chatbot for telecom customer support, grounded in three real knowledge sources: FAQs, resolved support tickets, and the operator's PDF guide.

**SignalDesk** answers customer questions about connectivity, billing, SIM issues, and roaming by retrieving from a multi-source vector store and generating context-grounded answers with a Qwen 3 32B model via Groq.

Built with **LangChain**, **Chroma**, **HuggingFace embeddings**, and **Streamlit**.

---

## ✨ What makes it different

Most RAG demos retrieve from a single PDF. Real customer support knowledge lives in **three different shapes**, and SignalDesk handles all of them:

| Source | Format | What it adds | Collection |
|---|---|---|---|
| 📋 **FAQs** | CSV | General policy and how-to answers | `faq` |
| 🎫 **Support Tickets** | SQLite | Real resolved cases with step-by-step fixes | `tickets` |
| 📖 **Operator Guide** | PDF | Authoritative technical reference | `guides` |

Each source is ingested with its own script, embedded into its own Chroma collection, and tagged with a `source` metadata field. At query time, all three retrievers run and their results are merged. The LLM sees clearly-labeled blocks like `[TICKET]`, `[FAQ]`, `[GUIDE]` so it can prefer one source over another based on the question.

---

## 🧠 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Data sources                          │
│   ┌───────────┐     ┌──────────────┐     ┌───────────────┐   │
│   │  faq.csv  │     │  tickets.db  │     │  guide.pdf    │   │
│   └─────┬─────┘     └──────┬───────┘     └──────┬────────┘   │
│         │ ingest_faq       │ ingest_tickets     │ ingest_pdf │
│         ▼                  ▼                    ▼            │
│   ┌───────────────────────────────────────────────────────┐  │
│   │      Chroma  (3 collections, MiniLM embeddings)       │  │
│   └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼  query
              ┌─────────────────────────────────┐
              │   Multi-collection retriever    │
              │   k=3 from each source          │
              │   → merge → tag with [SOURCE]   │
              └────────────────┬────────────────┘
                               ▼
              ┌─────────────────────────────────┐
              │      Qwen 3 32B (Groq)          │
              │  System prompt enforces:        │
              │   • answer only from context    │
              │   • fall back to "call 611"     │
              │     if confidence is low        │
              └────────────────┬────────────────┘
                               ▼
                       Streamed answer
```

### How a query flows

```
"Why is my bill higher than usual this month?"
  ↓
3 retrievers fire in parallel
  ↓
[FAQ]    "Bills can be higher due to international roaming or data overages..."
[TICKET] "Issue: billing. Description: Customer charged for premium SMS.
          Resolution: Reversed the charge after verifying it was unsolicited..."
[GUIDE]  "Section 4.2 — Charges outside the monthly plan include..."
  ↓
Qwen 3 32B answers, grounded only in these three blocks
  ↓
Streams to UI token by token
```

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Orchestration | [LangChain](https://github.com/langchain-ai/langchain) (LCEL chains) |
| Vector store | [Chroma](https://github.com/chroma-core/chroma) (persistent, 3 collections) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, local) |
| LLM | [Groq](https://groq.com/) — `qwen/qwen3-32b` |
| PDF parsing | `PyPDFLoader` + `RecursiveCharacterTextSplitter` (600 char chunks, 100 overlap) |
| Frontend | [Streamlit](https://streamlit.io/) with native streaming |

**Why local embeddings?** Telecom support has unlimited query volume — paying per-token for embeddings would be prohibitive. MiniLM-L6-v2 runs on CPU, is 90MB, and gives strong-enough retrieval for short customer queries.

---

## 📁 Project structure

```
signaldesk/
├── data/
│   ├── faq.csv             # FAQ source (CSV)
│   ├── tickets.db          # Resolved tickets (SQLite)
│   └── telecom_guide.pdf   # Operator guide (PDF)
├── ingest_faq.py           # CSV → Chroma 'faq' collection
├── ingest_tickets.py       # SQLite → Chroma 'tickets' collection
├── ingest_pdf.py           # PDF → chunks → Chroma 'guides' collection
├── retriever.py            # Multi-collection retriever
├── rag_chain.py            # LCEL chain: retrieve → format → prompt → LLM
├── app.py                  # Streamlit chat UI (streaming)
├── main.py                 # CLI version of the same chatbot
├── chroma_store/           # Generated vector DB (gitignored)
├── pyproject.toml
└── .env                    # GROQ_API_KEY=...  (not committed)
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com/keys)

### 1. Install

```bash
git clone https://github.com/nikhilgoud19/signaldesk.git
cd signaldesk
pip install -r requirements.txt   # or: uv sync
```

### 2. Configure

Create `.env` in the project root:

```
GROQ_API_KEY=your_key_here
```

### 3. Build the vector store (one-time)

The Chroma store is not committed — build it from the source data:

```bash
python ingest_faq.py
python ingest_pdf.py
python ingest_tickets.py
```

This downloads the embedding model (~90MB, first time only), then embeds and persists everything to `chroma_store/`. Expect a couple of minutes total.

### 4. Run

**Streamlit UI:**
```bash
streamlit run app.py
```

**Or the CLI:**
```bash
python main.py
```

---

## 💬 Example queries

Try these from the sidebar or chat input:

- *"Why is my mobile internet so slow?"* → pulls from tickets + guide
- *"How do I activate international roaming?"* → pulls from FAQ + guide
- *"My phone shows SIM not detected after a restart"* → pulls from tickets
- *"I was charged for roaming but had a bundle active"* → pulls from FAQ + tickets

When the retriever returns nothing confident, the assistant falls back to suggesting the customer call **611** or use the operator's app — rather than hallucinating an answer.

---

## 🗺️ Roadmap

- [ ] **Hybrid search** — combine BM25 keyword retrieval with vector similarity for better recall on specific error codes and plan names
- [ ] **Reranking** — add a cross-encoder reranker to filter the top-9 retrieved chunks down to the top-3 most relevant before passing to the LLM
- [ ] **Source citations in UI** — show which FAQ entry / ticket / guide section grounded each answer
- [ ] **Eval harness** — fixed Q/A set measuring retrieval recall@k and answer faithfulness
- [ ] **Conversational memory** — currently each turn is independent; add message history so follow-ups like *"what about for prepaid?"* work
- [ ] **Live ticket feedback loop** — when a new ticket is resolved, auto-ingest it into the `tickets` collection

---

## 📚 What I learned building this

- **Separating ingestion from retrieval pays off fast.** Three independent ingest scripts mean I can rebuild one collection without touching the others — useful when the FAQ updates monthly but the PDF guide barely changes.
- **Source tagging matters more than I expected.** Wrapping retrieved chunks with `[FAQ]` / `[TICKET]` / `[GUIDE]` markers in the prompt lets the LLM weight them differently — it tends to prefer tickets for "how do I fix X" questions and the guide for policy questions, without being told to.
- **Local embeddings are the right default for RAG demos.** MiniLM-L6-v2 is small enough to ship, runs on CPU fast enough to feel instant, and removes a paid API from the loop entirely. I'd only reach for a hosted embedding model when retrieval quality measurably required it.
- **The "fall back to 611" rule is the cheap, robust hallucination control.** Telling the model what to do when context is insufficient is much more reliable than telling it not to make things up.

---
