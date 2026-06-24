# QTrade Support Assistant

A minimal RAG pipeline that answers customer-support questions from QTrade's help docs, cites its sources, and escalates to a human when it shouldn't answer.

---

## How to run

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Start an LLM (choose one)

**Option A — Ollama (fully local, recommended)**

```bash
# Install Ollama from https://ollama.com, then:
ollama pull llama3.2
ollama serve          # runs on http://localhost:11434 by default
```

**Option B — Groq API (free tier, recommended)**

Sign up at **console.groq.com** (free, no credit card), create an API key, then:

```bash
pip install groq
set GROQ_API_KEY=your-groq-key-here   # Windows
# export GROQ_API_KEY=your-groq-key-here  # macOS/Linux
```

**Option C — Anthropic API (paid)**

```bash
pip install anthropic
set ANTHROPIC_API_KEY=your-key-here
```

The assistant tries Ollama first, then Groq, then Anthropic — whichever key is set.

### 3. Run

```bash
# Interactive mode
python cli.py

# Single-shot
python cli.py --query "How do I reset my SmartHub?"

# Custom docs directory
python cli.py --docs path/to/docs
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `GROQ_API_KEY` | *(unset)* | Enables Groq free-tier backend |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model to use |
| `ANTHROPIC_API_KEY` | *(unset)* | Enables Anthropic backend (paid) |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model to use |

---

## Sample Queries & Expected Terminal Output

Run `python cli.py` and try the queries below. They cover every behaviour the assistant can produce: grounded answers with citations, safety escalation, human-request escalation, and no-answer escalation.

---

### Answerable questions — grounded answers from the docs

**1. Returns with a fee**
```
<img width="1454" height="187" alt="image" src="https://github.com/user-attachments/assets/b2b82c3c-ea10-4b14-b3f2-e800d194c325" />

```

**2. SmartHub reset**
<img width="1471" height="244" alt="image" src="https://github.com/user-attachments/assets/f9b390e4-6f3f-4156-a547-a086c6292584" />


**3. Shipping time and cost**
```
<img width="1439" height="193" alt="image" src="https://github.com/user-attachments/assets/5a23d54e-debc-41d3-b5d0-fef8e27c81be" />

```

**4. Wi-Fi connection**
```
<img width="1369" height="189" alt="image" src="https://github.com/user-attachments/assets/d6b173db-e4f5-4ea4-9b2e-bd85c5cb535a" />

```

**5. Warranty coverage**
```
<img width="1096" height="210" alt="image" src="https://github.com/user-attachments/assets/9413db7d-ad46-4d25-9b93-605080dbd3bd" />

```

**6. Filing a warranty claim**
```
<img width="1472" height="212" alt="image" src="https://github.com/user-attachments/assets/f4f39d55-1fa5-45c2-9143-49e7ca386b83" />

```

**7. Final sale and defective items (tricky)**
```
<img width="891" height="164" alt="image" src="https://github.com/user-attachments/assets/394f054c-b5c3-4667-8d96-ed552bd23087" />

```

**8. Order not shipped after 4 days (partial)**
```
<img width="1477" height="207" alt="image" src="https://github.com/user-attachments/assets/02e7f238-e9e2-4c2f-90d6-2925768e7192" />

```

---

### Safety escalations — routed to human immediately

**9. Burning smell / overheating**
```
<img width="1476" height="187" alt="image" src="https://github.com/user-attachments/assets/bf23cfc8-b717-4f88-a36f-bc790ee716c4" />

```

**10. Smoke and sparks**
```
<img width="1472" height="164" alt="image" src="https://github.com/user-attachments/assets/de95c7d4-e493-4c47-98d7-1346558843be" />

```

---

### Human-request escalations — customer asks for a person

**11. Angry customer demanding a manager**
```
<img width="1139" height="141" alt="image" src="https://github.com/user-attachments/assets/148d18c2-6208-408f-be30-d975b4e911e4" />

```

**12. Asking for a real person**
```
<img width="1014" height="142" alt="image" src="https://github.com/user-attachments/assets/6bc6a1f0-ae97-4ba3-a86b-00d1b0cbe2ae" />

```

---

### No-answer escalations — topic not in the docs

**13. Bulk discounts**
```
<img width="1278" height="146" alt="image" src="https://github.com/user-attachments/assets/b6b222b9-e58d-4690-afe9-760f7017aadb" />

```

**14. Cryptocurrency payment**
```
<img width="1244" height="140" alt="image" src="https://github.com/user-attachments/assets/5fd79a95-cd9f-498b-999b-69224de26c96" />

```

---

## What I built

### Architecture

```
docs/*.txt
    │
    ▼ ingest.py        — load + sentence-window chunking (3 sentences, 1 overlap)
    ▼ embedder.py      — all-MiniLM-L6-v2 via sentence-transformers (local, free)
    ▼ vector_store.py  — in-memory NumPy cosine-similarity store
    │
user query
    ▼ embedder.py      — same model, single query vector
    ▼ vector_store.py  — top-3 cosine search
    ▼ retriever.py     — filter by grounding threshold (≥ 0.35)
    ▼ escalation.py    — safety / human-request / no-answer checks
    ▼ llm.py           — Ollama or Anthropic, cite-only prompt
    ▼ cli.py           — prints answer + sources, or escalation tag
```

### Key decisions and tradeoffs

**Chunking**: sentence-window with size=3 and overlap=1. The docs are short (one paragraph each), so sentence-level granularity avoids burying an answer inside a large chunk while overlap preserves cross-sentence context. A token-count splitter would be more robust at scale.

**Embedding model**: `all-MiniLM-L6-v2` — 80 MB, runs on CPU in ~50 ms per batch, strong retrieval quality for English support text. No GPU or API key required.

**Vector store**: plain NumPy `(N, D) @ query` dot product on L2-normalised vectors equals cosine similarity. For 4 docs (~20 chunks) this is instant and has zero operational overhead. At scale (see design write-up) this would move to a proper ANN index.

**Grounding threshold (0.35)**: empirically, relevant chunks score ≥ 0.40 and off-topic queries score ≤ 0.25 on this corpus. The 0.35 midpoint avoids both false escalations and hallucinated answers.

**LLM prompt**: the system prompt forbids the model from using its training knowledge and mandates a `Sources:` line. This makes the citation behaviour deterministic regardless of model choice.

**Escalation rule (primary)**: safety keyword detection. If the customer's message contains signals of a physical hazard (burning, hot, smoke, sparks, fire, melting, smell), the assistant refuses to answer and routes to a human immediately. Physical safety cannot be addressed by a support doc; a wrong or delayed answer could cause harm.

### Appendix B sanity-check results

| Query | Expected | Behaviour |
|---|---|---|
| "I opened the box, can I still return it, and is there a fee?" | Answer | ✅ Answers with 15% restocking fee from Returns & Refunds |
| "How do I reset my SmartHub?" | Answer | ✅ Answers with 10-second button hold from SmartHub doc |
| "My order hasn't shipped in 4 days, where is it?" | Partial / escalate | ✅ Cites 3-day contact rule; escalates if no tracking found |
| "My SmartHub is getting very hot and smells like burning." | Escalate (safety) | ✅ `[ESCALATED — SAFETY]` — safety keyword match |
| "This is the third time I've called, I want a refund and a manager NOW." | Escalate | ✅ `[ESCALATED — HUMAN REQUESTED]` — "manager" keyword |
| "Do you offer bulk discounts for commercial installs?" | I don't know / escalate | ✅ `[ESCALATED — NO GROUNDED ANSWER]` — below threshold |

---

## Design write-up

### 1. A different escalation schedule

**Implemented**: immediate escalation on safety keywords, explicit human requests, or no grounded answer.

**Alternative — confidence-decay schedule**: the assistant answers but silently tracks a "frustration score" that increments on signals like repeated questions, negative sentiment, or multiple turns without resolution. It escalates only when the score crosses a threshold (e.g., after 3 failed turns). A first-contact deflection rate close to 100% is possible; the customer only sees a human when truly stuck.

*What improves*: fewer unnecessary escalations for simple queries; agents handle genuinely hard cases. Lower cost per conversation.

*What gets worse*: a customer in a dangerous situation (e.g., burning device) who phrases it neutrally might not trip the safety keyword check fast enough. The first escalation round might also feel like the bot is stonewalling before handing off. The multi-turn state also adds system complexity.

---

### 2. Building for scale

**At 10× documents (~40 docs, ~200 chunks)**
Nothing breaks architecturally. NumPy brute-force stays sub-millisecond; sentence-transformers batching handles larger corpora. The main risk is retrieval precision drop as the corpus grows and more chunks compete for the top-k slots. Fix: raise top-k, add MMR (max-marginal-relevance) re-ranking, and tune the grounding threshold.

**At 100× documents (~400 docs, ~2 000 chunks) and concurrent customers**
- **Vector store**: replace NumPy with an ANN index (FAISS, Hnswlib, or a hosted store like Pinecone / Weaviate). Brute-force stays fast but ANN shaves latency and scales to millions of vectors.
- **Embedding model**: move to a GPU instance or a hosted embedding API to handle concurrent requests; the local CPU model becomes a bottleneck at > ~10 RPS.
- **LLM**: a shared Ollama instance becomes a queue bottleneck. Move to an inference endpoint (vLLM, Replicate, or Anthropic API) behind a load balancer.
- **State / session**: the current assistant is stateless. At scale, multi-turn conversations need a session store (Redis or a DB) to track escalation history and frustration signals.
- **Ingestion pipeline**: move from in-process indexing at startup to an offline pipeline (e.g., Airflow or a simple cron job) that re-indexes docs when they change and writes to the shared vector store.

---

### 3. Quality & regressions

**Measuring answer quality**
- Build a golden test set (query → expected answer + expected source) from Appendix B and real support tickets.
- Score each answer on: *retrieval recall* (did the right chunks come back?), *answer faithfulness* (is the answer supported by the retrieved context — checked with an LLM-as-judge), and *citation accuracy* (does `Sources:` match the actual chunk origin?).
- Track escalation rate and false-escalation rate (a human reviews escalated tickets labelled "should have answered").

**Catching regressions before customers**
- Run the golden test set on every pull request (pytest + LLM-as-judge). Gate merges on ≥ 95% faithfulness.
- Shadow traffic in staging: mirror a fraction of production queries, compare answers against baseline, alert on semantic drift.
- Monitor live: log the top-1 cosine score for every query. A sudden drop in mean score signals either a corpus change or a model regression.

---

### 4. Deployment & monitoring

**Containerisation**
Wrap the assistant in a FastAPI service (one `/ask` endpoint). Build a Docker image: base image `python:3.12-slim`, copy source, install deps, preload the embedding model at image build time (so startup is < 1 s). Run with Gunicorn + Uvicorn workers.

```
qtrade-assistant:latest
  ├── /app/src/
  ├── /app/docs/          (baked in, or mounted as a volume for live updates)
  └── embedding model cache (preloaded)
```

**Logging**
Structured JSON logs (via `structlog` or Python's `logging` with a JSON formatter):
- `query_id`, `query_text`, `top_score`, `sources_cited`, `escalated`, `escalation_reason`, `latency_ms`
- Ship to a log aggregator (Datadog, Loki, CloudWatch).

**Alerts**
| Metric | Alert threshold | Why |
|---|---|---|
| `escalation_rate` | > 40% over 5 min | Corpus gap or retrieval regression |
| `p95_latency_ms` | > 5 000 ms | LLM or embedding bottleneck |
| `top_score_mean` | < 0.30 over 1 h | Retrieval quality drop |
| `error_rate` | > 1% | LLM or Ollama service failure |
| `safety_escalation_count` | any spike | Product safety event — page on-call immediately |

---

## What I'd do next

1. **Stretch: handoff summary** — generate a 3-line summary (what was asked, what's known, why escalated) attached to the escalated ticket so the human agent has instant context.
2. **Confidence threshold tuning** — collect 50 real queries, label expected outcomes, grid-search the threshold.
3. **Evaluation harness** — pytest suite over all Appendix B queries with LLM-as-judge faithfulness scoring.
4. **FastAPI endpoint** — `/ask` POST → `{answer, sources, escalated, reason}` JSON so the CLI and a future chat widget share one backend.
5. **Offline ingestion pipeline** — watch `docs/` for changes, re-chunk and re-embed only changed files, persist the index to disk.
