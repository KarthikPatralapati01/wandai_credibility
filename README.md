
# 📊 Information Credibility System for LLM Research

This project evaluates and updates the credibility of claims extracted from diverse sources (news, peer-reviewed studies, PRs, ads, etc.). It enables **initial credibility scoring** and **incremental updates** with new files (like Pitchbook reports), while ensuring:
- ✅ High-quality updated research
- ⚡ Minimal execution time
- 💻 Minimal computational overhead

> 🎯 Submission for rubric: Ideation + Planning, Code & Build, Strategy Review (see below)

---

## 📂 Project Structure

```
.
├── src/
│   └── credibility_proto.py         # Main prototype script
├── data/
│   └── (Input source files: .txt)
├── outputs/
│   ├── report_initial.txt
│   ├── report_incremental.txt
│   ├── incremental_summary.txt
│   └── initial_scores.json
├── ideation_and_planning.txt       # Rubric Section 1
├── strategy_review.txt             # Rubric Section 3
├── README.md
├── requirements.txt
```

---

## ✅ How to Run

### Option 1: Run Initial Analysis
```bash
python3 src/credibility_proto.py
# Choose: 1
```
- Scans all `.txt` files in `data/`
- Extracts claims, computes credibility scores
- Saves report to `outputs/report_initial.txt`

---

### Option 2: Incremental Update with New File
```bash
python3 src/credibility_proto.py
# Choose: 2
```
- Adds a new `.txt` file (e.g., `data/new_update.txt`)
- Input publisher and type (news, peer_review, etc.)
- Re-scores only related claims
- Generates:
  - ✅ Updated report (`outputs/report_incremental.txt`)
  - ✅ Summary of changed/new claims (`outputs/incremental_summary.txt`)
  - ✅ LLM-generated explanations

> 💡 Set `OPENAI_API_KEY` in your environment to enable LLM summaries.

---

## 🧠 Rubric Sections

### 1. Ideation + Planning ✅  
→ See [`ideation_and_planning.txt`](./ideation_and_planning.txt)  
Includes:  
- Architecture  
- Agent design  
- KPIs  
- Infra sketch  
- Edge cases

---

### 2. Code & Build ✅  
✔ 24-hour working prototype  
✔ Modular, testable script  
✔ CLI interface for Option 1 and 2  
✔ Supports semantic sim, selective recomputation, LLM explanations

---

### 3. Strategy Review ✅  
→ See [`strategy_review.txt`](./strategy_review.txt)  
Covers:
- 🔄 How to scale (vector DB, agents, registry)
- 💰 Monetization (API, plugin, dashboards)
- ⚡ Cost optimization (embedding cache, LLM caps)
- 🚀 Shipping to prod (CI/CD, APIs, observability)

---

## 🛠 Requirements

```
openai
sentence-transformers
scikit-learn
```

> Install via: `pip install -r requirements.txt`  
Fallback mode (no embeddings or LLM) is supported.

---

## 🚀 Future Work (Optional)

- Vector DB for matching (Qdrant)
- Contradiction detection (NLI)
- Claim versioning and lineage
- PDF/HTML table ingestion
- Streamlit web UI

---


