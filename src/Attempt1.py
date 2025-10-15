import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any
# ---------- LLM-based Source Detection ----------
from openai import OpenAI
client = OpenAI()  # requires OPENAI_API_KEY in your environment

def infer_source_metadata(context_str: str) -> dict:
    """
    Use an LLM to infer source type and publisher from free-form context.
    Example input: 'I saw this on YouTube by JoeTechReview.'
    Output: {'source_type': 'social', 'publisher': 'JoeTechReview'}
    """
    try:
        prompt = f"""
        You are a precise parser that identifies where a statement came from.
        The user will describe the origin of some information.
        From that text, extract:
        1. source_type — one of [peer_review, gov, news, corp_pr, ad, social, unknown]
        2. publisher — name of the outlet, account, or organization.
        If uncertain, set values to 'unknown'.
        Text: "{context_str}"
        Respond in JSON only.
        """
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        import json
        content = resp.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ Could not infer source automatically: {e}")
        return {"source_type": "unknown", "publisher": "unknown"}


GLOBAL_SOURCES = []
GLOBAL_DOCS = {}
USER_DATA_FILE = "../data/user_added_sources.json"


# ---------- Data classes ----------
@dataclass
class Source:
    id: str
    type: str        # peer_review | gov | news | corp_pr | ad | social
    publisher: str
    date: str
    funding: str = ""
    path: str = ""

# ---------- Step 1: Read & attach metadata ----------
def read_sources() -> List[Source]:
    #Load documents from data/ and attach metadata describing each source.

    sources = [
        Source(id="S1", type="corp_pr", publisher="Tesla PR", date="2024-11-01", path="../data/tesla_pr.txt"),
        Source(id="S2", type="news", publisher="AutoReview", date="2025-02-10", path="../data/toyota_news.txt"),
        Source(id="S3", type="peer_review", publisher="NTSB Journal", date="2023-08-15", path="../data/ford_research.txt"),
        Source(id="S4", type="ad", publisher="Acme Marketing", date="2025-01-05", path="../data/acme_ad.txt"),
    ]
    return sources


def load_documents(sources: List[Source]) -> Dict[str, str]:
    #Read each file's text into memory.

    docs = {}
    for src in sources:
        with open(src.path, "r") as f:
            text = f.read().strip()
            docs[src.id] = text
    return docs

def load_user_added_sources():
    """
    Load any user-added sources from the JSON file (if it exists).
    """
    import json, os
    user_sources = []
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            data = json.load(f)
            for item in data:
                user_sources.append(Source(**item))
    return user_sources

from openai import OpenAI
import json

# Step 2: Claim Extractor (LLM)
client = OpenAI()

def extract_claims_llm(text: str) -> list[dict]:
    """
    Use GPT-4o-mini to extract factual or promotional claims.
    Returns list[dict] with claim, claim_type, tone, numeric fields.
    """
    prompt = f"""
    You are a precise information extraction agent.

    Extract each distinct statement or claim from the text below.
    For every claim, output a JSON list like:
    [
      {{
        "claim": "string",
        "claim_type": "factual | superlative | prediction | quote",
        "tone": "neutral | promotional | hedged",
        "numeric": true | false
      }}
    ]

    Only return JSON. Do not include commentary or markdown.
    Text:
    {text}
    """

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=0.0
    )

    raw_output = response.output_text.strip()

    # try to safely load JSON
    try:
        data = json.loads(raw_output)
    except Exception:
        try:
            json_str = raw_output.split("```json")[-1].split("```")[0].strip()
            data = json.loads(json_str)
        except Exception:
            print("⚠️ Could not parse LLM output, returning empty list.")
            print(raw_output[:200])
            data = []

    return data



import re
import math
from datetime import datetime

# ---------- Step 3: Feature Builder ----------
def compute_features(claim_obj: dict, source_type: str, source_date: str) -> dict:
    """
    Compute interpretable numeric features for credibility scoring.
    """

    # --- source prior ---
    SOURCE_PRIOR = {
        "peer_review": 0.9,
        "gov": 0.85,
        "news": 0.7,
        "corp_pr": 0.45,
        "ad": 0.25,
        "social": 0.2
    }
    source_prior = SOURCE_PRIOR.get(source_type, 0.5)

    # --- linguistic score ---
    tone = claim_obj.get("tone", "neutral").lower()
    claim_type = claim_obj.get("claim_type", "factual").lower()
    text = claim_obj.get("claim", "")
    numeric = bool(claim_obj.get("numeric", False))

    hedged_penalty = -0.1 if tone == "hedged" else 0
    promo_penalty = -0.2 if tone == "promotional" else 0
    specific_bonus = 0.1 if numeric else 0.05 if re.search(r"\d+", text) else 0
    linguistic = max(0, min(1, 0.5 + hedged_penalty + promo_penalty + specific_bonus))

    # --- recency score ---
    try:
        y = int(source_date.split("-")[0])
        current_year = datetime.now().year
        age = max(0, current_year - y)
        recency = max(0.2, 1.0 / (1 + 0.15 * age))
    except Exception:
        recency = 0.5

    # --- contextuality ---
    contextuality = 0.5
    if claim_type == "superlative" and source_type in {"ad", "corp_pr"}:
        contextuality -= 0.3
    if claim_type == "factual" and source_type in {"peer_review", "gov"}:
        contextuality += 0.3
    contextuality = max(0, min(1, contextuality))

    # --- conflict of interest (coi) ---
    coi = 0.8 if source_type in {"news", "peer_review"} else 0.5 if source_type == "corp_pr" else 0.3

    return {
        "source_prior": round(source_prior, 2),
        "linguistic": round(linguistic, 2),
        "recency": round(recency, 2),
        "contextuality": round(contextuality, 2),
        "coi": round(coi, 2),
        "corroboration": 0.0  # placeholder until we add FAISS later
    }

# ---------- Step 4: Credibility Scorer ----------

def score_claim(features: dict) -> dict:
    """
    Combine weighted features into a single credibility score.
    """

    # define weights (they sum to 1.0)
    WEIGHTS = {
        "source_prior": 0.35,
        "linguistic": 0.15,
        "corroboration": 0.25,
        "recency": 0.10,
        "contextuality": 0.10,
        "coi": 0.05,
    }

    # weighted sum
    score = 0.0
    for k, w in WEIGHTS.items():
        score += w * features.get(k, 0)

    score = round(min(max(score, 0.0), 1.0), 3)

    # band assignment
    if score >= 0.66:
        band = "HIGH"
    elif score >= 0.45:
        band = "MEDIUM"
    else:
        band = "LOW"

    return {"score": score, "band": band}



'''def build_embeddings(all_claims: list[str]) -> np.ndarray:
    """
    Compute normalized embeddings for all claims.
    """
    embeddings = embedder.encode(all_claims, normalize_embeddings=True)
    return np.array(embeddings, dtype="float32")

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a simple in-memory FAISS index (cosine similarity).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def update_corroboration_scores(claim_texts: list[str], source_ids: list[str], sources_meta: dict) -> list[float]:
    """
    For each claim, find semantically similar claims from *independent* sources.
    Returns a corroboration score in [0,1].
    """
    emb = build_embeddings(claim_texts)
    index = build_faiss_index(emb)
    sims, idxs = index.search(emb, k=3)   # top-3 neighbors

    corroboration = []
    for i, sim_list in enumerate(sims):
        score = 0.0
        for j, sim_val in enumerate(sim_list[1:], start=1):  # skip self
            if sim_val > 0.3:  # semantic threshold
                src_i = source_ids[i]
                src_j = source_ids[idxs[i][j]]
                # only count if independent publisher
                if sources_meta[src_i].publisher != sources_meta[src_j].publisher:
                    score = max(score, float(sim_val))
        corroboration.append(round(min(score, 1.0), 3))
    return corroboration
'''
# ---------- Step 5: Corroboration Engine (pure-Python, Apple-safe) ----------
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def update_corroboration_scores(claim_texts: list[str], source_ids: list[str], sources_meta: dict) -> list[float]:
    """
    Compute semantic similarity among all claims using cosine similarity.
    Boost credibility if similar claims exist from independent sources.
    """
    # 1. embed all claim texts
    emb = embedder.encode(claim_texts, normalize_embeddings=True)

    # 2. pairwise cosine similarity (N×N)
    sims = cosine_similarity(emb)

    # 3. for each claim, find best independent match
    corroboration = []
    for i in range(len(claim_texts)):
        best = 0.0
        for j in range(len(claim_texts)):
            if i == j:
                continue
            if sources_meta[source_ids[i]].publisher != sources_meta[source_ids[j]].publisher:
                best = max(best, sims[i][j])
        corroboration.append(round(float(best), 3))
    return corroboration

# ---------- Step 6: Action Agent + Report Generator ----------
from openai import OpenAI
client = OpenAI()

def generate_text_report(results):
    """
    Ask GPT-4o-mini to turn structured results into a formatted credibility report.
    """
    prompt = f"""
    You are a credibility analysis summarizer.
    Produce a professional text report from the following data.
    For each claim include:
      • Source name and type  
      • Claim text  
      • Credibility score and band  
      • One-line reason or recommended action:
        - HIGH → Keep
        - MEDIUM → Verify
        - LOW → Flag / Potential bias  

    Data:
    {json.dumps(results, indent=2)}

    Output only clean plain-text in a readable bullet or numbered list.
    """

    resp = client.responses.create(model="gpt-4o-mini", input=prompt, temperature=0.3)
    return resp.output_text.strip()


def action_agent_and_report(sources, docs):
    sources_meta = {s.id: s for s in sources}

    all_claim_texts, all_source_ids, all_claim_objs = [], [], []
    for s in sources:
        claims = extract_claims_llm(docs[s.id])
        for c in claims:
            all_claim_texts.append(c["claim"])
            all_source_ids.append(s.id)
            all_claim_objs.append((s, c))

    corroboration_scores = update_corroboration_scores(all_claim_texts, all_source_ids, sources_meta)

    # assemble structured results
    results = []
    for (src, c), corr in zip(all_claim_objs, corroboration_scores):
        features = compute_features(c, src.type, src.date)
        features["corroboration"] = corr
        res = score_claim(features)
        results.append({
            "publisher": src.publisher,
            "type": src.type,
            "claim": c["claim"],
            "score": res["score"],
            "band": res["band"],
            "features": features
        })

    # generate final formatted report
    report_text = generate_text_report(results)
    os.makedirs("../outputs", exist_ok=True)
    with open("../outputs/report.txt", "w") as f:
        f.write(report_text)
    print("\n✅ Credibility report saved to outputs/report.txt\n")
    print(report_text)
# ---------- Step 7: Interactive Dialogue Mode ----------
def interactive_mode():
    """
    Interactive CLI that uses LLM inference to guess source type and publisher.
    """
    import json, os, hashlib
    from datetime import datetime

    print("\n🧠 Welcome to the Credibility Analyzer")
    print("Type any statement you’d like to check for credibility.\n")

    claim_text = input("👉 Enter your claim or statement: ").strip()
    if not claim_text:
        print("⚠️ No text entered. Exiting.")
        return

    # 1️⃣ Ask user for natural description of where it came from
    context_str = input("\nWhere did you see or hear this? (e.g. 'on YouTube by JoeTechReview' or 'in a NYTimes article'): ").strip()
    metadata = infer_source_metadata(context_str)

    src_type = metadata.get("source_type", "unknown").lower()
    publisher = metadata.get("publisher", "unknown")
    print(f"🧩 Detected source → type: {src_type}, publisher: {publisher}")

    # 2️⃣ Auto-generate IDs and save claim text
    today = datetime.now().strftime("%Y-%m-%d")
    short_hash = hashlib.sha1(claim_text.encode()).hexdigest()[:6]
    source_id = f"S{today.replace('-', '')}_{short_hash}"
    filename = f"../data/user_claim_{short_hash}.txt"
    os.makedirs("../data", exist_ok=True)
    with open(filename, "w") as f:
        f.write(claim_text)

    # 3️⃣ Build Source object
    new_source = Source(
        id=source_id,
        type=src_type,
        publisher=publisher,
        date="unknown",
        path=filename
    )

    # 4️⃣ Merge & analyze
    GLOBAL_SOURCES.append(new_source)
    GLOBAL_DOCS[new_source.id] = claim_text

    try:
        existing = load_user_added_sources()
        merged = existing + GLOBAL_SOURCES
        with open(USER_DATA_FILE, "w") as f:
            json.dump([s.__dict__ for s in merged], f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save to {USER_DATA_FILE}: {e}")

    sources = read_sources() + load_user_added_sources()
    docs = load_documents(read_sources())
    docs.update(GLOBAL_DOCS)

    print(f"\n🔍 Analyzing credibility for {publisher} ({src_type}) ...\n")
    action_agent_and_report(sources, docs)
    print("\n✅ Credibility report saved to outputs/report.txt\n")



if __name__ == "__main__":
    print("Select mode:")
    print("1. Analyze preloaded dataset")
    print("2. Enter a new claim interactively")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        sources = read_sources()
        docs = load_documents(sources)
        action_agent_and_report(sources, docs)
    elif choice == "2":
        interactive_mode()
    else:
        print("Invalid choice. Exiting.")



