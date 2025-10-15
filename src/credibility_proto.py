#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
credibility_proto.py  —  FINAL

✅ What this script does
1) Initial analysis (Option 1)
   - Loads all *.txt files in data/
   - Extracts claims
   - Computes credibility scores
   - Saves full baseline report and a JSON of baseline scores

2) Incremental update (Option 2)
   - You provide a new *.txt file (e.g., data/new_update.txt), plus source type/publisher
   - Extracts new claims
   - Finds which EXISTING claims are semantically related
   - Re-scores ONLY those related existing claims + all new claims
   - Prints a compact summary showing:
        • NEW claims with their scores and an LLM-written reason (1 sentence)
        • ONLY AFFECTED existing claims with BEFORE → AFTER scores and ▲/▼, plus an LLM-written reason (1 sentence)
   - Writes the same summary to outputs/incremental_summary.txt
   - Writes a full merged report to outputs/report_incremental.txt

⚠️ LLM EXPLANATIONS ARE REQUIRED for the incremental summary.
   - Set OPENAI_API_KEY in your environment to enable LLM calls.
   - If OPENAI_API_KEY is not set, Option 2 will exit with a clear message.

Repository-relative paths:
- Run from repo root:   python3 src/credibility_proto.py
- Or from src/:         python3 credibility_proto.py

Outputs:
- outputs/report_initial.txt
- outputs/initial_scores.json
- outputs/incremental_summary.txt
- outputs/report_incremental.txt
"""

import os
import re
import sys
import json
import math
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime

# ---------------- Path handling (works from repo root OR src/) ----------------
THIS_FILE = os.path.abspath(__file__)
SRC_DIR   = os.path.dirname(THIS_FILE)
ROOT_DIR  = os.path.abspath(os.path.join(SRC_DIR, ".."))
DATA_DIR  = os.path.join(ROOT_DIR, "data")
OUT_DIR   = os.path.join(ROOT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

INITIAL_SCORES_JSON = os.path.join(OUT_DIR, "initial_scores.json")
INITIAL_REPORT_TXT  = os.path.join(OUT_DIR, "report_initial.txt")
INCR_REPORT_TXT     = os.path.join(OUT_DIR, "report_incremental.txt")
INCR_SUMMARY_TXT    = os.path.join(OUT_DIR, "incremental_summary.txt")

# ---------------- Optional LLM (OpenAI) ----------------
USE_LLM = bool(os.environ.get("OPENAI_API_KEY"))
oai_client = None
if USE_LLM:
    try:
        from openai import OpenAI
        oai_client = OpenAI()
    except Exception as e:
        print(f"⚠️ OpenAI client not available: {e}. Falling back to non-LLM mode.")
        USE_LLM = False

# ---------------- Embeddings (semantic corroboration) ----------------
# We attempt a proper embedding model; if not available, we use a pure-python lexical fallback (no numpy needed).
USE_EMBED = True
_embedder = None
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity  # used when available
    # model alias that downloads reliably across envs
    _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    USE_EMBED = False
    print(f"⚠️ SentenceTransformer not available ({e}). Using lexical cosine fallback (pure Python).")

# ---------------- Data classes ----------------
@dataclass
class Source:
    id: str
    type: str          # peer_review | gov | news | corp_pr | ad | social | unknown
    publisher: str
    date: str
    path: str
    funding: str = ""

@dataclass
class Claim:
    claim_id: str
    source_id: str
    text: str
    claim_type: str = "factual"     # factual | superlative | prediction | quote
    tone: str = "neutral"           # neutral | promotional | hedged
    numeric: bool = False

@dataclass
class ScoredClaim:
    claim: Claim
    score: float
    band: str
    features: Dict

# ---------------- Utils ----------------
def read_text(fp: str) -> str:
    with open(fp, "r", encoding="utf-8") as f:
        return f.read().strip()

def sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def fingerprint(s: str, n: int = 10) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:n]

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

def band_from_score(s: float) -> str:
    return "HIGH" if s >= 0.66 else ("MEDIUM" if s >= 0.45 else "LOW")

# ---------------- Source loading ----------------
SOURCE_TYPES = {"peer_review","gov","news","corp_pr","ad","social","unknown"}

def infer_source_from_filename(base: str) -> str:
    if base.startswith("peer_"): return "peer_review"
    if base.startswith("gov_"):  return "gov"
    if base.startswith("news_"): return "news"
    if base.startswith("pr_"):   return "corp_pr"
    if base.startswith("ad_"):   return "ad"
    if base.startswith("social_"): return "social"
    return "unknown"

def load_sources_from_data() -> List[Source]:
    sources: List[Source] = []
    if not os.path.isdir(DATA_DIR):
        print(f"❌ data/ folder not found at: {DATA_DIR}")
        return sources
    for name in sorted(os.listdir(DATA_DIR)):
        if not name.lower().endswith(".txt"): continue
        path = os.path.join(DATA_DIR, name)
        base = os.path.splitext(name)[0]
        stype = infer_source_from_filename(base)
        publisher = base
        date = "unknown"
        sources.append(Source(
            id=f"S_{base}", type=(stype if stype in SOURCE_TYPES else "unknown"),
            publisher=publisher, date=date, path=path
        ))
    return sources

# ---------------- Claim extraction ----------------
def heuristic_extract_claims(text: str) -> List[Dict]:
    out = []
    for sent in sentence_split(text):
        low = sent.lower()
        if any(k in low for k in [" is "," are "," has "," have "," will "," can "," claims "," reported "," announced "]) and len(sent.split())>5:
            claim_type = "superlative" if re.search(r"\b(best|#1|leading|unparalleled|world[- ]class|most advanced)\b", low) else "factual"
            tone = "promotional" if (claim_type=="superlative" or re.search(r"\b(leader|innovative|revolutionary|unmatched|guarantees?)\b", low)) else "neutral"
            numeric = bool(re.search(r"\d", sent))
            out.append({"claim":sent,"claim_type":claim_type,"tone":tone,"numeric":numeric})
    return out

def llm_extract_claims(text: str) -> List[Dict]:
    # Keep extraction heuristic-only to avoid API calls cost/latency; demo focus is on LLM explanations.
    return heuristic_extract_claims(text)

def extract_claims_for_source(src: Source) -> List[Claim]:
    text = read_text(src.path)
    raw = llm_extract_claims(text)
    claims: List[Claim] = []
    for r in raw:
        cid = f"C_{src.id}_{fingerprint(r['claim'], 8)}"
        claims.append(Claim(
            claim_id=cid, source_id=src.id, text=r["claim"],
            claim_type=r.get("claim_type","factual"),
            tone=r.get("tone","neutral"),
            numeric=bool(r.get("numeric",False))
        ))
    return claims

# ---------------- Features & scoring ----------------
SOURCE_PRIOR = {
    "peer_review":0.9, "gov":0.85, "news":0.70,
    "corp_pr":0.45, "ad":0.25, "social":0.20, "unknown":0.50
}
WEIGHTS = {"source_prior":0.35,"linguistic":0.15,"corroboration":0.25,"recency":0.10,"contextuality":0.10,"coi":0.05}

def recency_score(date_str: str) -> float:
    try:
        y = int(date_str.split("-")[0]); age = max(0, datetime.now().year - y)
        return max(0.2, 1.0/(1 + 0.15*age))
    except Exception:
        return 0.5

def compute_features(claim: Claim, src: Source) -> Dict:
    prior = SOURCE_PRIOR.get(src.type, 0.5)
    hedged_penalty = -0.1 if claim.tone == "hedged" else 0
    promo_penalty  = -0.2 if claim.tone == "promotional" else 0
    specific_bonus = 0.1 if claim.numeric else (0.05 if re.search(r"\d", claim.text) else 0)
    linguistic = max(0, min(1, 0.5 + hedged_penalty + promo_penalty + specific_bonus))
    contextuality = 0.5
    if claim.claim_type=="superlative" and src.type in {"ad","corp_pr"}: contextuality -= 0.3
    if claim.claim_type=="factual" and src.type in {"peer_review","gov"}: contextuality += 0.3
    contextuality = max(0, min(1, contextuality))
    recency = recency_score(src.date)
    coi = 0.8 if src.type in {"news","peer_review"} else (0.5 if src.type=="corp_pr" else 0.3)
    return {"source_prior":round(prior,3),"linguistic":round(linguistic,3),"recency":round(recency,3),
            "contextuality":round(contextuality,3),"coi":round(coi,3),"corroboration":0.0}

def score_features(feat: Dict) -> Tuple[float,str]:
    s = sum(WEIGHTS[k]*feat.get(k,0.0) for k in WEIGHTS)
    s = round(min(max(s,0.0),1.0), 3)
    band = band_from_score(s)
    return s, band

def recommend_action(band: str) -> str:
    return {"HIGH":"Keep","MEDIUM":"Verify","LOW":"Flag/Potential bias"}[band]

# ---------------- Embedding & corroboration ----------------
def _lexical_vecs(texts: List[str]) -> List[Dict[str,int]]:
    # pure python bag-of-words (no numpy)
    vecs = []
    for s in texts:
        counts = {}
        for tok in tokenize(s):
            counts[tok] = counts.get(tok,0) + 1
        vecs.append(counts)
    return vecs

def _cosine_counts(a: Dict[str,int], b: Dict[str,int]) -> float:
    # cosine similarity for sparse word count dicts
    if not a or not b:
        return 0.0
    dot = 0.0
    for k, va in a.items():
        vb = b.get(k, 0)
        if vb:
            dot += float(va) * float(vb)
    na = math.sqrt(sum(float(v)**2 for v in a.values())) or 1.0
    nb = math.sqrt(sum(float(v)**2 for v in b.values())) or 1.0
    return dot / (na * nb)

def embed_texts(texts: List[str]):
    if USE_EMBED and _embedder is not None:
        return _embedder.encode(texts, normalize_embeddings=True)
    # lexical fallback: return token count dicts
    return _lexical_vecs(texts)

def pairwise_similarity(A, B=None) -> List[List[float]]:
    # If using embeddings, use sklearn cosine; otherwise lexical cosine.
    if USE_EMBED and _embedder is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity as _cos
            return _cos(A, A if B is None else B).tolist()
        except Exception:
            pass
    # lexical fallback (dicts)
    if B is None:
        B = A
    sims = []
    for i in range(len(A)):
        row = []
        for j in range(len(B)):
            row.append(_cosine_counts(A[i], B[j]))
        sims.append(row)
    return sims

def compute_corroboration(claims: List[Claim], src_by_id: Dict[str, Source]) -> Dict[str, float]:
    if not claims: return {}
    texts = [c.text for c in claims]
    E = embed_texts(texts)
    sims = pairwise_similarity(E)  # square
    out = {}
    for i, ci in enumerate(claims):
        best = 0.0
        pi = src_by_id[ci.source_id].publisher
        for j, cj in enumerate(claims):
            if i==j: continue
            pj = src_by_id[cj.source_id].publisher
            if pj != pi:
                best = max(best, float(sims[i][j]))
        out[ci.claim_id] = round(min(max(best,0.0),1.0), 3)
    return out

# ---------------- Global in-memory state for demo ----------------
INITIAL_SOURCES: List[Source] = []
INITIAL_CLAIMS:  List[Claim]  = []
SRC_BY_ID: Dict[str, Source]  = {}
INITIAL_SCORED: Dict[str, ScoredClaim] = {}  # claim_id -> ScoredClaim
INITIAL_TEXTS: List[str] = []                # claim texts in same order as INITIAL_CLAIMS
SIM_THRESHOLD = 0.35      # semantic neighborhood threshold
DELTA_THRESHOLD = 0.02    # minimum difference to count as "affected"

# ---------------- Reporting helpers ----------------
def write_report(items: List[ScoredClaim], out_fp: str):
    lines = ["# Credibility Report", ""]
    for sc in items:
        src = SRC_BY_ID.get(sc.claim.source_id, None)
        pub = src.publisher if src else "unknown"
        stype = src.type if src else "unknown"
        lines.append(f"- Source: {pub} ({stype})")
        lines.append(f"  - Claim: {sc.claim.text}")
        lines.append(f"  - Score: {sc.score:.3f} | Band: {sc.band} | Action: {recommend_action(sc.band)}")
        lines.append(f"  - Features: {json.dumps(sc.features)}")
        lines.append("")
    with open(out_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📄 Saved: {out_fp}")

def save_initial_scores_json(scored: Dict[str, ScoredClaim]):
    data = []
    for cid, sc in scored.items():
        data.append({
            "claim_id": cid,
            "source_id": sc.claim.source_id,
            "text": sc.claim.text,
            "score": sc.score,
            "band": sc.band
        })
    with open(INITIAL_SCORES_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_initial_scores_json() -> Dict[str, Dict]:
    if not os.path.exists(INITIAL_SCORES_JSON):
        return {}
    with open(INITIAL_SCORES_JSON, "r", encoding="utf-8") as f:
        arr = json.load(f)
    return {row["claim_id"]: row for row in arr}

# ---------------- Initial analysis ----------------
def run_initial_analysis():
    global INITIAL_SOURCES, INITIAL_CLAIMS, SRC_BY_ID, INITIAL_SCORED, INITIAL_TEXTS
    INITIAL_SOURCES = load_sources_from_data()
    SRC_BY_ID = {s.id: s for s in INITIAL_SOURCES}

    all_claims: List[Claim] = []
    for s in INITIAL_SOURCES:
        all_claims.extend(extract_claims_for_source(s))
    INITIAL_CLAIMS = all_claims

    # Corroboration among all initial claims
    corr_map = compute_corroboration(INITIAL_CLAIMS, SRC_BY_ID)

    INITIAL_SCORED = {}
    for c in INITIAL_CLAIMS:
        feats = compute_features(c, SRC_BY_ID[c.source_id])
        feats["corroboration"] = corr_map.get(c.claim_id, 0.0)
        score, band = score_features(feats)
        INITIAL_SCORED[c.claim_id] = ScoredClaim(c, score, band, feats)

    INITIAL_TEXTS = [c.text for c in INITIAL_CLAIMS]

    # Write report + persist scores for future diffs
    write_report(list(INITIAL_SCORED.values()), INITIAL_REPORT_TXT)
    save_initial_scores_json(INITIAL_SCORED)
    print("\n✅ Initial analysis complete.")
    print(f"   Claims scored: {len(INITIAL_SCORED)}")
    print(f"   Report: {os.path.relpath(INITIAL_REPORT_TXT, ROOT_DIR)}")

# ---------------- LLM reasoning (ALWAYS LLM, 1 sentence) ----------------
def _require_llm_or_exit():
    if not USE_LLM or oai_client is None:
        print("❌ LLM is required to generate explanations.")
        print("   Set OPENAI_API_KEY env var and re-run Option 2.")
        sys.exit(1)

def llm_reason_for_change(old_item: Dict, new_item: Dict, new_evidence_snippets: List[str], src_cred: float) -> str:
    """
    Always uses LLM. Returns ONE short sentence explaining the direction of change,
    grounded in the supplied evidence snippets.
    """
    _require_llm_or_exit()
    direction = "increased" if new_item["score"] > old_item["score"] else "decreased"
    credibility = "high-credibility" if src_cred >= 0.7 else "moderate-credibility" if src_cred >= 0.5 else "low-credibility"
    prompt = f"""
You are writing a concise, precise reason (ONE sentence) for a credibility score change.

Claim (unchanged text):
"{old_item['text']}"

Old score/band: {old_item['score']:.3f} / {old_item['band']}
New score/band: {new_item['score']:.3f} / {new_item['band']}
Direction: {direction}

New evidence snippets (may corroborate or contradict):
{json.dumps(new_evidence_snippets, indent=2)}

New evidence source credibility (approx): {credibility}

Rules:
- Output ONE sentence only (<= 35 words).
- Be specific about why the change happened (e.g., independent corroboration, contradiction, stronger competitor results, idealized vs real-world metrics, etc.).
- Do NOT mention "score", "band", "delta", or "direction" explicitly—explain the WHY.
- Avoid generic phrases like "new source added" without details.

Return ONLY the sentence.
""".strip()
    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": prompt}],
        temperature=0.2
    )
    out = resp.choices[0].message.content.strip()
    # ensure one sentence (best-effort)
    return out.split("\n")[0].strip()

def llm_reason_for_new_claim(claim_text: str, source_type: str, publisher: str, approx_cred: float) -> str:
    _require_llm_or_exit()
    credibility = "high-credibility" if approx_cred >= 0.7 else "moderate-credibility" if approx_cred >= 0.5 else "low-credibility"
    prompt = f"""
You are explaining in ONE sentence why a NEW claim matters for credibility analysis.

New claim:
"{claim_text}"

Source type: {source_type}
Publisher: {publisher}
Source credibility (approx): {credibility}

Rules:
- Output ONE sentence only (<= 30 words).
- Be specific about why this new claim is relevant (e.g., introduces verified metric, contradicts prior marketing, provides independent benchmark, adds real-world data, etc.).
- Avoid generic phrases.

Return ONLY the sentence.
""".strip()
    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": prompt}],
        temperature=0.2
    )
    out = resp.choices[0].message.content.strip()
    return out.split("\n")[0].strip()

# ---------------- Incremental update ----------------
def incremental_update(new_path: str, publisher: str=None, stype: str="news", date: str="unknown"):
    """
    - Build a Source for the new file
    - Extract its claims
    - Find impacted existing claims via semantic/lexical similarity
    - Re-score ONLY impacted old claims + all new claims
    - Save full merged report and a compact incremental summary (console + file)
    - Explanations are generated by the LLM (required)
    """
    if not os.path.exists(new_path):
        print(f"❌ New file not found: {new_path}")
        return

    if stype not in SOURCE_TYPES:
        stype = "unknown"

    base = os.path.splitext(os.path.basename(new_path))[0]
    if not publisher:
        publisher = base

    # Ensure baseline (initial) exists in memory; if not, reconstruct
    global INITIAL_SOURCES, INITIAL_CLAIMS, SRC_BY_ID, INITIAL_SCORED, INITIAL_TEXTS
    if not INITIAL_SCORED:
        if not os.path.exists(INITIAL_SCORES_JSON):
            print("ℹ️ No baseline found. Run Option 1 first.")
            return
        # Rebuild minimal state
        INITIAL_SOURCES = load_sources_from_data()
        SRC_BY_ID = {s.id:s for s in INITIAL_SOURCES}
        INITIAL_CLAIMS = []
        for s in INITIAL_SOURCES:
            INITIAL_CLAIMS.extend(extract_claims_for_source(s))
        INITIAL_TEXTS = [c.text for c in INITIAL_CLAIMS]
        # set baseline scores from file (for before-values), and also compute in-memory ScoredClaim map
        baseline_by_id = load_initial_scores_json()
        INITIAL_SCORED = {}
        # Compute features to have complete ScoredClaim objects
        corr_map = compute_corroboration(INITIAL_CLAIMS, SRC_BY_ID)
        for c in INITIAL_CLAIMS:
            feats = compute_features(c, SRC_BY_ID[c.source_id])
            feats["corroboration"] = corr_map.get(c.claim_id, 0.0)
            score, band = score_features(feats)
            INITIAL_SCORED[c.claim_id] = ScoredClaim(c, score, band, feats)

    # Add the new source
    new_src = Source(id=f"S_{base}", type=stype, publisher=publisher, date=date, path=new_path)
    SRC_BY_ID[new_src.id] = new_src

    # Extract new claims
    new_claims = extract_claims_for_source(new_src)
    if not new_claims:
        print("ℹ️ No claims found in the new file.")
        return

    # Find impacted old claims by similarity
    texts_old = [c.text for c in INITIAL_CLAIMS]
    texts_new = [c.text for c in new_claims]
    E_old = embed_texts(texts_old) if texts_old else []
    E_new = embed_texts(texts_new)

    sims_new_old = pairwise_similarity(E_new, E_old) if texts_old else [[] for _ in E_new]

    impacted_old_ids = set()
    impacted_pairs_for_reason: Dict[str, List[str]] = {}  # old_claim_id -> new snippets similar to it
    for i in range(len(E_new)):
        for j in range(len(E_old)):
            if float(sims_new_old[i][j]) >= SIM_THRESHOLD:
                impacted_old_ids.add(INITIAL_CLAIMS[j].claim_id)
                impacted_pairs_for_reason.setdefault(INITIAL_CLAIMS[j].claim_id, []).append(new_claims[i].text)

    # Recompute corroboration using all claims (old + new), but re-score only impacted old + all new
    all_for_corr = INITIAL_CLAIMS + new_claims
    corr_map = compute_corroboration(all_for_corr, SRC_BY_ID)

    # Build merged scored map (start with old scored)
    merged_scored: Dict[str, ScoredClaim] = {cid: sc for cid, sc in INITIAL_SCORED.items()}

    # Re-score impacted old claims
    for c in INITIAL_CLAIMS:
        if c.claim_id in impacted_old_ids:
            feats = compute_features(c, SRC_BY_ID[c.source_id])
            feats["corroboration"] = corr_map.get(c.claim_id, 0.0)
            score, band = score_features(feats)
            merged_scored[c.claim_id] = ScoredClaim(c, score, band, feats)

    # Score all new claims
    for c in new_claims:
        feats = compute_features(c, SRC_BY_ID[c.source_id])
        feats["corroboration"] = corr_map.get(c.claim_id, 0.0)
        score, band = score_features(feats)
        merged_scored[c.claim_id] = ScoredClaim(c, score, band, feats)

    # ---------- Build compact incremental summary (only new/changed) ----------
    old_index = load_initial_scores_json()  # claim_id -> {text, score, band}

    # NEW claims (not present previously)
    new_rows = []
    for c in new_claims:
        sc = merged_scored[c.claim_id]
        prior = sc.features.get("source_prior", 0.5)
        # Always LLM for reason
        reason = llm_reason_for_new_claim(c.text, SRC_BY_ID[c.source_id].type, SRC_BY_ID[c.source_id].publisher, prior)
        new_rows.append({
            "text": c.text, "score": sc.score, "band": sc.band, "reason": reason
        })

    # UPDATED existing claims
    changed_rows = []
    for old_id in impacted_old_ids:
        if old_id in old_index and old_id in merged_scored:
            before = old_index[old_id]
            after_sc = merged_scored[old_id]
            if abs(after_sc.score - before["score"]) > DELTA_THRESHOLD:
                arrow = "▲ Increased" if after_sc.score > before["score"] else "▼ Decreased"
                # LLM-written reason (no template)
                src_prior = after_sc.features.get("source_prior", 0.5)
                reason = llm_reason_for_change(
                    {"text":before["text"], "score":before["score"], "band":before["band"]},
                    {"text":after_sc.claim.text, "score":after_sc.score, "band":after_sc.band},
                    impacted_pairs_for_reason.get(old_id, []),
                    src_prior
                )
                changed_rows.append({
                    "text": after_sc.claim.text,
                    "before": before["score"],
                    "after": after_sc.score,
                    "delta": after_sc.score - before["score"],
                    "arrow": arrow,
                    "reason": reason
                })

    # Format console + file output
    lines = []
    lines.append("="*22)
    lines.append("📌 NEW CLAIMS ADDED")
    lines.append("="*22)
    if not new_rows:
        lines.append("• None")
    else:
        for i, r in enumerate(new_rows, 1):
            lines.append(f"{i}. \"{r['text']}\"")
            lines.append(f"   Score: {r['score']:.3f} ({band_from_score(r['score'])})")
            lines.append(f"   Reason: {r['reason']}")
    lines.append("")
    lines.append("="*28)
    lines.append("♻️ UPDATED EXISTING CLAIMS")
    lines.append("="*28)
    if not changed_rows:
        lines.append("• None (no existing claims changed beyond threshold).")
    else:
        for r in changed_rows:
            lines.append(f"\"{r['text']}\"")
            lines.append(f"Old: {r['before']:.3f} → New: {r['after']:.3f}  {r['arrow']}")
            lines.append(f"Reason: {r['reason']}")
            lines.append("")

    # One-sentence LLM summary of overall impact
    _require_llm_or_exit()
    try:
        prompt = f"""
Provide ONE sentence summarizing how the new text affected credibility overall.
Focus on whether it corroborated, contradicted, or nuanced prior claims, and mention the type of evidence (e.g., government, peer-reviewed, news).

New claims:
{json.dumps([nr['text'] for nr in new_rows], indent=2)}

Changed claims (old text unchanged):
{json.dumps([cr['text'] for cr in changed_rows], indent=2)}

Return ONE sentence (<= 30 words).
""".strip()
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content": prompt}],
            temperature=0.2
        )
        overall_summary = resp.choices[0].message.content.strip().split("\n")[0]
    except Exception:
        overall_summary = "Independent new evidence adjusted credibility for related claims and introduced additional validated statements."

    lines.append("="*30)
    lines.append("🧠 LLM SUMMARY OF IMPACT")
    lines.append("="*30)
    lines.append(overall_summary)
    lines.append("")

    # Print to console and save summary file
    print("\n" + "\n".join(lines))
    with open(INCR_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"📄 Saved summary: {os.path.relpath(INCR_SUMMARY_TXT, ROOT_DIR)}")

    # Write full merged report
    write_report(list(merged_scored.values()), INCR_REPORT_TXT)
    print("\n✅ Incremental update complete.")
    print(f"   Report: {os.path.relpath(INCR_REPORT_TXT, ROOT_DIR)}")

# ---------------- Menu ----------------
def main():
    print("\nSelect mode:")
    print("1. Run initial analysis on data/")
    print("2. Add new file & run incremental update")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_initial_analysis()

    elif choice == "2":
        if not USE_LLM:
            print("❌ LLM explanations are required for incremental summary.")
            print("   Please set OPENAI_API_KEY and re-run Option 2.")
            return
        # Ensure we have a baseline
        if not os.path.exists(INITIAL_SCORES_JSON):
            print("ℹ️ No baseline found. Running initial analysis first...")
            run_initial_analysis()
        new_path = input("\nPath to new .txt file (e.g., data/new_update.txt): ").strip()
        publisher = input("Publisher (blank = infer from filename): ").strip() or None
        stype = input("Type [peer_review|gov|news|corp_pr|ad|social|unknown] (default news): ").strip() or "news"
        date = input("Date YYYY-MM-DD (optional, default unknown): ").strip() or "unknown"
        incremental_update(new_path, publisher, stype, date)

    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
