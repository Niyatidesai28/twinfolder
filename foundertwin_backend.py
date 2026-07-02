# foundertwin_backend.py

import os
import re
import json
import requests
import numpy as np
import pandas as pd

# Force Hugging Face/Transformers to use the existing local cache only.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# CONFIG
# =========================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_TIMEOUT = 180

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

STARTUP_DATA_CANDIDATES = [
    "startup_only_companies_with_startup_clusters.csv",
    "startup_only_companies_with_tags.csv",
    "startup_only_companies.csv",
]

STARTUP_EMBEDDINGS_FILE = "startup_embeddings.npy"


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text_v2(text):
    """
    Stronger text cleaner used in the merged/master phase.
    Keeps it robust for user input and fallback recompute paths.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # remove employee counts and age-like metadata
    text = re.sub(r"\b\d+\s*(employees|employee|people|person)\b", " ", text)
    text = re.sub(r"\b\d+\+?\s*(years|yrs|yr)\b", " ", text)

    # remove noisy business-type words
    text = re.sub(r"\b(public|private|startup|company|companies)\b", " ", text)

    # remove +N more tails
    text = re.sub(r"\+\d+\s*more", " ", text)

    # replace separators with spaces
    text = text.replace("|", " ")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")

    # keep alphanumeric and basic separators
    text = re.sub(r"[^a-z0-9\s\-\&]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# TAGGING
# =========================================================

def assign_tags_v2(text):
    """
    Rule-based tags aligned with your project's tag layer.
    """
    if pd.isna(text):
        return []

    text = str(text).lower()

    tag_rules = {
        "AI": [
            "ai", "artificial intelligence", "llm", "gpt", "agent",
            "copilot", "machine learning", "ml", "generative", "rag"
        ],
        "SaaS": [
            "saas", "software as a service", "subscription software", "b2b software"
        ],
        "Fintech": [
            "fintech", "payments", "banking", "finance", "financial",
            "lending", "credit", "insurance", "wealth", "tax"
        ],
        "EdTech": [
            "education", "edtech", "learning", "student", "course", "school", "tutoring"
        ],
        "HealthTech": [
            "health", "healthcare", "medical", "clinical", "hospital",
            "radiology", "ehr", "patient", "diagnostic"
        ],
        "Marketplace": [
            "marketplace", "matching", "buyer", "seller", "two-sided", "network"
        ],
        "DevTools": [
            "developer", "devtools", "api", "sdk", "infra", "infrastructure",
            "observability", "testing", "deployment", "code"
        ],
        "Logistics": [
            "logistics", "supply chain", "delivery", "shipment", "warehouse", "routing"
        ],
        "HRTech": [
            "recruitment", "hiring", "resume", "candidate", "job matching",
            "talent", "hr", "interview"
        ],
        "OpsTech": [
            "operations", "workflow", "automation", "back office", "compliance", "ops"
        ],
        "B2B": [
            "b2b", "enterprise", "business", "team", "internal tool", "workflow"
        ],
        "B2C": [
            "b2c", "consumer", "creator", "social", "shopping", "personal"
        ],
    }

    tags = []
    for tag, keywords in tag_rules.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)

    return sorted(set(tags))


def ensure_tags_column(df):
    if "tags" not in df.columns:
        base_text = (
            df.get("combined", pd.Series([""] * len(df))).fillna("").astype(str) + " " +
            df.get("description", pd.Series([""] * len(df))).fillna("").astype(str) + " " +
            df.get("category", pd.Series([""] * len(df))).fillna("").astype(str)
        )
        df["tags"] = base_text.apply(assign_tags_v2)

    # normalize tag strings if already saved as strings
    df["tags"] = df["tags"].apply(_normalize_tags_value)
    return df


def _normalize_tags_value(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    value = str(value).strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    # fallback: comma-split
    return [x.strip() for x in value.split(",") if x.strip()]


# =========================================================
# SCORING
# =========================================================

def label_score(score):
    """
    Per your earlier scoring thresholds.
    """
    try:
        score = float(score)
    except Exception:
        return "poor"

    if score > 0.70:
        return "strong"
    elif score >= 0.50:
        return "usable"
    elif score >= 0.35:
        return "weak"
    else:
        return "poor"


def tag_overlap_score(query_tags, result_tags):
    query_tags = set(query_tags or [])
    result_tags = set(result_tags or [])
    if not query_tags or not result_tags:
        return 0.0
    return len(query_tags.intersection(result_tags)) / max(len(query_tags), 1)


def compute_confidence_label(best_score, avg_top_score):
    try:
        best_score = float(best_score)
    except Exception:
        best_score = 0.0

    try:
        avg_top_score = float(avg_top_score)
    except Exception:
        avg_top_score = 0.0

    if best_score >= 0.72 and avg_top_score >= 0.60:
        return "high"
    elif best_score >= 0.55 and avg_top_score >= 0.45:
        return "medium"
    return "low"


def market_gap_signal(best_score, avg_top_score, cluster_size):
    """
    Simple readable market signal layer.
    """
    try:
        best_score = float(best_score)
    except Exception:
        best_score = 0.0

    try:
        avg_top_score = float(avg_top_score)
    except Exception:
        avg_top_score = 0.0

    try:
        cluster_size = int(cluster_size)
    except Exception:
        cluster_size = 0

    if best_score >= 0.72 and avg_top_score >= 0.60 and cluster_size >= 60:
        return "crowded"
    if best_score >= 0.55 and avg_top_score >= 0.45:
        return "viable"
    if best_score >= 0.40 and cluster_size < 40:
        return "niche"
    return "unclear"


# =========================================================
# LOADING DATA
# =========================================================

def _find_existing_file(candidates):
    for filename in candidates:
        full_path = os.path.join(ROOT_DIR, filename)
        if os.path.exists(full_path):
            return full_path
    return None


def _load_startup_dataframe():
    csv_path = _find_existing_file(STARTUP_DATA_CANDIDATES)
    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find any startup CSV in {STARTUP_DATA_CANDIDATES}. "
            "Make sure one of these files exists in the same folder as app.py/backend."
        )

    df = pd.read_csv(csv_path)

    # normalize common columns
    rename_map = {}
    if "Company_name" in df.columns and "company_name" not in df.columns:
        rename_map["Company_name"] = "company_name"
    if "Description" in df.columns and "description" not in df.columns:
        rename_map["Description"] = "description"

    if rename_map:
        df = df.rename(columns=rename_map)

    # ensure expected columns exist
    if "company_name" not in df.columns:
        df["company_name"] = ""
    if "description" not in df.columns:
        df["description"] = ""
    if "category" not in df.columns:
        df["category"] = ""
    if "combined" not in df.columns:
        df["cleaned_description"] = df["description"].fillna("").astype(str).apply(clean_text_v2)
        df["combined"] = (
            df["company_name"].fillna("").astype(str) + " " +
            df["cleaned_description"].fillna("").astype(str) + " " +
            df["category"].fillna("").astype(str)
        ).str.strip()

    if "startup_cluster" not in df.columns:
        df["startup_cluster"] = -1
    if "startup_cluster_name" not in df.columns:
        df["startup_cluster_name"] = "Unknown"
    if "startup_cluster_quality" not in df.columns:
        df["startup_cluster_quality"] = "Unknown"

    df = ensure_tags_column(df)
    df = df.reset_index(drop=True)
    return df, csv_path


def _load_or_recompute_embeddings(startup_df):
    emb_path = os.path.join(ROOT_DIR, STARTUP_EMBEDDINGS_FILE)
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception as e:
        raise RuntimeError(
            "Embedding model could not be loaded from the local Hugging Face cache. "
            f"Expected cached model: {EMBEDDING_MODEL_NAME}. "
            "Connect once to cache the model, then rerun the app offline."
        ) from e

    if os.path.exists(emb_path):
        startup_embeddings = np.load(emb_path)
        if len(startup_embeddings) == len(startup_df):
            return startup_embeddings, model

    # fallback if mismatch or file missing
    startup_embeddings = model.encode(
        startup_df["combined"].fillna("").tolist(),
        show_progress_bar=True
    )
    return startup_embeddings, model


startup_df, STARTUP_DATA_PATH = _load_startup_dataframe()
startup_embeddings, model = _load_or_recompute_embeddings(startup_df)


# =========================================================
# CORE ANALYZER
# =========================================================

def analyze_startup_idea_v3(user_input, top_n=5):
    """
    Main structured analysis layer.
    Returns:
      - query
      - query_tags
      - startup_cluster
      - startup_cluster_name
      - startup_cluster_quality
      - cluster_size
      - best_score
      - avg_top_score
      - market_signal
      - summary
      - detailed_results
    """
    cleaned_input = clean_text_v2(user_input)
    query_tags = assign_tags_v2(cleaned_input)

    query_emb = model.encode([cleaned_input])
    sims = cosine_similarity(query_emb, startup_embeddings)[0]

    top_n = max(1, int(top_n))
    top_indices = np.argsort(sims)[::-1][:top_n]

    results = startup_df.iloc[top_indices].copy()
    results["similarity_score"] = [float(sims[i]) for i in top_indices]

    if "tags" in results.columns:
        results["tag_overlap_score"] = results["tags"].apply(lambda x: tag_overlap_score(query_tags, x))
    else:
        results["tag_overlap_score"] = 0.0

    # sort primarily by similarity; use tag overlap as secondary tie-breaker
    results = results.sort_values(
        by=["similarity_score", "tag_overlap_score"],
        ascending=False
    ).reset_index(drop=True)

    best_row = results.iloc[0]
    startup_cluster = best_row.get("startup_cluster", -1)
    startup_cluster_name = best_row.get("startup_cluster_name", "Unknown")
    startup_cluster_quality = best_row.get("startup_cluster_quality", "Unknown")

    if "startup_cluster" in startup_df.columns:
        try:
            cluster_size = int((startup_df["startup_cluster"] == startup_cluster).sum())
        except Exception:
            cluster_size = 0
    else:
        cluster_size = 0

    best_score = float(results["similarity_score"].max()) if len(results) else 0.0
    avg_top_score = float(results["similarity_score"].mean()) if len(results) else 0.0
    market_signal = market_gap_signal(best_score, avg_top_score, cluster_size)

    matched_names = results["company_name"].head(min(3, len(results))).tolist()
    matched_names_str = ", ".join([x for x in matched_names if str(x).strip()])

    summary = (
        f"This idea most closely maps to '{startup_cluster_name}' with {label_score(best_score)} "
        f"retrieval strength. Top matches include {matched_names_str if matched_names_str else 'no clear matches'}. "
        f"Overall market signal looks {market_signal}."
    )

    keep_cols = [
        c for c in [
            "company_name",
            "description",
            "category",
            "source",
            "tags",
            "startup_cluster",
            "startup_cluster_name",
            "startup_cluster_quality",
            "similarity_score",
            "tag_overlap_score",
        ]
        if c in results.columns
    ]

    detailed_results = results[keep_cols].head(top_n).to_dict(orient="records")

    return {
        "query": user_input,
        "query_tags": query_tags,
        "startup_cluster": startup_cluster,
        "startup_cluster_name": startup_cluster_name,
        "startup_cluster_quality": startup_cluster_quality,
        "cluster_size": cluster_size,
        "best_score": round(best_score, 4),
        "avg_top_score": round(avg_top_score, 4),
        "market_signal": market_signal,
        "summary": summary,
        "detailed_results": detailed_results,
    }


# =========================================================
# PROMPT FORMATTING
# =========================================================

def normalize_detailed_results(detailed_results, max_rows=5):
    if detailed_results is None:
        return []

    if isinstance(detailed_results, pd.DataFrame):
        return detailed_results.head(max_rows).to_dict(orient="records")

    if isinstance(detailed_results, list):
        cleaned = []
        for row in detailed_results[:max_rows]:
            if isinstance(row, dict):
                cleaned.append(row)
            else:
                cleaned.append({"raw_result": str(row)})
        return cleaned

    return [{"raw_result": str(detailed_results)}]


def format_analysis_for_prompt(analysis):
    """
    Compact structured block for LLM prompts.
    """
    if not isinstance(analysis, dict):
        return "No structured analysis available."

    detailed_results = normalize_detailed_results(
        analysis.get("detailed_results", []),
        max_rows=5
    )

    confidence = compute_confidence_label(
        analysis.get("best_score", 0),
        analysis.get("avg_top_score", 0)
    )

    lines = []
    lines.append(f"Query: {analysis.get('query', '')}")
    lines.append(f"Query Tags: {analysis.get('query_tags', [])}")
    lines.append(f"Startup Cluster: {analysis.get('startup_cluster', '')}")
    lines.append(f"Startup Cluster Name: {analysis.get('startup_cluster_name', '')}")
    lines.append(f"Startup Cluster Quality: {analysis.get('startup_cluster_quality', '')}")
    lines.append(f"Cluster Size: {analysis.get('cluster_size', '')}")
    lines.append(f"Best Score: {analysis.get('best_score', '')}")
    lines.append(f"Average Top Score: {analysis.get('avg_top_score', '')}")
    lines.append(f"Market Signal: {analysis.get('market_signal', '')}")
    lines.append(f"Summary: {analysis.get('summary', '')}")
    lines.append(f"Retrieval Confidence: {confidence}")

    lines.append("\nTop Matches:")
    if not detailed_results:
        lines.append("No detailed matches available.")
    else:
        for i, row in enumerate(detailed_results, start=1):
            company_name = row.get("company_name", row.get("Company_name", ""))
            description = row.get("description", row.get("Description", ""))
            similarity_score = row.get("similarity_score", "")
            tags = row.get("tags", "")
            startup_cluster_name = row.get("startup_cluster_name", "")
            raw_result = row.get("raw_result", "")

            if raw_result:
                lines.append(f"{i}. {raw_result}")
            else:
                lines.append(
                    f"{i}. {company_name} | "
                    f"score={similarity_score} | "
                    f"cluster={startup_cluster_name} | "
                    f"tags={tags} | "
                    f"desc={description}"
                )

    return "\n".join(lines)


# =========================================================
# OLLAMA
# =========================================================

def check_ollama_server(base_url=OLLAMA_BASE_URL):
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "ok": True,
            "models": [m.get("name") for m in data.get("models", [])]
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "models": []
        }


def chat_with_ollama(system_prompt, user_prompt, model=OLLAMA_MODEL, temperature=0.1):
    """
    Non-streaming Ollama call.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
    except requests.exceptions.RequestException as e:
        return f"OLLAMA_REQUEST_ERROR: {str(e)}"
    except Exception as e:
        return f"OLLAMA_UNKNOWN_ERROR: {str(e)}"


# =========================================================
# GROUNDED LLM FUNCTIONS
# =========================================================

def explain_idea_llm_grounded(user_input, top_n=5):
    analysis = analyze_startup_idea_v3(user_input, top_n=top_n)
    analysis_text = format_analysis_for_prompt(analysis)

    system_prompt = """
You are a strict startup analyst inside FounderTwin.

Rules:
- Only use the evidence in the structured analysis.
- Do not invent industries, customers, competitors, or claims.
- If evidence is weak, say that the conclusion is low-confidence.
- Prefer cautious answers over creative ones.
""".strip()

    user_prompt = f"""
STRUCTURED ANALYSIS:
{analysis_text}

Return the answer in this exact format:

Market category:
Similar startups:
Market signal:
Why these matches make sense:
Differentiation suggestion:
Execution risk:
Confidence:

Startup idea:
{user_input}
""".strip()

    llm_response = chat_with_ollama(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=OLLAMA_MODEL,
        temperature=0.1
    )

    return {
        "query": user_input,
        "structured_analysis": analysis,
        "llm_explanation": llm_response
    }


def build_project_steps_llm_grounded(user_input, top_n=5):
    analysis = analyze_startup_idea_v3(user_input, top_n=top_n)
    analysis_text = format_analysis_for_prompt(analysis)

    system_prompt = """
You are a strict startup product and technical planning assistant inside FounderTwin.

Rules:
- Use only the evidence in the structured analysis.
- Do not invent industries, customers, competitors, pricing, or claims.
- If something is unclear, explicitly say it is uncertain.
- Keep the answer practical, MVP-focused, and grounded.
""".strip()

    user_prompt = f"""
STRUCTURED ANALYSIS:
{analysis_text}

Return the answer in this exact format:

Product idea:
Likely target users:
Core problem:
MVP features:
Suggested tech stack:
Step-by-step build roadmap:
What to build first in week 1:
Biggest product risk:
Biggest go-to-market risk:
One realistic differentiation angle:
Confidence:

Startup idea:
{user_input}
""".strip()

    llm_response = chat_with_ollama(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=OLLAMA_MODEL,
        temperature=0.1
    )

    return {
        "query": user_input,
        "structured_analysis": analysis,
        "project_steps": llm_response
    }


def ask_foundertwin_llm_grounded(user_input, question, top_n=5):
    analysis = analyze_startup_idea_v3(user_input, top_n=top_n)
    analysis_text = format_analysis_for_prompt(analysis)

    system_prompt = """
You are FounderTwin's strict startup reasoning assistant.

Rules:
- Answer only from the structured analysis.
- Do not invent facts, market claims, or competitors.
- If evidence is weak, say so clearly.
- Be direct, useful, and grounded.
""".strip()

    user_prompt = f"""
STRUCTURED ANALYSIS:
{analysis_text}

STARTUP IDEA:
{user_input}

FOLLOW-UP QUESTION:
{question}

Return the answer in this exact format:

Answer:
Reasoning based on retrieved matches:
Confidence:
""".strip()

    llm_response = chat_with_ollama(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=OLLAMA_MODEL,
        temperature=0.1
    )

    return {
        "query": user_input,
        "question": question,
        "structured_analysis": analysis,
        "llm_answer": llm_response
    }
