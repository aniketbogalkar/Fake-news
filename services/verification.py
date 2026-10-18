"""
Evidence-Based Verification Engine for NewsCheck AI
Performs factual claim extraction, queries credible news publications,
analyzes entity alignment, debunking signals, and assigns evidence-based verdicts.
Separates linguistic ML predictions from real-world evidence verification.
"""

import os
import re
import joblib
import numpy as np
from typing import Dict, Any, List

from services.news_search import search_credible_news, CREDIBLE_DOMAINS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Stopwords for query keyword extraction
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", 
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", 
    "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", 
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", 
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", 
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", 
    "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", 
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", 
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", 
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", 
    "you're", "you've", "your", "yours", "yourself", "yourselves", "forwarded", "received", "viral",
    "breaking", "shocking", "urgent", "secret", "alert", "must", "share", "watch", "today"
}

# Tokens indicating fact-checking, debunking, or direct contradiction
DEBUNK_TOKENS = {
    "fact check", "fact-check", "hoax", "debunk", "debunks", "debunked", "false", "misleading", 
    "fake", "no evidence", "untrue", "scam", "rumour", "rumor", "refutes", "denies", "fabricated", 
    "bogus", "unfounded", "baseless", "myth", "busted", "clarifies", "hoaxes"
}


class ModelInferenceEngine:
    """Handles linguistic ML classification using trained model and TF-IDF."""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.load_model()

    def load_model(self):
        model_path = os.path.join(MODELS_DIR, "model.pkl")
        vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")

        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                print("[ModelInference] Loaded model and vectorizer successfully.")
            except Exception as e:
                print(f"[ModelInference] Error loading model: {e}")
                self.model = None
                self.vectorizer = None
        else:
            print("[ModelInference] Model artifacts not found. Please run train_model.py.")

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.model or not self.vectorizer:
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "explanation": "Trained machine learning model is not loaded. Run train_model.py to initialize."
            }

        # Clean text
        clean = re.sub(r'https?://\S+|www\.\S+', ' ', text.lower())
        clean = clean.translate(str.maketrans('', '', '!@#$%^&*()[]{};:,.<>?/\\|`~-=_+'))
        clean = re.sub(r'\s+', ' ', clean).strip()

        if len(clean) < 10:
            return {
                "prediction": "Inconclusive",
                "confidence": 50.0,
                "explanation": "Text input too short for linguistic statistical analysis."
            }

        tfidf_features = self.vectorizer.transform([clean])
        pred_label = self.model.predict(tfidf_features)[0]

        # Calculate confidence
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(tfidf_features)[0]
            confidence = round(float(np.max(probs)) * 100, 1)
        elif hasattr(self.model, "decision_function"):
            decision = self.model.decision_function(tfidf_features)[0]
            # Sigmoid scaling
            prob_real = 1.0 / (1.0 + np.exp(-decision))
            confidence = round(float(prob_real if pred_label == 1 else (1.0 - prob_real)) * 100, 1)
        else:
            confidence = 80.0

        label_str = "Likely Real News Patterns" if pred_label == 1 else "Likely Fake / Sensational Patterns"
        explanation = (
            f"The statistical model classifies the linguistic style of this text as '{label_str}' "
            f"with a {confidence}% pattern match confidence against the training corpus. "
            f"Note: This measures writing style and lexical markers, not factual truth in the real world."
        )

        return {
            "prediction": label_str,
            "raw_label": "Real" if pred_label == 1 else "Fake",
            "confidence": confidence,
            "explanation": explanation
        }


# Global singleton instance
ml_engine = ModelInferenceEngine()


def extract_claim_and_keywords(text: str) -> Dict[str, Any]:
    """
    Extracts key claims, entities, dates, and search terms from user input.
    Prevents query dilution from conversational chatter or WhatsApp forwarding instructions.
    """
    # Extract years and dates
    years = re.findall(r'\b(19\d\d|20\d\d)\b', text)
    
    # Extract numbers or monetary figures
    amounts = re.findall(r'(\d+(?:\.\d+)?\s*(?:crore|lakh|million|billion|rupees|rs|usd|\$|%|percent))', text, flags=re.IGNORECASE)

    # Clean punctuation for keyword extraction
    clean = re.sub(r'[^\w\s]', ' ', text)
    words = clean.split()

    filtered_keywords = [
        w for w in words 
        if w.lower() not in STOPWORDS and len(w) > 2
    ]

    # Prioritize capitalized terms (Proper nouns/Named Entities)
    named_entities = [w for w in words if w.istitle() and w.lower() not in STOPWORDS]

    # Combine top named entities and distinctive keywords
    top_terms = []
    seen = set()
    for item in named_entities + filtered_keywords:
        item_lower = item.lower()
        if item_lower not in seen and len(item_lower) > 2:
            seen.add(item_lower)
            top_terms.append(item)
            if len(top_terms) >= 6:
                break

    # If no distinctive terms found, take first few words
    if not top_terms:
        top_terms = words[:5]

    search_query = " ".join(top_terms)
    return {
        "search_query": search_query,
        "keywords": top_terms,
        "detected_years": list(set(years)),
        "detected_amounts": list(set(amounts))
    }


def compute_evidence_verdict(input_text: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compares the user's claim with retrieved live news articles.
    Returns one of the 4 strict categories:
    - Supported by available evidence
    - Contradicted by available evidence
    - Mixed or conflicting evidence
    - Unverified — insufficient evidence
    """
    if not articles:
        return {
            "status": "Unverified — insufficient evidence",
            "badge_class": "badge-unverified",
            "summary": "No verified reports matching this specific claim were found in accessible credible news outlets. Note: Absence of matching articles is not proof of falsity; the claim remains unverified.",
            "supporting_count": 0,
            "contradicting_count": 0,
            "neutral_count": 0
        }

    input_lower = input_text.lower()
    input_tokens = set(re.findall(r'\b[a-z]{3,}\b', input_lower)) - STOPWORDS

    supporting_sources = []
    contradicting_sources = []
    neutral_sources = []

    for art in articles:
        title = art.get("title", "")
        snippet = art.get("snippet", "")
        combined = f"{title} {snippet}".lower()
        source_name = art.get("source", "News Source")

        # Check for debunk/contradiction signals in credible headline or snippet
        has_debunk_token = any(token in combined for token in DEBUNK_TOKENS)

        # Check token overlap
        article_tokens = set(re.findall(r'\b[a-z]{3,}\b', combined)) - STOPWORDS
        overlap = input_tokens.intersection(article_tokens)
        overlap_score = len(overlap) / max(len(input_tokens), 1)

        if has_debunk_token and overlap_score >= 0.2:
            contradicting_sources.append({
                "source": source_name,
                "title": title,
                "reason": "Source explicitly flags or debunks this claim as false, hoax, or misleading."
            })
        elif overlap_score >= 0.35:
            # Significant overlap indicating affirmative reporting
            supporting_sources.append({
                "source": source_name,
                "title": title,
                "overlap": round(overlap_score * 100, 1)
            })
        else:
            neutral_sources.append(source_name)

    # Determine final verdict
    if len(contradicting_sources) > 0 and len(supporting_sources) == 0:
        status = "Contradicted by available evidence"
        badge_class = "badge-contradicted"
        summary = (
            f"Credible news organizations and fact-checkers ({', '.join([c['source'] for c in contradicting_sources[:2]])}) "
            f"directly contradict or debunk this claim."
        )
    elif len(supporting_sources) >= 1 and len(contradicting_sources) == 0:
        status = "Supported by available evidence"
        badge_class = "badge-supported"
        names = ", ".join(list(dict.fromkeys([s['source'] for s in supporting_sources[:3]])))
        summary = (
            f"The core claim is supported by reporting in established news outlets including {names}."
        )
    elif len(supporting_sources) > 0 and len(contradicting_sources) > 0:
        status = "Mixed or conflicting evidence"
        badge_class = "badge-mixed"
        summary = (
            f"Evidence is divided: Some sources report on the subject affirmatively, while others present disputes, clarifications, or counter-claims."
        )
    else:
        status = "Unverified — insufficient evidence"
        badge_class = "badge-unverified"
        summary = (
            "Related articles were discovered, but none provided sufficient direct confirmation or refutation of the exact specific claim. "
            "A newspaper's silence is not evidence that the claim is false."
        )

    return {
        "status": status,
        "badge_class": badge_class,
        "summary": summary,
        "supporting_count": len(supporting_sources),
        "contradicting_count": len(contradicting_sources),
        "neutral_count": len(neutral_sources)
    }


def analyze_news_item(text: str, source_url: str = "") -> Dict[str, Any]:
    """
    Main verification pipeline orchestrating:
    1. Linguistic ML model classification.
    2. Claim and keyword extraction.
    3. Live news search across credible sources.
    4. Evidence comparison and verdict assignment.
    """
    # 1. Linguistic ML
    ml_result = ml_engine.predict(text)

    # 2. Claim Extraction
    claim_info = extract_claim_and_keywords(text)

    # 3. Live News Search
    search_response = search_credible_news(claim_info["search_query"])
    articles = search_response.get("articles", [])
    provider_used = search_response.get("provider", "None")

    # 4. Evidence Engine
    evidence_verdict = compute_evidence_verdict(text, articles)

    return {
        "input_text": text,
        "source_url": source_url,
        "ml_result": ml_result,
        "claim_info": claim_info,
        "provider_used": provider_used,
        "articles": articles,
        "evidence_verdict": evidence_verdict
    }
