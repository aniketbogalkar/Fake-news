"""
News Search Service for NewsCheck AI
Searches live verified news sources (Reuters, BBC, The Hindu, Times of India, etc.)
Supports NewsAPI when an API key is configured, and automatic live search fallback.
Never invents or fabricates citations.
"""

import os
import re
import requests
from urllib.parse import urlparse
from datetime import datetime

# Known credible news domains and credibility tagging
CREDIBLE_DOMAINS = {
    "reuters.com": {"name": "Reuters", "tier": "Global Wire Agency (Highest Credibility)"},
    "bbc.com": {"name": "BBC News", "tier": "Public Broadcaster (Highest Credibility)"},
    "bbc.co.uk": {"name": "BBC News", "tier": "Public Broadcaster (Highest Credibility)"},
    "thehindu.com": {"name": "The Hindu", "tier": "Established National Daily (High Credibility)"},
    "timesofindia.indiatimes.com": {"name": "The Times of India", "tier": "Major National Daily (Credible)"},
    "indianexpress.com": {"name": "The Indian Express", "tier": "Established National Daily (High Credibility)"},
    "ndtv.com": {"name": "NDTV", "tier": "Major News Network (Credible)"},
    "hindustantimes.com": {"name": "Hindustan Times", "tier": "Major National Daily (Credible)"},
    "pib.gov.in": {"name": "Press Information Bureau (PIB)", "tier": "Official Government Source (Authoritative)"},
    "apnews.com": {"name": "Associated Press (AP)", "tier": "Global Wire Agency (Highest Credibility)"},
    "aljazeera.com": {"name": "Al Jazeera", "tier": "International News (Credible)"},
    "theguardian.com": {"name": "The Guardian", "tier": "International Daily (Credible)"},
    "nature.com": {"name": "Nature", "tier": "Peer-Reviewed Scientific Journal (Authoritative)"},
    "who.int": {"name": "World Health Organization", "tier": "Global Health Authority (Authoritative)"},
    "isro.gov.in": {"name": "ISRO Official", "tier": "Official Government Agency (Authoritative)"},
    "nasa.gov": {"name": "NASA Official", "tier": "Official Space Agency (Authoritative)"}
}


def identify_source_credibility(url: str, detected_source: str = ""):
    """Classifies domain into credibility tiers."""
    if not url:
        return {"source": detected_source or "Unknown Source", "tier": "General Source"}
    
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    for cred_domain, info in CREDIBLE_DOMAINS.items():
        if cred_domain in domain:
            return {"source": info["name"], "tier": info["tier"], "is_priority": True}

    # If not in priority map, format cleanly
    clean_name = detected_source or domain
    return {"source": clean_name, "tier": "Independent / Regional Source", "is_priority": False}


def search_newsapi(query: str, api_key: str, max_results: int = 6):
    """Queries official NewsAPI.org v2 everything endpoint."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": max_results,
        "apiKey": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=8)
        if response.status_code == 200:
            data = response.json()
            articles = []
            for item in data.get("articles", []):
                cred = identify_source_credibility(item.get("url", ""), item.get("source", {}).get("name", ""))
                articles.append({
                    "title": item.get("title", ""),
                    "source": cred["source"],
                    "url": item.get("url", ""),
                    "snippet": item.get("description", "") or item.get("content", ""),
                    "date": item.get("publishedAt", "")[:10] if item.get("publishedAt") else "Recent",
                    "credibility_tier": cred["tier"],
                    "is_priority": cred.get("is_priority", False)
                })
            return {"success": True, "articles": articles, "provider": "NewsAPI.org"}
        else:
            return {"success": False, "error": f"NewsAPI returned status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_ddg_news(query: str, max_results: int = 8):
    """Queries live news via DuckDuckGo news search without requiring an API key."""
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        raw_results = list(ddgs.news(query, max_results=max_results))
        
        # If news returns empty, try broad text search for news items
        if not raw_results:
            raw_results = list(ddgs.text(f"{query} news", max_results=max_results))

        articles = []
        for item in raw_results:
            url = item.get("url") or item.get("href", "")
            title = item.get("title", "")
            body = item.get("body", "")
            source_raw = item.get("source", "")
            date_raw = item.get("date", "")
            if date_raw and len(date_raw) >= 10:
                date_str = date_raw[:10]
            else:
                date_str = "Recent"

            cred = identify_source_credibility(url, source_raw)
            articles.append({
                "title": title,
                "source": cred["source"],
                "url": url,
                "snippet": body,
                "date": date_str,
                "credibility_tier": cred["tier"],
                "is_priority": cred.get("is_priority", False)
            })
        return {"success": True, "articles": articles, "provider": "Live News Provider (DuckDuckGo Search)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_credible_news(query: str, max_results: int = 8):
    """
    Main news search entrypoint.
    Checks environment for NEWS_API_KEY.
    If valid, uses NewsAPI. If absent or fails, automatically falls back to live news search.
    Reports honest status if sources are unavailable.
    """
    api_key = os.environ.get("NEWS_API_KEY", "").strip()

    # Try NewsAPI if configured
    if api_key and api_key != "your_newsapi_key_here":
        res = search_newsapi(query, api_key, max_results)
        if res.get("success") and res.get("articles"):
            return res
        print(f"[NewsSearch] NewsAPI returned no results or failed: {res.get('error')}. Falling back to live search...")

    # Fallback to zero-key search
    ddg_res = search_ddg_news(query, max_results)
    if ddg_res.get("success"):
        return ddg_res

    # If both failed, report truthfully
    return {
        "success": False,
        "articles": [],
        "provider": "None",
        "error": "Live news search provider is temporarily unreachable. Verification marked unverified due to lack of network search availability."
    }
