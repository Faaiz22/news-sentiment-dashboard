# app.py
# app.py
"""
News Sentiment Monitor (Streamlit)

"""

import os
import time
import re
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# Configuration
# -------------------------
class Config:
    # Prefer environment variable; override in sidebar if needed
    NEWS_API_KEY = os.getenv("NEWSDATA_API_KEY", "")
    NEWS_API_URL = "https://newsdata.io/api/1/latest"
    UPDATE_INTERVAL = 300
    MAX_ARTICLES = 500

# -------------------------
# SimpleNewsFetcher (NewsData.io)
# -------------------------
class SimpleNewsFetcher:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key required for SimpleNewsFetcher")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "news-sentiment-monitor/1.0"})

    def fetch_news(self,
                   category: Optional[str] = None,
                   country: Optional[str] = None,
                   language: str = "en",
                   max_pages: int = 1) -> List[Dict]:
        """
        Fetch news using NewsData.io /latest with cursor pagination.
        - First request made WITHOUT 'page' param.
        - For subsequent requests, use the `nextPage` token returned by API.
        - max_pages: how many pages to request (1 = only the first page).
        Returns list of normalized articles.
        """
        articles: List[Dict] = []
        page_token: Optional[str] = None
        pages_fetched = 0

        while pages_fetched < max_pages:
            params = {"apikey": self.api_key, "language": language}
            if category:
                params["category"] = category
            if country:
                params["country"] = country
            if page_token:
                params["page"] = page_token  # cursor token, not numeric index

            try:
                resp = self.session.get(Config.NEWS_API_URL, params=params, timeout=12)
            except requests.RequestException as e:
                st.error(f"Network/API error while calling NewsData.io: {e}")
                break

            # If non-200, show response body for debugging
            if resp.status_code != 200:
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text[:1000]
                st.error(f"NewsData API returned {resp.status_code}: {body}")
                break

            try:
                data = resp.json()
            except ValueError:
                st.error("NewsData returned invalid JSON")
                break

            # Response structure: { "status": "success", "results": [...], "nextPage": "..." }
            status = data.get("status")
            if status not in ("success", "ok"):
                # often message is inside results or message field
                err_info = data.get("results") or data.get("message") or data
                st.error(f"NewsData non-success status: {status} - {err_info}")
                break

            results = data.get("results") or []
            if not isinstance(results, list) or len(results) == 0:
                # nothing to process
                break

            for r in results:
                title = (r.get("title") or "").strip()
                if not title or title.lower() in ("[removed]", "removed"):
                    continue
                description = (r.get("description") or r.get("content") or "").strip()
                source = r.get("source_id") or r.get("source") or "unknown"
                url = r.get("link") or r.get("url") or ""
                pubdate = r.get("pubDate") or ""

                normalized = {
                    "title": title,
                    "description": description,
                    "source": source,
                    "url": url,
                    "published_at": pubdate,
                    "category": category or r.get("category") or "unknown",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
                articles.append(normalized)

            pages_fetched += 1

            # get nextPage token (cursor). If absent -> stop
            page_token = data.get("nextPage")
            if not page_token:
                break

            # safety cap
            if len(articles) >= Config.MAX_ARTICLES:
                articles = articles[: Config.MAX_ARTICLES]
                break

            time.sleep(0.3)

        return articles

# -------------------------
# SimpleSentimentModel
# -------------------------
class SimpleSentimentModel:
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.is_trained = False

    def create_training_data(self) -> pd.DataFrame:
        positive_samples = [
            "Stock market reaches new heights with strong earnings",
            "Breakthrough in medical research offers new hope",
            "Economy shows signs of robust recovery",
            "Technology innovation creates new opportunities",
            "Education program achieves remarkable success",
            "Clean energy project exceeds expectations",
            "Community celebrates new development milestone",
            "Research team discovers promising treatment",
            "Company reports record profits and growth",
            "International cooperation leads to positive outcomes",
            "Scientific advancement promises better future",
            "Local business expansion creates jobs",
            "Environmental protection efforts show progress",
            "Students achieve academic excellence",
            "Healthcare improvements benefit millions"
        ]

        negative_samples = [
            "Market crashes amid economic uncertainty",
            "Natural disaster causes widespread damage",
            "Company announces major layoffs",
            "Crime rates increase in urban areas",
            "Environmental crisis worsens conditions",
            "Healthcare system faces critical shortage",
            "Government corruption scandal emerges",
            "Supply chain disruptions affect economy",
            "Cyber attack compromises data security",
            "Infrastructure failures cause delays",
            "Budget cuts threaten public services",
            "Trade disputes escalate tensions",
            "Unemployment rates reach concerning levels",
            "Privacy violations exposed in investigation",
            "Safety concerns raised over products"
        ]

        data = []
        for t in positive_samples:
            data.append({"text": t, "sentiment": 1})
        for t in negative_samples:
            data.append({"text": t, "sentiment": 0})
        return pd.DataFrame(data)

    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def train_model(self) -> float:
        df = self.create_training_data()
        df["processed_text"] = df["text"].apply(self.preprocess_text)

        X = df["processed_text"].values
        y = df["sentiment"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=2000, stop_words="english")),
            ("clf", LogisticRegression(random_state=42, max_iter=400, solver="liblinear"))
        ])

        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        self.is_trained = True
        return float(acc)

    def predict_sentiment(self, text: str) -> Dict:
        if not self.is_trained or self.pipeline is None:
            return {"sentiment": "neutral", "confidence": 0.5, "probability_positive": 0.5, "probability_negative": 0.5}

        processed = self.preprocess_text(text)
        if not processed:
            return {"sentiment": "neutral", "confidence": 0.5, "probability_positive": 0.5, "probability_negative": 0.5}

        preds = self.pipeline.predict([processed])
        probs = self.pipeline.predict_proba([processed])[0] if hasattr(self.pipeline, "predict_proba") else [0.5, 0.5]

        prob_positive = float(probs[1]) if len(probs) > 1 else float(probs[0])
        prob_negative = float(probs[0]) if len(probs) > 1 else 1.0 - prob_positive
        sentiment = "positive" if int(preds[0]) == 1 else "negative"
        confidence = max(prob_positive, prob_negative)

        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "probability_positive": prob_positive,
            "probability_negative": prob_negative
        }

    def save_model(self, path: str):
        if self.pipeline is None:
            raise ValueError("No trained pipeline to save")
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load_model(self, path: str):
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True

# -------------------------
# SimpleDataStore
# -------------------------
class SimpleDataStore:
    def __init__(self):
        self.articles: List[Dict] = []
        self.processed_articles: List[Dict] = []

    def add_articles(self, articles: List[Dict]):
        if not articles:
            return
        self.articles.extend(articles)
        self.articles = self.articles[-Config.MAX_ARTICLES :]

    def add_processed_articles(self, processed: List[Dict]):
        if not processed:
            return
        self.processed_articles.extend(processed)
        self.processed_articles = self.processed_articles[-Config.MAX_ARTICLES :]

    def get_recent_articles(self, limit: int = 100) -> List[Dict]:
        return list(self.processed_articles[-limit:])

    def get_sentiment_stats(self) -> Dict:
        if not self.processed_articles:
            return {"positive": 0, "negative": 0, "total": 0}
        sentiments = [a.get("sentiment", "neutral") for a in self.processed_articles]
        positive = sentiments.count("positive")
        negative = sentiments.count("negative")
        return {"positive": positive, "negative": negative, "total": len(sentiments)}

# -------------------------
# Pipeline
# -------------------------
class ColadNewsSentimentPipeline:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key if api_key is not None else Config.NEWS_API_KEY) or ""
        self.fetcher: Optional[SimpleNewsFetcher] = None
        if self.api_key:
            try:
                self.fetcher = SimpleNewsFetcher(self.api_key)
            except ValueError:
                self.fetcher = None

        self.model = SimpleSentimentModel()
        self.data_store = SimpleDataStore()
        self.last_update: Optional[datetime] = None

    def setup(self):
        acc = self.model.train_model()
        st.success(f"Model trained (sample data) — accuracy: {acc:.2%}")

        if self.fetcher:
            self.fetch_and_process_news()
        else:
            self._load_sample_data()

    def fetch_and_process_news(self,
                               countries: Optional[List[Optional[str]]] = None,
                               categories: Optional[List[str]] = None,
                               max_pages_per_category: int = 1):
        if not self.fetcher:
            st.warning("No API key configured — using sample data.")
            self._load_sample_data()
            return

        if categories is None:
            categories = ["technology", "business", "health", "science", "world"]
        if countries is None:
            countries = [None]  # None indicates no country filter

        all_articles: List[Dict] = []
        for country in countries:
            for category in categories:
                try:
                    fetched = self.fetcher.fetch_news(category=category, country=country, language="en", max_pages=max_pages_per_category)
                except Exception as e:
                    st.error(f"Error fetching {category}/{country}: {e}")
                    fetched = []
                all_articles.extend(fetched)
                time.sleep(0.2)
                if len(all_articles) >= Config.MAX_ARTICLES:
                    break
            if len(all_articles) >= Config.MAX_ARTICLES:
                break

        if not all_articles:
            st.warning("No articles fetched from NewsData.io — check API key / quota.")
            return

        # dedupe by url/title
        seen = set()
        unique = []
        for a in all_articles:
            key = (a.get("url") or a.get("title") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(a)

        processed = []
        for art in unique:
            title = art.get("title", "")
            desc = art.get("description", "")
            text = f"{title} {desc}".strip()
            sentiment = self.model.predict_sentiment(text)
            proc = {
                **art,
                "text": text,
                "sentiment": sentiment["sentiment"],
                "confidence": sentiment["confidence"],
                "probability_positive": sentiment["probability_positive"],
                "probability_negative": sentiment["probability_negative"],
                "processed_at": datetime.utcnow().isoformat() + "Z"
            }
            processed.append(proc)

        self.data_store.add_articles(unique)
        self.data_store.add_processed_articles(processed)
        self.last_update = datetime.utcnow()
        st.success(f"Fetched & processed {len(processed)} articles")

    def _load_sample_data(self):
        sample_articles = [
            {
                "title": "Tech Giant Reports Record Quarterly Earnings",
                "description": "Company shows strong growth across all sectors",
                "source": "TechNews",
                "category": "technology",
                "published_at": (datetime.utcnow() - timedelta(hours=1)).isoformat() + "Z"
            },
            {
                "title": "Market Volatility Concerns Investors Worldwide",
                "description": "Economic uncertainty continues to affect global markets",
                "source": "Financial Times",
                "category": "business",
                "published_at": (datetime.utcnow() - timedelta(hours=2)).isoformat() + "Z"
            }
        ]
        processed = []
        for a in sample_articles:
            text = f"{a['title']} {a.get('description','')}".strip()
            sentiment = self.model.predict_sentiment(text)
            proc = {
                **a,
                "text": text,
                "sentiment": sentiment["sentiment"],
                "confidence": sentiment["confidence"],
                "probability_positive": sentiment["probability_positive"],
                "probability_negative": sentiment["probability_negative"],
                "processed_at": datetime.utcnow().isoformat() + "Z"
            }
            processed.append(proc)

        self.data_store.add_processed_articles(processed)
        self.last_update = datetime.utcnow()
        st.info("Loaded sample data (no API key)")

# -------------------------
# Streamlit Dashboard
# -------------------------
def create_dashboard():
    st.set_page_config(page_title="News Sentiment Monitor", page_icon="📈", layout="wide")
    st.title("📈 News Sentiment Monitor — NewsData.io")
    st.markdown("Lightweight demo that fetches headlines from NewsData.io and runs a toy sentiment classifier.")

    # Sidebar: API key + controls
    st.sidebar.header("Configuration")
    api_key_input = st.sidebar.text_input(
        "NewsData.io API Key (optional). If empty, env var NEWSDATA_API_KEY will be used.",
        value="",
        type="password",
    )
    use_sample_if_no_key = st.sidebar.checkbox("Use sample data if no key", value=True)
    max_pages = st.sidebar.number_input("Pages per category (max 3 recommended)", min_value=1, max_value=5, value=1, step=1)
    init_button = st.sidebar.button("Initialize / Reinitialize")

    if "pipeline" not in st.session_state or init_button:
        key_to_use = api_key_input.strip() or Config.NEWS_API_KEY
        if not key_to_use and not use_sample_if_no_key:
            st.sidebar.error("Provide API key or enable 'Use sample data' to proceed.")
        pipeline = ColadNewsSentimentPipeline(api_key=key_to_use)
        st.session_state.pipeline = pipeline
        with st.spinner("Setting up model and initial data..."):
            pipeline.setup()

    pipeline: ColadNewsSentimentPipeline = st.session_state.pipeline

    # Controls
    left, mid, right = st.columns([1, 1, 2])
    with left:
        if st.button("🔄 Refresh Data (manual)"):
            with st.spinner("Fetching latest news..."):
                pipeline.fetch_and_process_news(max_pages_per_category=int(max_pages))
    with mid:
        auto_refresh = st.checkbox("Auto refresh every 30s", value=False)
    with right:
        if pipeline.last_update:
            st.info(f"Last update (UTC): {pipeline.last_update.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    recent = pipeline.data_store.get_recent_articles(200)
    stats = pipeline.data_store.get_sentiment_stats()

    if stats["total"] > 0:
        pos = stats["positive"]
        neg = stats["negative"]
        total = stats["total"]
        pos_pct = (pos / total) * 100 if total else 0.0
        avg_conf = np.mean([a.get("confidence", 0.0) for a in recent]) if recent else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Articles", total)
        c2.metric("Positive", f"{pos} ({pos_pct:.1f}%)")
        c3.metric("Negative", f"{neg} ({100-pos_pct:.1f}%)")
        c4.metric("Avg Confidence", f"{avg_conf:.2%}")

        fig = px.pie(names=["Positive", "Negative"], values=[pos, neg], title="Sentiment distribution (recent)")
        st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame(recent)
        cols = ["title", "source", "category", "sentiment", "confidence", "processed_at"]
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        display_df = df[cols].copy()
        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{float(x):.2%}" if x != "" else "N/A")
        st.subheader("Recent articles")
        st.dataframe(display_df, use_container_width=True, height=400)

        pos_list = [a for a in recent if a.get("sentiment") == "positive"]
        pos_list = sorted(pos_list, key=lambda x: x.get("confidence", 0), reverse=True)[:5]
        neg_list = [a for a in recent if a.get("sentiment") == "negative"]
        neg_list = sorted(neg_list, key=lambda x: x.get("confidence", 0), reverse=True)[:5]

        lcol, rcol = st.columns(2)
        with lcol:
            st.subheader("🟢 Most positive")
            if pos_list:
                for art in pos_list:
                    st.write(f"**{art.get('title')}**")
                    st.caption(f"Source: {art.get('source')} | Confidence: {art.get('confidence'):.2%}")
                    if art.get("url"):
                        st.write(art.get("url"))
                    st.markdown("---")
            else:
                st.write("No positive articles found.")
        with rcol:
            st.subheader("🔴 Most negative")
            if neg_list:
                for art in neg_list:
                    st.write(f"**{art.get('title')}**")
                    st.caption(f"Source: {art.get('source')} | Confidence: {art.get('confidence'):.2%}")
                    if art.get("url"):
                        st.write(art.get("url"))
                    st.markdown("---")
            else:
                st.write("No negative articles found.")
    else:
        st.warning("No processed articles yet. Click 'Refresh Data' to fetch (or initialize with API key).")

    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()

if __name__ == "__main__":
    create_dashboard()

