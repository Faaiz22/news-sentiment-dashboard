
# Paste the complete colab_news_sentiment.py code here
# colab_news_sentiment.py
"""
Simplified News Sentiment Analysis for Google Colab
This version works without Kafka/Spark infrastructure
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import pickle
import os
from typing import List, Dict, Optional

# Configuration
class Config:
    NEWS_API_KEY = "pub_e9c5fbef127c4d8ea7b60c1f05259361"  
    NEWS_API_URL = "https://newsdata.io/api/1/news?"
    UPDATE_INTERVAL = 300  # 5 minutes in seconds
    MAX_ARTICLES = 100

# Simple News Fetcher (no Kafka needed)
class SimpleNewsFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'X-Api-Key': api_key})
        
    def fetch_news(self, category: str = 'general', country: str = 'us') -> List[Dict]:
        """Fetch news from NewsAPI"""
        params = {
            'category': category,
            'country': country,
            'pageSize': 50
        }
        
        try:
            response = self.session.get(Config.NEWS_API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'ok':
                articles = []
                for article in data['articles']:
                    if article['title'] and article['title'] != '[Removed]':
                        articles.append({
                            'title': article['title'],
                            'description': article.get('description', ''),
                            'source': article['source']['name'],
                            'url': article['url'],
                            'published_at': article['publishedAt'],
                            'category': category,
                            'timestamp': datetime.now().isoformat()
                        })
                return articles
            else:
                st.error(f"API Error: {data.get('message', 'Unknown error')}")
                return []
        except Exception as e:
            st.error(f"Failed to fetch news: {str(e)}")
            return []

# Simple ML Model (no Spark needed)
class SimpleSentimentModel:
    def __init__(self):
        self.pipeline = None
        self.is_trained = False
    
    def create_training_data(self) -> pd.DataFrame:
        """Create sample training data"""
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
        
        # Create training DataFrame
        data = []
        for text in positive_samples:
            data.append({'text': text, 'sentiment': 1})
        for text in negative_samples:
            data.append({'text': text, 'sentiment': 0})
        
        return pd.DataFrame(data)
    
    def preprocess_text(self, text: str) -> str:
        """Simple text preprocessing"""
        if not text:
            return ""
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train_model(self):
        """Train the sentiment model"""
        # Get training data
        train_df = self.create_training_data()
        
        # Preprocess text
        train_df['processed_text'] = train_df['text'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            train_df['processed_text'], 
            train_df['sentiment'], 
            test_size=0.2, 
            random_state=42
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        return accuracy
    
    def predict_sentiment(self, text: str) -> Dict:
        """Predict sentiment for a single text"""
        if not self.is_trained:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        prediction = self.pipeline.predict([processed_text])[0]
        probability = self.pipeline.predict_proba([processed_text])[0]
        
        sentiment = 'positive' if prediction == 1 else 'negative'
        confidence = max(probability)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'probability_positive': probability[1] if len(probability) > 1 else 0.5,
            'probability_negative': probability[0] if len(probability) > 1 else 0.5
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    def load_model(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True

# Data Storage (simple in-memory storage)
class SimpleDataStore:
    def __init__(self):
        self.articles = []
        self.processed_articles = []
    
    def add_articles(self, articles: List[Dict]):
        """Add new articles to store"""
        self.articles.extend(articles)
        # Keep only last 500 articles
        self.articles = self.articles[-500:]
    
    def add_processed_articles(self, processed: List[Dict]):
        """Add processed articles with sentiment"""
        self.processed_articles.extend(processed)
        # Keep only last 500 processed articles
        self.processed_articles = self.processed_articles[-500:]
    
    def get_recent_articles(self, limit: int = 100) -> List[Dict]:
        """Get recent processed articles"""
        return self.processed_articles[-limit:]
    
    def get_sentiment_stats(self) -> Dict:
        """Get sentiment statistics"""
        if not self.processed_articles:
            return {'positive': 0, 'negative': 0, 'total': 0}
        
        sentiments = [a['sentiment'] for a in self.processed_articles]
        positive_count = sentiments.count('positive')
        negative_count = sentiments.count('negative')
        
        return {
            'positive': positive_count,
            'negative': negative_count,
            'total': len(sentiments)
        }

# Main Pipeline Class
class ColadNewsSentimentPipeline:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or Config.NEWS_API_KEY
        self.fetcher = SimpleNewsFetcher(self.api_key) if self.api_key != "your-newsapi-key-here" else None
        self.model = SimpleSentimentModel()
        self.data_store = SimpleDataStore()
        self.last_update = None
        
    def setup(self):
        """Initialize the pipeline"""
        # Train model
        accuracy = self.model.train_model()
        st.success(f"Model trained with accuracy: {accuracy:.2%}")
        
        # Try to load some initial data
        if self.fetcher:
            self.fetch_and_process_news()
        else:
            self._load_sample_data()
    
    def fetch_and_process_news(self):
        """Fetch news and process sentiment"""
        if not self.fetcher:
            st.warning("No API key provided, using sample data")
            self._load_sample_data()
            return
        
        # Fetch from multiple categories
        categories = ['general', 'business', 'technology', 'health']
        all_articles = []
        
        for category in categories:
            articles = self.fetcher.fetch_news(category)
            all_articles.extend(articles)
            time.sleep(1)  # Rate limiting
        
        if all_articles:
            # Process sentiment
            processed_articles = []
            for article in all_articles:
                # Combine title and description for sentiment analysis
                text = f"{article['title']} {article.get('description', '')}"
                sentiment_result = self.model.predict_sentiment(text)
                
                processed_article = {
                    **article,
                    'text': text,
                    **sentiment_result,
                    'processed_at': datetime.now().isoformat()
                }
                processed_articles.append(processed_article)
            
            # Store data
            self.data_store.add_articles(all_articles)
            self.data_store.add_processed_articles(processed_articles)
            self.last_update = datetime.now()
            
            st.success(f"Processed {len(processed_articles)} articles")
        else:
            st.error("No articles fetched")
    
    def _load_sample_data(self):
        """Load sample data for demonstration"""
        sample_articles = [
            {
                'title': 'Tech Giant Reports Record Quarterly Earnings',
                'description': 'Company shows strong growth across all sectors',
                'source': 'TechNews',
                'category': 'technology',
                'published_at': (datetime.now() - timedelta(hours=1)).isoformat(),
            },
            {
                'title': 'Market Volatility Concerns Investors Worldwide',
                'description': 'Economic uncertainty continues to affect global markets',
                'source': 'Financial Times',
                'category': 'business',
                'published_at': (datetime.now() - timedelta(hours=2)).isoformat(),
            },
            # Add more sample articles here...
        ]
        
        # Process sample articles
        processed_articles = []
        for article in sample_articles:
            text = f"{article['title']} {article.get('description', '')}"
            sentiment_result = self.model.predict_sentiment(text)
            
            processed_article = {
                **article,
                'text': text,
                **sentiment_result,
                'processed_at': datetime.now().isoformat()
            }
            processed_articles.append(processed_article)
        
        self.data_store.add_processed_articles(processed_articles)
        self.last_update = datetime.now()

# Streamlit Dashboard
def create_dashboard():
    st.set_page_config(
        page_title="News Sentiment Monitor",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Real-Time News Sentiment Analysis")
    st.markdown("---")
    
    # Initialize pipeline
    if 'pipeline' not in st.session_state:
        # Get API key from user
        api_key = st.sidebar.text_input(
            "NewsAPI Key (optional)", 
            value="",
            type="password",
            help="Get free key from https://newsapi.org"
        )
        
        st.session_state.pipeline = ColadNewsSentimentPipeline(api_key)
        st.session_state.pipeline.setup()
    
    pipeline = st.session_state.pipeline
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔄 Refresh Data"):
            with st.spinner("Fetching latest news..."):
                pipeline.fetch_and_process_news()
    
    with col2:
        auto_refresh = st.checkbox("Auto Refresh (30s)")
    
    with col3:
        if pipeline.last_update:
            st.info(f"Last updated: {pipeline.last_update.strftime('%H:%M:%S')}")
    
    # Get current data
    recent_articles = pipeline.data_store.get_recent_articles(100)
    stats = pipeline.data_store.get_sentiment_stats()
    
    if stats['total'] > 0:
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        positive_pct = (stats['positive'] / stats['total']) * 100
        
        with col1:
            st.metric("Total Articles", stats['total'])
        with col2:
            st.metric("Positive", f"{stats['positive']} ({positive_pct:.1f}%)")
        with col3:
            st.metric("Negative", f"{stats['negative']} ({100-positive_pct:.1f}%)")
        with col4:
            avg_confidence = np.mean([a['confidence'] for a in recent_articles])
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        # Sentiment distribution chart
        fig_pie = px.pie(
            values=[stats['positive'], stats['negative']], 
            names=['Positive', 'Negative'],
            title="Sentiment Distribution",
            color_discrete_map={'Positive': 'green', 'Negative': 'red'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Recent articles table
        st.subheader("Recent Articles")
        
        df = pd.DataFrame(recent_articles)
        display_df = df[['title', 'source', 'sentiment', 'confidence', 'category']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Top articles by sentiment
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 Most Positive")
            positive_articles = [a for a in recent_articles if a['sentiment'] == 'positive']
            positive_articles.sort(key=lambda x: x['confidence'], reverse=True)
            for article in positive_articles[:5]:
                st.write(f"**{article['title']}**")
                st.write(f"Source: {article['source']} | Confidence: {article['confidence']:.2%}")
                st.write("---")
        
        with col2:
            st.subheader("🔴 Most Negative")
            negative_articles = [a for a in recent_articles if a['sentiment'] == 'negative']
            negative_articles.sort(key=lambda x: x['confidence'], reverse=True)
            for article in negative_articles[:5]:
                st.write(f"**{article['title']}**")
                st.write(f"Source: {article['source']} | Confidence: {article['confidence']:.2%}")
                st.write("---")
    
    else:
        st.warning("No data available. Click 'Refresh Data' to fetch news articles.")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(30)
        st.experimental_rerun()

if __name__ == "__main__":
    create_dashboard()
