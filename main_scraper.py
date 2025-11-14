import os
import asyncio
from typing import Optional, Any
import trafilatura
import httpx 
import hashlib 
import json 
from firebase_admin import credentials, firestore, initialize_app
import firebase_admin 
from collections import defaultdict

# --- 1. FIREBASE INITIALIZATION & CONFIGURATION ---

def initialize_firebase():
    """Initializes the Firebase Admin SDK using the service account file or environment variable."""
    if not firebase_admin._apps:
        try:
            # Check if we're in GitHub Actions (uses FIREBASE_CREDENTIALS_PATH env var)
            creds_path = os.getenv('FIREBASE_CREDENTIALS_PATH', './firebase-service-account.json')
            
            if not os.path.exists(creds_path):
                # Try alternative paths for different environments
                alt_paths = ['./key.json', '../firebase-service-account.json']
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        creds_path = alt_path
                        break
                else:
                    print(f"\n*** ERROR: Firebase credentials file not found ***")
                    print(f"*** Searched paths: {creds_path}, {', '.join(alt_paths)} ***")
                    print("*** For local: Ensure 'firebase-service-account.json' or 'key.json' exists ***")
                    print("*** For GitHub Actions: Check FIREBASE_CREDENTIALS_PATH env var ***\n")
                    return
            
            cred = credentials.Certificate(creds_path)
            initialize_app(cred)
            print(f"✅ Firebase Admin SDK initialized using: {creds_path}")
            
        except Exception as e:
            print(f"🚨 FATAL ERROR initializing Firebase: {e}")

def get_scraper_config() -> dict[str, list[str]]:
    """Reads configuration from Firestore or uses hardcoded values if uninitialized."""
    try:
        db = firestore.client()
        doc_ref = db.collection("config").document("scraper_settings")
        doc = doc_ref.get()
        if doc.exists:
            config = doc.to_dict()
            print(f"✅ Loaded config from Firestore: {len(config.get('keywords', []))} keywords, {len(config.get('rss_feeds', []))} RSS feeds")
            return config
    except Exception as e:
        print(f"⚠️ Could not load config from Firestore: {e}")
        
    # Fallback to default configuration
    default_config = {
        "keywords": ["geopolitics", "trade war", "AI regulation"], 
        "rss_feeds": [
            "https://www.cnbc.com/id/100003114/device/rss/rss.html", 
            "http://rss.cnn.com/rss/cnn_topstories.rss"
        ],
        "subreddits": ["worldnews", "geopolitics"]
    }
    print(f"ℹ️ Using default configuration")
    return default_config

def get_existing_hashes() -> set[str]:
    """Retrieves all existing url_hash values from the articles collection for deduplication."""
    try:
        db = firestore.client()
        hashes = set()
        docs = db.collection("articles").stream()
        for doc in docs:
            data = doc.to_dict()
            if 'url_hash' in data:
                hashes.add(data['url_hash'])
        print(f"✅ Retrieved {len(hashes)} existing article hashes for deduplication.")
        return hashes
    except Exception as e:
        print(f"❌ ERROR: Failed to retrieve existing hashes from Firestore: {e}")
        return set()

def save_articles_to_firestore(articles_list: list[dict]):
    """
    Saves a list of processed articles to Firestore using Batched Writes
    to reduce cost and increase efficiency. (SCALING SOLUTION: BATCH WRITES)
    """
    if not articles_list:
        print("ℹ️ No articles to save.")
        return
        
    db = firestore.client()
    batch = db.batch()
    articles_ref = db.collection('articles')
    total_saved = 0
    
    for article in articles_list:
        doc_ref = articles_ref.document(article['url_hash'])
        
        # Build the final document to be saved
        final_data = {
            "url": article.get('url'),
            "url_hash": article.get('url_hash'),
            "title": article.get('title'),
            "source_type": article.get('source_type'),
            "timestamp": article.get('timestamp'),
            # Merge analysis results (sentiment, entities, keywords_matched)
            **article.get('analysis_results', {}),
            # NOTE: clean_content is NOT saved to the DB to save cost/size.
        }
        
        batch.set(doc_ref, final_data)
        total_saved += 1
        
        # Commit the batch every 500 operations (Firestore limit is 500)
        if total_saved % 500 == 0:
            batch.commit()
            print(f"📦 Committed {total_saved} documents to Firestore.")
            batch = db.batch() # Start a new batch
            
    # Commit any remaining operations
    if total_saved % 500 != 0:
        batch.commit()

    print(f"✅ Successfully saved a total of {total_saved} new documents via batch writes.")

# --- 2. QUALITY CONTROL AND EXTRACTION LAYER ---

def fetch_content(url: str) -> Optional[str]:
    """Fetches a URL and extracts only the main article text."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None
        
        content = trafilatura.extract(downloaded, favor_precision=True)
        
        if content is None or len(content) < 100:
            # Handles 'Garbage In, Garbage Out' (paywalls, short snippets)
            return None
            
        return content
        
    except Exception as e:
        print(f"⚠️ Content extraction failed for {url}: {e}")
        return None

# --- 3. AI ANALYSIS FUNCTION (Placeholder for API Offload) ---

def analyze_text(title: str, content: str, keywords: list[str]) -> dict[str, Any]:
    """
    Placeholder for AI analysis. The final implementation will use ASYNC httpx 
    to call the Hugging Face Inference API.
    """
    # Simple keyword matching for now
    matched_keywords = [kw for kw in keywords if kw.lower() in content.lower() or kw.lower() in title.lower()]
    
    return {
        "sentiment": {"label": "NEUTRAL", "score": 0.5},
        "entities": [],
        "keywords_matched": matched_keywords,
    }

# --- 4. ASYNCHRONOUS DATA FETCHING (SCALING SOLUTION) ---

def generate_hash(url: str) -> str:
    """Generates a SHA-256 hash of the URL for unique identification."""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

async def fetch_gnews_articles(keywords: list[str]) -> list[dict]:
    """Asynchronously fetches articles for configured keywords using the GNews API."""
    gnews_api_key = os.environ.get("GNEWS_API_KEY", "YOUR_PLACEHOLDER_KEY")
    
    if gnews_api_key == "YOUR_PLACEHOLDER_KEY":
        print("⚠️ WARNING: GNEWS_API_KEY not set. Skipping GNews scraping.")
        return []
    
    base_url = "https://gnews.io/api/v4/search"
    all_articles = []
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        tasks = []
        for keyword in keywords:
            params = {"q": keyword, "apikey": gnews_api_key, "lang": "en", "max": 10}
            tasks.append(client.get(base_url, params=params))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"❌ GNews API error for '{keywords[i]}': {response}")
                continue
            if response.status_code == 200:
                try:
                    data = response.json()
                    articles = data.get('articles', [])
                    for article in articles:
                        article['source_type'] = 'GNEWS'
                        article['url_hash'] = generate_hash(article['url'])
                    all_articles.extend(articles)
                    print(f"✅ Fetched {len(articles)} articles for keyword: {keywords[i]}")
                except Exception as e:
                    print(f"❌ Failed to parse GNews response: {e}")
            else:
                print(f"❌ GNews API returned status {response.status_code}")
                
    return all_articles

async def fetch_rss_articles(rss_feeds: list[str]) -> list[dict]:
    """Placeholder for asynchronous RSS fetching."""
    # TODO: Implement RSS parsing with feedparser
    print("ℹ️ RSS fetching not yet implemented")
    return []

# --- 5. SERVER-SIDE AGGREGATION (SCALING SOLUTION: FRONTEND DECOUPLING) ---

def aggregate_metrics(articles_list: list[dict]):
    """
    Calculates summary data (metrics) from newly saved articles.
    This prevents the frontend from querying massive datasets.
    """
    metrics = defaultdict(int)
    keyword_counts = defaultdict(int)

    # Example aggregation logic: Count keywords matched
    for article in articles_list:
        if 'keywords_matched' in article.get('analysis_results', {}):
            for keyword in article['analysis_results']['keywords_matched']:
                keyword_counts[keyword] += 1
    
    # Store aggregated data
    metrics['total_new_articles'] = len(articles_list)
    metrics['keyword_counts'] = dict(keyword_counts)
    
    # Save to the dashboard_metrics collection
    db = firestore.client()
    db.collection('dashboard_metrics').document('latest_metrics').set(metrics, merge=True)
    print(f"✅ Metrics aggregated and saved: {len(articles_list)} new articles, {len(keyword_counts)} unique keywords")


# --- 6. MAIN PIPELINE ORCHESTRATION ---

async def main_pipeline():
    """
    The main asynchronous orchestrator for the entire data pipeline.
    """
    print("\n" + "="*70)
    print("🛡️  SENTINEL DASHBOARD PIPELINE - Starting")
    print("="*70 + "\n")
    
    initialize_firebase()
    
    if not firebase_admin._apps:
        print("❌ Pipeline aborted: Firebase initialization failed")
        return
    
    # 1. Load configuration
    config = get_scraper_config()
    keywords = config.get('keywords', [])
    rss_feeds = config.get('rss_feeds', [])
    
    # 2. Deduplication Preparation 
    existing_hashes = get_existing_hashes() 
    
    # 3. Fetch data concurrently
    print("\n📡 Fetching articles from sources...")
    fetch_tasks = [
        fetch_gnews_articles(keywords),
        fetch_rss_articles(rss_feeds),
    ]
    gnews_data, rss_data = await asyncio.gather(*fetch_tasks)
    all_raw_articles = gnews_data + rss_data
    print(f"\n📊 Total raw articles fetched: {len(all_raw_articles)}")

    # 4. Deduplication and Filter
    new_articles = [article for article in all_raw_articles if article['url_hash'] not in existing_hashes]
    print(f"✅ Filtered down to {len(new_articles)} new articles for analysis")

    # 5. Content Extraction, Filtering, and Analysis
    articles_to_save = []
    server_timestamp = firestore.SERVER_TIMESTAMP

    if new_articles:
        print("\n🔍 Starting content extraction and analysis...")
        
        for i, article in enumerate(new_articles, 1):
            if not article.get('url') or not article.get('title'):
                continue
            
            print(f"  Processing [{i}/{len(new_articles)}]: {article.get('title', 'Untitled')[:60]}...")
                
            clean_content = fetch_content(article['url'])
            
            if clean_content:
                # AI Analysis
                analysis = analyze_text(article['title'], clean_content, keywords)
                
                # Prepare final document structure
                final_document = {
                    "url": article['url'],
                    "url_hash": article['url_hash'],
                    "title": article['title'],
                    "source_type": article['source_type'],
                    "timestamp": server_timestamp,
                    "analysis_results": analysis,
                }
                articles_to_save.append(final_document)

        print(f"\n✅ Content extraction complete: {len(articles_to_save)}/{len(new_articles)} articles ready")
        
        # 6. Database Storage (Batch Write)
        if articles_to_save:
            print("\n💾 Saving to Firestore...")
            save_articles_to_firestore(articles_to_save)

            # 7. Server-Side Aggregation (SCALING SOLUTION: FRONTEND DECOUPLING)
            print("\n📊 Aggregating metrics...")
            aggregate_metrics(articles_to_save)
    else:
        print("\nℹ️ No new articles to process")
    
    print("\n" + "="*70)
    print("✅ PIPELINE FINISHED SUCCESSFULLY!")
    print("="*70 + "\n")


# --- 7. ENTRY POINT ---
if __name__ == "__main__":
    # Check for API key
    if os.environ.get("GNEWS_API_KEY") in [None, "YOUR_PLACEHOLDER_KEY"]:
        print("\n⚠️  WARNING: GNEWS_API_KEY is not set or using placeholder")
        print("   Set it with: export GNEWS_API_KEY='your-key-here'")
        print("   Pipeline will run but skip GNews scraping\n")

    try:
        asyncio.run(main_pipeline())
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline stopped by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()