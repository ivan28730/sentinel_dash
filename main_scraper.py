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
    """Initializes the Firebase Admin SDK using the service account file."""
    if not firebase_admin._apps:
        try:
            # NOTE: For local testing, we look for the file. 
            cred = credentials.Certificate("./firebase-service-account.json")
            initialize_app(cred)
            print("Firebase Admin SDK initialized.")
        except FileNotFoundError:
            print("\n*** ERROR: Firebase service account file not found. ***")
            print("*** Please ensure 'firebase-service-account.json' is in the root directory. ***\n")
        except Exception as e:
            print(f"FATAL ERROR initializing Firebase: {e}")

def get_scraper_config() -> dict[str, list[str]]:
    """Reads configuration from Firestore or uses hardcoded values if uninitialized."""
    try:
        db = firestore.client()
        doc_ref = db.collection("config").document("scraper_settings")
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
    except Exception:
        pass 
        
    return {
        "keywords": ["geopolitics", "trade war", "AI regulation"], 
        "rss_feeds": [
            "https://www.cnbc.com/id/100003114/device/rss/rss.html", 
            "http://rss.cnn.com/rss/cnn_topstories.rss"
        ],
        "subreddits": ["worldnews", "geopolitics"]
    }

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
        print(f"Retrieved {len(hashes)} existing article hashes for deduplication.")
        return hashes
    except Exception as e:
        print(f"ERROR: Failed to retrieve existing hashes from Firestore: {e}")
        return set()

def save_articles_to_firestore(articles_list: list[dict]):
    """
    Saves a list of processed articles to Firestore using Batched Writes
    to reduce cost and increase efficiency. (SCALING SOLUTION: BATCH WRITES)
    """
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
            print(f"Committed {total_saved} documents to Firestore.")
            batch = db.batch() # Start a new batch
            
    # Commit any remaining operations
    if total_saved % 500 != 0:
        batch.commit()

    print(f"Successfully saved a total of {total_saved} new documents via batch writes.")

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
        print(f"FATAL ERROR during content extraction for {url}: {e}")
        return None

# --- 3. AI ANALYSIS FUNCTION (Placeholder for API Offload) ---

def analyze_text(title: str, content: str) -> dict[str, Any]:
    """
    Placeholder for AI analysis. The final implementation will use ASYNC httpx 
    to call the Hugging Face Inference API.
    """
    return {
        "sentiment": {"label": "POSITIVE", "score": 0.99},
        "entities": [{"text": "Elon Musk", "type": "PER"}],
        "keywords_matched": ["AI", "Tech"],
    }

# --- 4. ASYNCHRONOUS DATA FETCHING (SCALING SOLUTION) ---

def generate_hash(url: str) -> str:
    """Generates a SHA-256 hash of the URL for unique identification."""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

async def fetch_gnews_articles() -> list[dict]:
    """Asynchronously fetches articles for configured keywords using the GNews API."""
    gnews_api_key = os.environ.get("GNEWS_API_KEY", "YOUR_PLACEHOLDER_KEY") 
    config = get_scraper_config()
    base_url = "https://gnews.io/api/v4/search"
    all_articles = []
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        tasks = []
        for keyword in config['keywords']:
            params = {"q": keyword, "apikey": gnews_api_key, "lang": "en", "max": 10}
            tasks.append(client.get(base_url, params=params))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, Exception):
                print(f"FATAL ERROR during GNews API call: {response}")
                continue
            if response.status_code == 200:
                try:
                    data = response.json()
                    articles = data.get('articles', [])
                    for article in articles:
                        article['source_type'] = 'GNEWS'
                        article['url_hash'] = generate_hash(article['url'])
                    all_articles.extend(articles)
                except Exception as e:
                    print(f"ERROR: Failed to parse GNews response: {e}")
            else:
                print(f"ERROR: GNews API failed with status {response.status_code} for URL: {response.url}")
                
    return all_articles

async def fetch_rss_articles() -> list[dict]:
    """Placeholder for asynchronous RSS fetching."""
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
    print("Metrics aggregated and saved to dashboard_metrics collection.")


# --- 6. MAIN PIPELINE ORCHESTRATION ---

async def main_pipeline():
    """
    The main asynchronous orchestrator for the entire data pipeline.
    """
    initialize_firebase() 
    print("--- Starting Sentinel Dashboard Pipeline ---")
    
    # 1. Deduplication Preparation 
    existing_hashes = get_existing_hashes() 
    
    # 2. Fetch data concurrently
    fetch_tasks = [
        fetch_gnews_articles(),
        fetch_rss_articles(),
    ]
    gnews_data, rss_data = await asyncio.gather(*fetch_tasks)
    all_raw_articles = gnews_data + rss_data
    print(f"Total raw articles fetched: {len(all_raw_articles)}")

    # 3. Deduplication and Filter
    new_articles = [article for article in all_raw_articles if article['url_hash'] not in existing_hashes]
    print(f"Filtered down to {len(new_articles)} new articles for analysis.")

    # 4. Content Extraction, Filtering, and Analysis
    articles_to_save = []
    server_timestamp = firestore.SERVER_TIMESTAMP

    print("Starting content extraction and filtering...")
    
    for article in new_articles:
        if not article.get('url') or not article.get('title'):
            continue
            
        clean_content = fetch_content(article['url'])
        
        if clean_content:
            # Placeholder for AI Analysis call
            analysis = analyze_text(article['title'], clean_content)
            
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

    print(f"Content extraction complete. {len(articles_to_save)} documents ready for saving.")
    
    # 5. Database Storage (Batch Write)
    save_articles_to_firestore(articles_to_save)

    # 6. Server-Side Aggregation (SCALING SOLUTION: FRONTEND DECOUPLING)
    aggregate_metrics(articles_to_save)
    
    print("--- Pipeline Finished Successfully! ---")


# --- 7. ENTRY POINT ---
if __name__ == "__main__":
    if os.environ.get("GNEWS_API_KEY") == "YOUR_PLACEHOLDER_KEY":
        print("\n*** WARNING: GNEWS_API_KEY is using a placeholder value. ***")
        print("*** Please set the actual key in your environment variables for real data. ***\n")

    try:
        asyncio.run(main_pipeline())
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")