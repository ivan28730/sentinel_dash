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
import feedparser
import textwrap

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
        
    # Fallback to default configuration with 100+ RSS feeds
    default_config = {
        "keywords": ["geopolitics", "trade war", "AI regulation", "international relations", "diplomacy"], 
        "rss_feeds": [
            # === INTERNATIONAL NEWS (Major Networks) ===
            "http://rss.cnn.com/rss/cnn_topstories.rss",
            "http://rss.cnn.com/rss/cnn_world.rss",
            "http://rss.cnn.com/rss/cnn_us.rss",
            "http://feeds.bbci.co.uk/news/world/rss.xml",
            "http://feeds.bbci.co.uk/news/world/africa/rss.xml",
            "http://feeds.bbci.co.uk/news/world/asia/rss.xml",
            "http://feeds.bbci.co.uk/news/world/europe/rss.xml",
            "http://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
            "http://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
            "https://www.theguardian.com/world/rss",
            "https://www.theguardian.com/uk/rss",
            "https://www.theguardian.com/us-news/rss",
            "https://www.aljazeera.com/xml/rss/all.xml",
            "https://www.dw.com/en/top-stories/s-9097/rss",
            
            # === US NEWS ===
            "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
            "https://feeds.washingtonpost.com/rss/world",
            "https://feeds.washingtonpost.com/rss/politics",
            "https://abcnews.go.com/abcnews/topstories",
            "https://abcnews.go.com/abcnews/internationalheadlines",
            "http://feeds.foxnews.com/foxnews/world",
            "http://feeds.foxnews.com/foxnews/politics",
            "https://www.cbsnews.com/latest/rss/world",
            "https://www.nbcnews.com/id/3032091/device/rss/rss.xml",
            
            # === BUSINESS & FINANCE ===
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",
            "https://www.cnbc.com/id/10001147/device/rss/rss.html",  # CNBC World News
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.ft.com/world?format=rss",
            "https://www.wsj.com/xml/rss/3_7085.xml",
            "https://www.reuters.com/rssFeed/worldNews",
            "https://www.marketwatch.com/rss/topstories",
            
            # === TECH NEWS ===
            "https://techcrunch.com/feed/",
            "https://www.theverge.com/rss/index.xml",
            "https://www.wired.com/feed/rss",
            "https://arstechnica.com/feed/",
            "https://www.zdnet.com/news/rss.xml",
            "https://www.cnet.com/rss/news/",
            "https://www.engadget.com/rss.xml",
            
            # === GEOPOLITICS & POLICY ===
            "https://foreignpolicy.com/feed/",
            "https://www.cfr.org/content/newsletters/rss.xml",
            "https://carnegieendowment.org/rss",
            "https://www.brookings.edu/feed/",
            "https://www.rand.org/blog.rss",
            "https://www.chathamhouse.org/rss.xml",
            "https://www.fpri.org/feed/",
            
            # === EUROPEAN NEWS ===
            "https://www.euronews.com/rss",
            "https://www.france24.com/en/rss",
            "https://www.dw.com/en/rss",
            "https://www.thelocal.com/feed",
            "https://www.politico.eu/feed/",
            
            # === ASIAN NEWS ===
            "https://www.scmp.com/rss/91/feed",  # South China Morning Post
            "https://www.straitstimes.com/news/world/rss.xml",
            "https://www.japantimes.co.jp/feed/",
            "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
            "https://www.thehindu.com/news/international/?service=rss",
            
            # === MIDDLE EAST NEWS ===
            "https://english.alarabiya.net/rss.xml",
            "https://www.haaretz.com/cmlink/1.628752",
            "https://www.jpost.com/Rss/RssFeedsHeadlines.aspx",
            "https://www.timesofisrael.com/feed/",
            
            # === LATIN AMERICA ===
            "https://www.batimes.com.ar/feed",
            "https://www.bnamericas.com/en/rss/all",
            
            # === AFRICA ===
            "https://www.africanews.com/feed/",
            "https://allafrica.com/tools/headlines/rdf/latest/headlines.rdf",
            
            # === SCIENCE & ENVIRONMENT ===
            "https://www.sciencedaily.com/rss/all.xml",
            "https://www.nature.com/nature.rss",
            "https://www.newscientist.com/feed/home",
            "https://www.nationalgeographic.com/feeds/destinations/",
            
            # === DEFENSE & SECURITY ===
            "https://www.defensenews.com/arc/outboundfeeds/rss/",
            "https://www.janes.com/feeds/news",
            "https://breakingdefense.com/feed/",
            
            # === HUMANITARIAN & DEVELOPMENT ===
            "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
            "https://www.devex.com/rss",
            "https://www.oxfam.org/en/rss.xml",
            
            # === ADDITIONAL QUALITY SOURCES ===
            "https://www.economist.com/the-world-this-week/rss.xml",
            "https://www.theatlantic.com/feed/all/",
            "https://www.newyorker.com/feed/news",
            "https://www.vox.com/rss/index.xml",
            "https://www.axios.com/feeds/feed.rss",
            "https://www.politico.com/rss/politics08.xml",
            "https://thehill.com/feed",
            "https://www.salon.com/feed/",
            "https://slate.com/feeds/all.rss",
            "https://www.thedailybeast.com/rss",
            "https://qz.com/feed/",
            "https://www.huffpost.com/section/world-news/feed",
        ],
        "subreddits": ["worldnews", "geopolitics"]
    }
    print(f"ℹ️ Using default configuration with {len(default_config['rss_feeds'])} RSS feeds")
    return default_config

# API Keys
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "mistralai/Mixtral-8x7B-Instruct")
HF_API_URL = os.getenv("HF_API_URL", f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}")
HF_TIMEOUT = float(os.getenv("HF_API_TIMEOUT", "60"))
LLM_ANALYSIS_LIMIT = int(os.getenv("LLM_ANALYSIS_LIMIT", "80"))
LLM_CALLS_MADE = 0

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


def _build_llm_prompt(title: str, content: str, matched_keywords: list[str]) -> str:
    snippet = textwrap.shorten(content, width=4800, placeholder="")
    tracked = ", ".join(matched_keywords) if matched_keywords else "None"
    prompt = f"""
You are Sentinel Analyst, an AI that writes concise geopolitical intelligence briefs.
Summarize the article below and output ONLY valid JSON with keys:
summary (<=3 sentences),
sentiment (object with label in [POSITIVE, NEGATIVE, NEUTRAL] and score 0-1),
entities (list of objects {{name, type}}),
focus_country (string or null),
risk_level (Low/Moderate/Elevated/Critical),
key_points (list of short bullet strings),
keywords (list of tracked keywords referenced).

Title: {title}
Tracked Keywords Mentioned: {tracked}
Article Body:
---
{snippet}
---

Return valid JSON only.
"""
    return textwrap.dedent(prompt).strip()


def _extract_json_block(text: str) -> Optional[dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def run_llm_analysis(title: str, content: str, matched_keywords: list[str]) -> Optional[dict[str, Any]]:
    global LLM_CALLS_MADE
    if not HF_API_KEY:
        return None
    if LLM_ANALYSIS_LIMIT > 0 and LLM_CALLS_MADE >= LLM_ANALYSIS_LIMIT:
        return None
    prompt = _build_llm_prompt(title, content, matched_keywords)
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 512, "temperature": 0.2},
        "options": {"wait_for_model": True}
    }
    try:
        response = httpx.post(HF_API_URL, headers=headers, json=payload, timeout=HF_TIMEOUT)
        LLM_CALLS_MADE += 1
        if response.status_code != 200:
            print(f"⚠️ LLM request failed ({response.status_code}): {response.text[:120]}")
            return None
        data = response.json()
        generated_text = ""
        if isinstance(data, list) and data:
            generated_text = data[0].get("generated_text", "")
        elif isinstance(data, dict) and data.get("generated_text"):
            generated_text = data.get("generated_text", "")
        if generated_text:
            parsed = _extract_json_block(generated_text)
            if parsed:
                return parsed
        if isinstance(data, dict):
            if data.get("error"):
                print(f"⚠️ LLM error: {data['error']}")
                return None
            return data
    except Exception as e:
        print(f"⚠️ LLM analysis error: {e}")
    return None

def analyze_text(title: str, content: str, keywords: list[str]) -> dict[str, Any]:
    """
    Placeholder for AI analysis. The final implementation will use ASYNC httpx 
    to call the Hugging Face Inference API.
    """
    matched_keywords = [kw for kw in keywords if kw.lower() in content.lower() or kw.lower() in title.lower()]
    base_result = {
        "sentiment": {"label": "NEUTRAL", "score": 0.5},
        "entities": [],
        "keywords_matched": matched_keywords,
        "summary": textwrap.shorten(content, width=320, placeholder="..."),
        "focus_country": None,
        "risk_level": "Low",
        "key_points": [],
        "analysis_model": "heuristic"
    }

    llm_payload = run_llm_analysis(title, content, matched_keywords)
    if llm_payload:
        sentiment = llm_payload.get("sentiment", base_result["sentiment"])
        if isinstance(sentiment, dict) and sentiment.get("label"):
            sentiment["label"] = sentiment["label"].upper()
        entities = llm_payload.get("entities", base_result["entities"])
        if isinstance(entities, dict):
            entities = [entities]
        key_points = llm_payload.get("key_points", base_result["key_points"])
        if isinstance(key_points, str):
            key_points = [key_points]
        base_result.update({
            "summary": llm_payload.get("summary") or base_result["summary"],
            "sentiment": sentiment,
            "entities": entities,
            "focus_country": llm_payload.get("focus_country") or llm_payload.get("primary_country"),
            "risk_level": llm_payload.get("risk_level", base_result["risk_level"]),
            "key_points": key_points,
            "analysis_model": llm_payload.get("model", HF_MODEL_ID),
        })
        if llm_payload.get("keywords"):
            merged = set(base_result["keywords_matched"]) | {kw.strip() for kw in llm_payload["keywords"] if kw}
            base_result["keywords_matched"] = sorted(merged)
    return base_result

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
                    print(f"✅ GNews: Fetched {len(articles)} articles for '{keywords[i]}'")
                except Exception as e:
                    print(f"❌ Failed to parse GNews response: {e}")
            else:
                print(f"❌ GNews API returned status {response.status_code}")
                
    return all_articles


async def fetch_newsapi_articles(keywords: list[str]) -> list[dict]:
    """Asynchronously fetches articles for configured keywords using NewsAPI."""
    newsapi_key = os.environ.get("NEWSAPI_KEY", "")
    
    if not newsapi_key:
        print("⚠️ WARNING: NEWSAPI_KEY not set. Skipping NewsAPI scraping.")
        return []
    
    base_url = "https://newsapi.org/v2/everything"
    all_articles = []
    
    async with httpx.AsyncClient(timeout=20.0) as client:
        tasks = []
        for keyword in keywords:
            params = {
                "q": keyword,
                "apiKey": newsapi_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10
            }
            tasks.append(client.get(base_url, params=params))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"❌ NewsAPI error for '{keywords[i]}': {response}")
                continue
            if response.status_code == 200:
                try:
                    data = response.json()
                    articles = data.get('articles', [])
                    for article in articles:
                        # Convert NewsAPI format to match our structure
                        formatted_article = {
                            'url': article['url'],
                            'title': article['title'],
                            'description': article.get('description', ''),
                            'publishedAt': article.get('publishedAt', ''),
                            'source_type': 'NEWSAPI',
                            'url_hash': generate_hash(article['url'])
                        }
                        all_articles.append(formatted_article)
                    print(f"✅ NewsAPI: Fetched {len(articles)} articles for '{keywords[i]}'")
                except Exception as e:
                    print(f"❌ Failed to parse NewsAPI response: {e}")
            else:
                print(f"❌ NewsAPI returned status {response.status_code}")
                
    return all_articles


async def fetch_rss_articles(rss_feeds: list[str]) -> list[dict]:
    """Asynchronously fetches articles from RSS feeds."""
    all_articles = []
    
    print(f"📡 Starting RSS fetch from {len(rss_feeds)} feeds...")
    
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        # Process feeds in batches to avoid overwhelming the system
        batch_size = 20
        for i in range(0, len(rss_feeds), batch_size):
            batch = rss_feeds[i:i + batch_size]
            tasks = [client.get(feed_url) for feed_url in batch]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for j, response in enumerate(responses):
                feed_url = batch[j]
                
                if isinstance(response, Exception):
                    print(f"❌ RSS error for '{feed_url}': {response}")
                    continue
                    
                if response.status_code == 200:
                    try:
                        # Parse the RSS feed
                        feed = feedparser.parse(response.text)
                        
                        # Extract articles from feed entries
                        for entry in feed.entries:
                            # Get the link (URL)
                            url = entry.get('link', '')
                            if not url:
                                continue
                            
                            article = {
                                'url': url,
                                'title': entry.get('title', 'No Title'),
                                'description': entry.get('summary', entry.get('description', '')),
                                'publishedAt': entry.get('published', entry.get('updated', '')),
                                'source_type': 'RSS',
                                'url_hash': generate_hash(url)
                            }
                            all_articles.append(article)
                        
                        print(f"✅ RSS [{i+j+1}/{len(rss_feeds)}]: Fetched {len(feed.entries)} articles from {feed_url[:50]}...")
                        
                    except Exception as e:
                        print(f"❌ Failed to parse RSS feed {feed_url}: {e}")
                else:
                    print(f"❌ RSS feed returned status {response.status_code} for {feed_url}")
            
            # Small delay between batches to be respectful
            if i + batch_size < len(rss_feeds):
                await asyncio.sleep(1)
    
    print(f"✅ RSS: Total fetched {len(all_articles)} articles from {len(rss_feeds)} feeds")
    return all_articles

# --- 5. SERVER-SIDE AGGREGATION (SCALING SOLUTION: FRONTEND DECOUPLING) ---

def aggregate_metrics(articles_list: list[dict]):
    """
    Calculates summary data (metrics) from newly saved articles.
    This prevents the frontend from querying massive datasets.
    """
    metrics = defaultdict(int)
    keyword_counts = defaultdict(int)
    source_counts = defaultdict(int)

    # Example aggregation logic: Count keywords matched and sources
    for article in articles_list:
        # Count by source type
        source_type = article.get('source_type', 'UNKNOWN')
        source_counts[source_type] += 1
        
        # Count keywords matched
        if 'keywords_matched' in article.get('analysis_results', {}):
            for keyword in article['analysis_results']['keywords_matched']:
                keyword_counts[keyword] += 1
    
    # Store aggregated data
    metrics['total_new_articles'] = len(articles_list)
    metrics['keyword_counts'] = dict(keyword_counts)
    metrics['source_counts'] = dict(source_counts)
    
    # Save to the dashboard_metrics collection
    db = firestore.client()
    db.collection('dashboard_metrics').document('latest_metrics').set(metrics, merge=True)
    print(f"✅ Metrics aggregated and saved: {len(articles_list)} new articles")
    print(f"   📊 By source: {dict(source_counts)}")
    print(f"   🔑 Unique keywords: {len(keyword_counts)}")


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
        fetch_newsapi_articles(keywords),
        fetch_rss_articles(rss_feeds),
    ]
    gnews_data, newsapi_data, rss_data = await asyncio.gather(*fetch_tasks)
    all_raw_articles = gnews_data + newsapi_data + rss_data
    print(f"\n📊 Total raw articles fetched: {len(all_raw_articles)}")
    print(f"   - GNews: {len(gnews_data)}")
    print(f"   - NewsAPI: {len(newsapi_data)}")
    print(f"   - RSS: {len(rss_data)}")

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
            
            # Show progress every 100 articles
            if i % 100 == 0:
                print(f"  Progress: [{i}/{len(new_articles)}] articles processed...")
            
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
