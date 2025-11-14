from typing import Optional 
import trafilatura
# ... other imports (like httpx)
# ... rest of your imports
import trafilatura
import httpx 

# Placeholder for the new GNews API fetch and RSS fetch
# This function is what cleans the raw article URL
def fetch_content(url: str) -> Optional[str]:
    """
    Fetches a URL and extracts only the main article text,
    removing boilerplate content (ads, navbars, etc.).
    """
    try:
        # 1. Fetch the raw HTML content
        downloaded = trafilatura.fetch_url(url)

        if downloaded is None:
            # Handle cases where the URL cannot be reached or is blocked
            print(f"ERROR: Could not fetch content for URL: {url}")
            return None

        # 2. Extract the main body text
        # We favor precision to get cleaner article text
        content = trafilatura.extract(downloaded, favor_precision=True)

        # 3. Handle extraction failure (e.g., paywalls)
        if content is None or len(content) < 100:
            print(f"WARNING: Extraction failed or content too short for URL: {url}")
            return None

        return content

    except Exception as e:
        print(f"FATAL ERROR during content extraction for {url}: {e}")
        return None

# Example test (You can temporarily uncomment this to test later)
# if __name__ == "__main__":
#     test_url = "https://www.independent.co.uk/news/world/russia-ukraine-war-live-latest-b2447958.html"
#     article_text = fetch_content(test_url)
#     if article_text:
#         print("--- Extracted Article Content (First 500 chars) ---")
#         print(article_text[:500])
#         print("-----------------------------------------------------")