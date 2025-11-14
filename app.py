import streamlit as st
import pandas as pd
from firebase_admin import credentials, initialize_app
from google.cloud import firestore
import firebase_admin 
from google.oauth2 import service_account


# --- 1. FIREBASE INITIALIZATION & CLIENT (FINAL SCALABLE FIX) ---

@st.cache_resource # SCALING SOLUTION: Ensures connection runs only once per deployment
def initialize_firebase_client():
    """Initializes the Admin SDK and passes the loaded credentials to the Firestore Client."""
    
    # 1. Read secrets into a structured dictionary for the Admin SDK
    try:
        service_account_info = {
            "type": st.secrets["firebase_backend"]["type"],
            "project_id": st.secrets["firebase_backend"]["project_id"],
            "private_key_id": st.secrets["firebase_backend"]["private_key_id"],
            "private_key": st.secrets["firebase_backend"]["private_key"], 
            "client_email": st.secrets["firebase_backend"]["client_email"],
            "client_id": st.secrets["firebase_backend"]["client_id"],
            "auth_uri": st.secrets["firebase_backend"]["auth_uri"],
            "token_uri": st.secrets["firebase_backend"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["firebase_backend"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["firebase_backend"]["client_x509_cert_url"],
        }
    except KeyError as ke:
         st.error(f"🚨 Config Error: Missing key in secrets.toml: {ke}. Check [firebase_backend].")
         return None
    
    # 2. Initialize the Admin SDK if not already initialized
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(service_account_info)
            initialize_app(cred)
            st.sidebar.success("✅ Database connection established.")
        except Exception as e:
             st.error(f"🚨 FIREBASE CONNECTION FAILED: {e}. Check 'private_key' format in secrets.toml!")
             return None
             
    # 3. FIX: Create credentials object and pass it to firestore.Client().
    # This prevents the DefaultCredentialsError by forcing the authentication context.
    
    try:
        # Create GoogleCredentials object from the dictionary
        google_creds = service_account.Credentials.from_service_account_info(
            service_account_info
        )

        # Pass the explicit credentials to the Firestore Client
        return firestore.Client(
            project=st.secrets["firebase_backend"]["project_id"],
            credentials=google_creds 
        )
    except Exception as e:
        st.error(f"🚨 Firestore Client Error: {e}")
        return None

# --- 2. DATA RETRIEVAL & CACHING ---

@st.cache_data(ttl=600) # SCALING SOLUTION: Cache the data for 10 minutes
def fetch_dashboard_metrics(_db_client):
    """
    Retrieves the small, pre-computed dashboard metrics document 
    (Frontend Decoupling Solution).
    Note: _db_client parameter name starts with underscore to prevent Streamlit from hashing it.
    """
    if _db_client:
        try:
            # SCALING SOLUTION: We only read the small metrics document, not 100k articles
            doc_ref = _db_client.collection('dashboard_metrics').document('latest_metrics')
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            else:
                st.warning("⚠️ Backend pipeline has not run yet. No metrics found.")
                return {"total_new_articles": 0, "status": "Not run"}
        except Exception as e:
            st.error(f"❌ Error fetching metrics: {e}")
            return None
    return None

@st.cache_data(ttl=3600) # Configuration changes less frequently, cache for 1 hour
def fetch_scraper_config(_db_client):
    """
    Retrieves the scraper configuration document from Firestore.
    Note: _db_client parameter name starts with underscore to prevent Streamlit from hashing it.
    """
    if _db_client:
        try:
            doc_ref = _db_client.collection('config').document('scraper_settings')
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            # If config is missing, return a structure that prevents crashes
            else:
                st.info("ℹ️ No configuration found. Creating default structure...")
                return {"keywords": [], "rss_feeds": [], "subreddits": []} 
        except Exception as e:
            st.error(f"❌ Error fetching config: {e}")
            return None
    return None

# --- 3. PAGE LOGIC ---

def show_main_dashboard(db_client):
    """Renders the main trend analysis dashboard."""
    st.title("🛡️ Sentinel Trend Dashboard")
    st.markdown("A real-time geopolitical trend analysis platform.")

    metrics = fetch_dashboard_metrics(db_client)

    if metrics:
        st.markdown("### 📊 Pipeline Overview")
        
        col1, col2 = st.columns(2)
        col1.metric("Total Articles Saved (Last Run)", metrics.get("total_new_articles", 0))
        col2.metric("Pipeline Status", "✅ Operational (Hourly)")
        
        st.markdown("---")

        st.markdown("### 🔑 Live Keyword Trends")
        
        if metrics.get("keyword_counts"):
            trend_df = pd.DataFrame(
                metrics["keyword_counts"].items(), 
                columns=["Keyword", "Count"]
            ).sort_values(by="Count", ascending=False)
            
            # Use native Streamlit dataframe with better styling
            st.dataframe(
                trend_df, 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("ℹ️ No keyword data generated yet. Please run the backend pipeline.")
    else:
        st.warning("⚠️ Unable to load dashboard metrics. Check Firebase connection.")
    
    # Placeholder for advanced visualizations (Mind Map, Geo Map)
    st.markdown("---")
    st.subheader("🎯 Advanced Visualization Placeholders")
    st.info("The interactive Mind Map and Geo Map visualizations (Phase 4.3) will be built here.")


def show_settings_page(db_client):
    """Renders the page allowing users to update keywords and feeds."""
    st.title("⚙️ Dynamic Scraper Configuration")
    st.markdown("Update the arrays below to instantly change what the hourly backend pipeline scrapes.")

    config_data = fetch_scraper_config(db_client)

    if config_data:
        st.info("ℹ️ Current settings are loaded from Firestore. Edit and save to update the backend's behavior.")
        
        # --- Form for Saving Settings ---
        with st.form("config_form"):
            st.subheader("1. 🔍 Scraper Keywords")
            
            keywords_text = st.text_area(
                "Keywords (One per line)", 
                value="\n".join(config_data.get('keywords', [])),
                height=150,
                help="Enter keywords to track in news articles and RSS feeds"
            )
            
            st.subheader("2. 📡 RSS Feeds")
            rss_feeds_text = st.text_area(
                "RSS Feeds (One URL per line)",
                value="\n".join(config_data.get('rss_feeds', [])),
                height=150,
                help="Enter full RSS feed URLs to scrape"
            )

            st.subheader("3. 💬 Subreddits (Placeholder)")
            subreddits_text = st.text_area(
                "Subreddits (One per line)",
                value="\n".join(config_data.get('subreddits', [])),
                height=150,
                help="This controls the Reddit scraper if implemented (Phase 6 backlog)."
            )
            
            submitted = st.form_submit_button("💾 Save Configuration to Firestore", type="primary")
            
            if submitted:
                # Process the text areas back into lists
                new_keywords = [k.strip() for k in keywords_text.split('\n') if k.strip()]
                new_rss_feeds = [r.strip() for r in rss_feeds_text.split('\n') if r.strip()]
                new_subreddits = [s.strip() for s in subreddits_text.split('\n') if s.strip()]

                # Update Firestore
                try:
                    db_client.collection('config').document('scraper_settings').set({
                        'keywords': new_keywords,
                        'rss_feeds': new_rss_feeds,
                        'subreddits': new_subreddits,
                        'last_updated': firestore.SERVER_TIMESTAMP
                    }, merge=True)
                    st.success("✅ Configuration saved successfully! The hourly pipeline will use these settings next.")
                    # Clear cache to force a reload on the next run
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to save configuration: {e}")
    else:
        st.error("❌ Unable to load configuration. Check Firebase connection.")


# --- 4. MAIN APP ENTRY POINT ---

def main():
    st.set_page_config(
        layout="wide", 
        page_title="Sentinel Dashboard", 
        page_icon="🛡️",
        initial_sidebar_state="expanded"
    )

    # 1. Initialize Firestore 
    db = initialize_firebase_client()
    
    if db is None:
        st.error("🚨 Failed to initialize Firebase. Please check your secrets configuration.")
        st.stop() # Stop execution if Firebase initialization failed

    # 2. Sidebar Navigation
    st.sidebar.title("🧭 App Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Settings"])
    
    # 3. Page Rendering
    if page == "Dashboard":
        show_main_dashboard(db)
    elif page == "Settings":
        show_settings_page(db)

    # 4. Footer/Debug Info
    st.sidebar.markdown("---")
    st.sidebar.caption("⏱️ Pipeline: Hourly GitHub Action")
    st.sidebar.caption(f"🗄️ Project: {st.secrets['firebase_backend']['project_id']}")

if __name__ == "__main__":
    main()