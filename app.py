import streamlit as st
import pandas as pd
from firebase_admin import credentials, initialize_app
from google.cloud import firestore
import firebase_admin 
from google.oauth2 import service_account
import requests
import time
from datetime import datetime


# --- 1. FIREBASE INITIALIZATION & CLIENT ---

@st.cache_resource
def initialize_firebase_client():
    """Initializes the Admin SDK and passes the loaded credentials to the Firestore Client."""
    
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
    
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate(service_account_info)
            initialize_app(cred)
            st.sidebar.success("✅ Database connection established.")
        except Exception as e:
             st.error(f"🚨 FIREBASE CONNECTION FAILED: {e}. Check 'private_key' format in secrets.toml!")
             return None
             
    try:
        google_creds = service_account.Credentials.from_service_account_info(
            service_account_info
        )
        return firestore.Client(
            project=st.secrets["firebase_backend"]["project_id"],
            credentials=google_creds 
        )
    except Exception as e:
        st.error(f"🚨 Firestore Client Error: {e}")
        return None


# --- 2. GITHUB ACTIONS API INTEGRATION ---

def trigger_github_workflow():
    """Triggers the GitHub Actions workflow via API."""
    try:
        url = f"https://api.github.com/repos/{st.secrets['github']['repo_owner']}/{st.secrets['github']['repo_name']}/actions/workflows/{st.secrets['github']['workflow_file']}/dispatches"
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {st.secrets['github']['token']}"
        }
        
        data = {"ref": "main"}
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 204:
            return True, "Workflow triggered successfully!"
        else:
            return False, f"Failed to trigger workflow: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Error triggering workflow: {str(e)}"


def get_latest_workflow_run():
    """Gets the status of the latest workflow run."""
    try:
        url = f"https://api.github.com/repos/{st.secrets['github']['repo_owner']}/{st.secrets['github']['repo_name']}/actions/workflows/{st.secrets['github']['workflow_file']}/runs"
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {st.secrets['github']['token']}"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data['workflow_runs']:
                latest_run = data['workflow_runs'][0]
                return {
                    'status': latest_run['status'],
                    'conclusion': latest_run['conclusion'],
                    'created_at': latest_run['created_at'],
                    'html_url': latest_run['html_url']
                }
        return None
    except Exception as e:
        st.error(f"Error fetching workflow status: {str(e)}")
        return None


# --- 3. DATA MANAGEMENT ---

def clear_all_data(db_client):
    """Clears all articles and resets metrics in Firebase."""
    try:
        # Clear articles collection
        articles_ref = db_client.collection('articles')
        articles = articles_ref.stream()
        deleted_count = 0
        
        for doc in articles:
            doc.reference.delete()
            deleted_count += 1
        
        # Reset metrics
        db_client.collection('dashboard_metrics').document('latest_metrics').set({
            'total_new_articles': 0,
            'keyword_counts': {},
            'last_updated': firestore.SERVER_TIMESTAMP
        })
        
        return True, f"✅ Cleared {deleted_count} articles and reset metrics"
    except Exception as e:
        return False, f"❌ Error clearing data: {str(e)}"


# --- 4. DATA RETRIEVAL & CACHING ---

@st.cache_data(ttl=600)
def fetch_dashboard_metrics(_db_client):
    """Retrieves the small, pre-computed dashboard metrics document."""
    if _db_client:
        try:
            doc_ref = _db_client.collection('dashboard_metrics').document('latest_metrics')
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            else:
                st.warning("⚠️ Backend pipeline has not run yet. No metrics found.")
                return {"total_new_articles": 0, "keyword_counts": {}, "status": "Not run"}
        except Exception as e:
            st.error(f"❌ Error fetching metrics: {e}")
            return None
    return None


@st.cache_data(ttl=3600)
def fetch_scraper_config(_db_client):
    """Retrieves the scraper configuration document from Firestore."""
    if _db_client:
        try:
            doc_ref = _db_client.collection('config').document('scraper_settings')
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            else:
                st.info("ℹ️ No configuration found. Creating default structure...")
                return {"keywords": [], "rss_feeds": [], "subreddits": []} 
        except Exception as e:
            st.error(f"❌ Error fetching config: {e}")
            return None
    return None


# --- 5. PAGE LOGIC ---

def show_main_dashboard(db_client):
    """Renders the main trend analysis dashboard."""
    st.title("🛡️ Sentinel Trend Dashboard")
    st.markdown("A real-time geopolitical trend analysis platform.")

    metrics = fetch_dashboard_metrics(db_client)

    if metrics:
        st.markdown("### 📊 Pipeline Overview")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        col1.metric("Total Articles Saved (Last Run)", metrics.get("total_new_articles", 0))
        col2.metric("Pipeline Status", "✅ Operational (Hourly)")
        
        # Add Clear Data button in dashboard
        with col3:
            if st.button("🗑️ Clear Data", use_container_width=True, help="Delete all articles and reset metrics"):
                with st.spinner("Clearing all data..."):
                    success, message = clear_all_data(db_client)
                    if success:
                        st.success(message)
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
        
        st.markdown("---")

        st.markdown("### 🔑 Live Keyword Trends")
        
        if metrics.get("keyword_counts"):
            trend_df = pd.DataFrame(
                metrics["keyword_counts"].items(), 
                columns=["Keyword", "Count"]
            ).sort_values(by="Count", ascending=False)
            
            st.dataframe(
                trend_df, 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("ℹ️ No keyword data generated yet. Please run the backend pipeline.")
    else:
        st.warning("⚠️ Unable to load dashboard metrics. Check Firebase connection.")
    
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
                new_keywords = [k.strip() for k in keywords_text.split('\n') if k.strip()]
                new_rss_feeds = [r.strip() for r in rss_feeds_text.split('\n') if r.strip()]
                new_subreddits = [s.strip() for s in subreddits_text.split('\n') if s.strip()]

                try:
                    db_client.collection('config').document('scraper_settings').set({
                        'keywords': new_keywords,
                        'rss_feeds': new_rss_feeds,
                        'subreddits': new_subreddits,
                        'last_updated': firestore.SERVER_TIMESTAMP
                    }, merge=True)
                    st.success("✅ Configuration saved successfully! The hourly pipeline will use these settings next.")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to save configuration: {e}")
        
        # --- Manual Scraper Trigger Section ---
        st.markdown("---")
        st.subheader("▶️ Manual Scraper Control")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🚀 Run Scraper Now", type="primary", use_container_width=True):
                with st.spinner("Triggering GitHub Actions workflow..."):
                    success, message = trigger_github_workflow()
                    
                    if success:
                        st.success("✅ " + message)
                        st.info("⏳ The scraper is starting... Check status below.")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ " + message)
        
        with col2:
            if st.button("🔄 Refresh Dashboard", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear All Data", use_container_width=True, help="Delete all articles and reset metrics"):
                with st.spinner("Clearing all data..."):
                    success, message = clear_all_data(db_client)
                    if success:
                        st.success(message)
                        st.cache_data.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
        
        # --- Workflow Status Display ---
        st.markdown("#### 📊 Latest Scraper Run Status")
        
        workflow_status = get_latest_workflow_run()
        
        if workflow_status:
            status = workflow_status['status']
            conclusion = workflow_status['conclusion']
            
            if status == 'completed':
                if conclusion == 'success':
                    st.success(f"✅ Last run completed successfully")
                elif conclusion == 'failure':
                    st.error(f"❌ Last run failed")
                else:
                    st.warning(f"⚠️ Last run completed with status: {conclusion}")
            elif status == 'in_progress':
                st.info("⏳ Scraper is currently running...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f"Running... {i + 1}%")
                    time.sleep(0.1)
                
                st.info("🔄 Workflow still running. Refresh this page to check status.")
            else:
                st.info(f"ℹ️ Status: {status}")
            
            st.markdown(f"[View workflow details on GitHub]({workflow_status['html_url']})")
        else:
            st.info("ℹ️ No recent workflow runs found.")
    
    else:
        st.error("❌ Unable to load configuration. Check Firebase connection.")


# --- 6. MAIN APP ENTRY POINT ---

def main():
    st.set_page_config(
        layout="wide", 
        page_title="Sentinel Dashboard", 
        page_icon="🛡️",
        initial_sidebar_state="expanded"
    )

    db = initialize_firebase_client()
    
    if db is None:
        st.error("🚨 Failed to initialize Firebase. Please check your secrets configuration.")
        st.stop()

    st.sidebar.title("🧭 App Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Settings"])
    
    if page == "Dashboard":
        show_main_dashboard(db)
    elif page == "Settings":
        show_settings_page(db)

    st.sidebar.markdown("---")
    st.sidebar.caption("⏱️ Pipeline: Hourly GitHub Action")
    st.sidebar.caption(f"🗄️ Project: {st.secrets['firebase_backend']['project_id']}")


if __name__ == "__main__":
    main()