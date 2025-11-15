import streamlit as st
import pandas as pd
from firebase_admin import credentials, initialize_app
from google.cloud import firestore
import firebase_admin 
from google.oauth2 import service_account
import requests
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(layout="wide", page_title="Sentinel Dashboard", page_icon="🛡️", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stProgress > div > div > div > div {background: linear-gradient(to right, #667eea, #764ba2);}
    h1 {color: #00D9FF !important;}
    .metric-card {padding: 20px; border-radius: 10px; text-align: center;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_firebase_client():
    try:
        service_account_info = {k: st.secrets["firebase_backend"][k] for k in 
            ["type", "project_id", "private_key_id", "private_key", "client_email", 
             "client_id", "auth_uri", "token_uri", "auth_provider_x509_cert_url", "client_x509_cert_url"]}
    except KeyError as ke:
        st.error(f"🚨 Missing key: {ke}")
        return None
    
    if not firebase_admin._apps:
        try:
            initialize_app(credentials.Certificate(service_account_info))
        except Exception as e:
            st.error(f"🚨 Firebase failed: {e}")
            return None
    
    try:
        return firestore.Client(
            project=st.secrets["firebase_backend"]["project_id"],
            credentials=service_account.Credentials.from_service_account_info(service_account_info)
        )
    except Exception as e:
        st.error(f"🚨 Firestore error: {e}")
        return None


def trigger_github_workflow():
    try:
        url = f"https://api.github.com/repos/{st.secrets['github']['repo_owner']}/{st.secrets['github']['repo_name']}/actions/workflows/{st.secrets['github']['workflow_file']}/dispatches"
        response = requests.post(url, headers={"Accept": "application/vnd.github.v3+json", "Authorization": f"token {st.secrets['github']['token']}"}, json={"ref": "main"})
        return (True, "Workflow triggered!") if response.status_code == 204 else (False, f"Failed: {response.status_code}")
    except Exception as e:
        return False, str(e)


def get_latest_workflow_run():
    try:
        url = f"https://api.github.com/repos/{st.secrets['github']['repo_owner']}/{st.secrets['github']['repo_name']}/actions/workflows/{st.secrets['github']['workflow_file']}/runs"
        response = requests.get(url, headers={"Accept": "application/vnd.github.v3+json", "Authorization": f"token {st.secrets['github']['token']}"})
        if response.status_code == 200:
            data = response.json()
            if data['workflow_runs']:
                run = data['workflow_runs'][0]
                return {'status': run['status'], 'conclusion': run['conclusion'], 'html_url': run['html_url']}
        return None
    except:
        return None


def clear_all_data(db_client):
    try:
        count = sum(1 for doc in db_client.collection('articles').stream() if doc.reference.delete() or True)
        db_client.collection('dashboard_metrics').document('latest_metrics').set({'total_new_articles': 0, 'keyword_counts': {}, 'last_updated': firestore.SERVER_TIMESTAMP})
        return True, f"✅ Cleared {count} articles"
    except Exception as e:
        return False, f"❌ Error: {e}"


@st.cache_data(ttl=600)
def fetch_dashboard_metrics(_db_client):
    if _db_client:
        try:
            doc = _db_client.collection('dashboard_metrics').document('latest_metrics').get()
            return doc.to_dict() if doc.exists else {"total_new_articles": 0, "keyword_counts": {}}
        except:
            return None
    return None


@st.cache_data(ttl=3600)
def fetch_scraper_config(_db_client):
    if _db_client:
        try:
            doc = _db_client.collection('config').document('scraper_settings').get()
            return doc.to_dict() if doc.exists else {"keywords": [], "rss_feeds": [], "subreddits": []}
        except:
            return None
    return None


def show_main_dashboard(db_client):
    st.markdown("<h1 style='text-align: center;'>🛡️ SENTINEL INTELLIGENCE DASHBOARD</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 1.2em;'>Real-time Geopolitical Trend Analysis</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    metrics = fetch_dashboard_metrics(db_client)
    if not metrics:
        st.warning("⚠️ Unable to load metrics")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    total = metrics.get("total_new_articles", 0)
    counts = metrics.get("keyword_counts", {})
    mentions = sum(counts.values())
    topics = len(counts)
    
    with col1:
        st.markdown(f"<div class='metric-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'><h1 style='color: white; margin: 0;'>📰 {total}</h1><p style='color: #E0E0E0;'>Articles</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'><h1 style='color: white; margin: 0;'>🔥 {mentions}</h1><p style='color: #E0E0E0;'>Mentions</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'><h1 style='color: white; margin: 0;'>🌍 {topics}</h1><p style='color: #E0E0E0;'>Topics</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card' style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);'><h1 style='color: white; margin: 0;'>✅</h1><p style='color: #E0E0E0;'>Live</p></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🗑️ Clear All Data", type="secondary"):
        with st.spinner("Clearing..."):
            success, msg = clear_all_data(db_client)
            st.success(msg) if success else st.error(msg)
            if success:
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()
    
    st.markdown("---")
    st.subheader("🔥 Trending Topics")
    
    if counts:
        df = pd.DataFrame(counts.items(), columns=["Topic", "Count"]).sort_values("Count", ascending=False)
        
        fig = px.bar(df, x="Count", y="Topic", orientation='h', color="Count", 
                     color_continuous_scale=["#667eea", "#764ba2", "#f093fb", "#f5576c"])
        fig.update_layout(height=max(400, len(df) * 40), showlegend=False, 
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 Detailed Breakdown")
        for idx, row in df.iterrows():
            progress = min(row['Count'] / df['Count'].max(), 1.0)
            st.markdown(f"**{row['Topic']}**")
            st.progress(progress)
            st.caption(f"{row['Count']} mentions")
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("ℹ️ No data yet. Run the scraper to see trending topics!")
    
    st.markdown("---")
    st.subheader("🎯 Coming Soon")
    col1, col2 = st.columns(2)
    with col1:
        st.info("🗺️ **Geographic Heat Map**\nVisualize global hotspots")
    with col2:
        st.info("🔗 **Network Graph**\nTopic connections & relationships")


def show_settings_page(db_client):
    st.title("⚙️ Configuration & Controls")
    
    config = fetch_scraper_config(db_client)
    if not config:
        st.error("❌ Unable to load configuration")
        return
    
    st.info("💡 Configure what topics to track and sources to scrape")
    
    with st.form("config_form"):
        st.subheader("🔍 Keywords to Track")
        keywords = st.text_area("One keyword per line", value="\n".join(config.get('keywords', [])), height=150, 
                               help="These keywords will be searched across news sources")
        
        st.subheader("📡 RSS Feed URLs")
        feeds = st.text_area("One URL per line", value="\n".join(config.get('rss_feeds', [])), height=150,
                            help="Full RSS feed URLs to scrape")
        
        if st.form_submit_button("💾 Save Configuration", type="primary", use_container_width=True):
            try:
                db_client.collection('config').document('scraper_settings').set({
                    'keywords': [k.strip() for k in keywords.split('\n') if k.strip()],
                    'rss_feeds': [f.strip() for f in feeds.split('\n') if f.strip()],
                    'last_updated': firestore.SERVER_TIMESTAMP
                }, merge=True)
                st.success("✅ Configuration saved successfully!")
                st.cache_data.clear()
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save: {e}")
    
    st.markdown("---")
    st.subheader("▶️ Manual Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Run Scraper Now", type="primary", use_container_width=True, help="Trigger GitHub Actions workflow"):
            with st.spinner("Triggering scraper..."):
                success, msg = trigger_github_workflow()
                if success:
                    st.success("✅ " + msg)
                    st.info("⏳ Scraper starting... Check status below")
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ " + msg)
    
    with col2:
        if st.button("🔄 Refresh Data", use_container_width=True, help="Clear cache and reload"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear All Data", use_container_width=True, help="Delete all articles and reset metrics"):
            with st.spinner("Clearing all data..."):
                success, msg = clear_all_data(db_client)
                if success:
                    st.success(msg)
                    st.cache_data.clear()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(msg)
    
    st.markdown("---")
    st.subheader("📊 Latest Scraper Run Status")
    
    status = get_latest_workflow_run()
    if status:
        if status['status'] == 'completed':
            if status['conclusion'] == 'success':
                st.success("✅ Last run completed successfully")
            elif status['conclusion'] == 'failure':
                st.error("❌ Last run failed")
            else:
                st.warning(f"⚠️ Status: {status['conclusion']}")
        elif status['status'] == 'in_progress':
            st.info("⏳ Scraper is currently running...")
            st.progress(0.5)
        else:
            st.info(f"ℹ️ Status: {status['status']}")
        
        st.markdown(f"[View details on GitHub →]({status['html_url']})")
    else:
        st.info("ℹ️ No recent workflow runs found")


def main():
    db = initialize_firebase_client()
    
    if db is None:
        st.error("🚨 Failed to initialize Firebase")
        st.stop()
    
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio("", ["Dashboard", "Settings"], label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("⏱️ Auto-updates every hour")
    st.sidebar.caption(f"🗄️ Project: {st.secrets['firebase_backend']['project_id']}")
    st.sidebar.caption("Made with ❤️ by Sentinel AI")
    
    if page == "Dashboard":
        show_main_dashboard(db)
    else:
        show_settings_page(db)


if __name__ == "__main__":
    main()