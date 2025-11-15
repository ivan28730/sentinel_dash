# 📊 SENTINEL TREND DASHBOARD

## Overview

The Sentinel Trend Dashboard is a professional-grade, scalable data platform designed for real-time geopolitical intelligence and trend analysis. It integrates various data sources and utilizes advanced AI for sentiment and entity recognition, presenting insights through an interactive web dashboard.

---

## 🧱 Core Architecture

The project is structured into distinct, asynchronous services:

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Dashboard & Hosting** | Streamlit Community Cloud | Interactive visualization and public access. |
| **Database** | Firebase Firestore | Highly scalable, zero-cost data storage. |
| **Data Pipeline** | Python (Async) / GitHub Actions | Scheduled scraping, analysis, and data aggregation. |
| **AI Analysis** | Hugging Face Inference API | Offloading heavy-duty Sentiment and NER computation. |

---

## 🚀 Quickstart & Setup

Follow these steps to set up the project locally for development.

### 1. Clone the Repository & Setup Environment

1.  Clone this repository:
    ```bash
    git clone [YOUR_REPO_URL]
    cd sentinel_dash
    ```
2.  Create and activate the Python virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # (Linux/macOS)
    # OR: .\venv\Scripts\Activate.ps1 (Windows PowerShell)
    ```

### 2. Install Dependencies

With the environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure AI Analysis (Optional but recommended)

Set the Hugging Face Inference credentials so the scraper can call the LLM for summaries and sentiment:

```bash
export HF_API_KEY="hf_your_key"
# Optional overrides
# export HF_MODEL_ID="mistralai/Mixtral-8x7B-Instruct"
# export LLM_ANALYSIS_LIMIT=60  # Cap requests per run
```
