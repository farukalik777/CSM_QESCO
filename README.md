# NEPRA Consumer Service Manual (CSM) Assistant — QESCO
### AI for Everyone — Batch 05 | Project 01 | Groq LLaMA RAG

A **Retrieval-Augmented Generation (RAG)** chatbot for NEPRA Consumer Service Manual (CSM NOV-2025), built for **QESCO Balochistan**.

> ⚡ Ask about New Connections · Billing · Metering · Detection · Complaints · Net Metering · EV Charging

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Groq LLaMA 3.1 8B Instant |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (local) |
| PDF Parsing | pypdf |

## Project Structure

```
CSM-QESCO/
├── app_nepra.py              # Main Streamlit app
├── requirements.txt           # Dependencies
├── .streamlit/
│   ├── config.toml            # Streamlit configuration
│   └── secrets.toml           # API keys (NOT committed)
├── pdfs/
│   └── CONSUMER SERVICE MANUAL (CSM) REVISED 2025.pdf
└── vectorstore_index/         # Auto-generated FAISS index
```

---

## Local Setup

### 1 — Clone & Virtual Environment
```bash
git clone <repo-url>
cd CSM-QESCO
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### 3 — Configure API Key
Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```
Get your key at → https://console.groq.com/keys

### 4 — Run
```bash
streamlit run app_nepra.py
```
> First run: FAISS index is built from the PDF automatically (~1-2 min).

---

## Online Deployment — Streamlit Community Cloud

### Prerequisites
- GitHub repo with all project files (except `vectorstore_index/` and `.streamlit/secrets.toml`)
- Groq API key

### Steps
1. **Push to GitHub** including `app_nepra.py`, `requirements.txt`, and the `pdfs/` folder.
2. Go to [share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub.
3. **Deploy** → select your repo and branch.
4. Under **Secrets**, add:
   ```
   GROQ_API_KEY = gsk_your_key_here
   ```
5. Deploy. The app will build the FAISS index on first cold start.

### Settings for Deployment
```
Branch: main
Main file path: app_nepra.py
Python version: 3.11+
```

---

## Adding / Updating PDFs

1. Replace or add PDF files in the `pdfs/` folder.
2. **Local**: delete `vectorstore_index/` folder and restart the app.
3. **Online**: push new PDFs to GitHub → trigger a re-deploy from Streamlit Cloud.

---

## RAG Pipeline

```
PDF → pypdf text extraction → 400-char sliding window chunks
                                        ↓
User Query → Sentence Transformer embedding → FAISS top-8 search
                                        ↓
Groq LLaMA 3.1 8B Instant → Structured Answer (SUMMARY / DETAILS / REFERENCE)
```

---

*NEPRA CSM QESCO Assistant — AI for Everyone Batch-05*
