"""
NEPRA Consumer Service Manual (CSM) NOV-2025 — QESCO
RAG Chatbot — Groq LLaMA + FAISS + Sentence Transformers
AI for Everyone — Batch 05 | Project 01

Programmer: Faruk Ali Khan
AI Engineer: Faruk Ali Khan
"""

import os, pickle, re
from pathlib import Path
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

# ── CONFIG ────────────────────────────────────
PDF_FOLDER       = "pdfs"
VECTORSTORE_PATH = "vectorstore_index"
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
PAGE_OFFSET      = 4

@st.cache_resource(show_spinner=False)
def get_groq_client():
    return Groq(api_key=st.secrets.get("GROQ_API_KEY", ""))

groq_client = get_groq_client()

st.set_page_config(page_title="NEPRA CSM Assistant — QESCO", page_icon="⚡", layout="centered")

# ── PASSWORD PROTECTION ─────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <style>
    .login-container{max-width:400px;margin:80px auto;padding:30px;
        background:linear-gradient(135deg,#001433,#002266);border:2px solid #0055dd;
        border-radius:14px;box-shadow:0 8px 32px rgba(0,80,220,0.4);text-align:center;}
    .login-title{color:#fff;font-size:1.4rem;font-weight:800;margin-bottom:20px;}
    .login-subtitle{color:#55aaff;font-size:0.8rem;margin-bottom:25px;}
    </style>
    <div class="login-container">
        <div class="login-title">⚡ NEPRA CSM Assistant</div>
        <div class="login-subtitle">QESCO Balochistan — Restricted Access</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=True):
        password = st.text_input("Enter Password", type="password")
        submitted = st.form_submit_button("Access")
        if submitted and password:
            if password == st.secrets.get("APP_PASSWORD", "qesco2025"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
    st.stop()

# ── MAIN APP STARTS HERE ──────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
*{font-family:'Inter',sans-serif;box-sizing:border-box;}
.block-container{padding-top:0.4rem!important;padding-bottom:0.3rem!important;max-width:900px!important;}
.stApp{background:#04090f;}

/* ── HEADER ── */
.hdr{display:flex;align-items:center;justify-content:space-between;
    background:linear-gradient(135deg,#001433,#002266,#003d99,#001433);
    border:2px solid #0055dd;border-radius:14px;padding:10px 16px;margin-bottom:8px;
    box-shadow:0 4px 24px rgba(0,80,220,0.45);}
.logo-nepra{background:linear-gradient(135deg,#002d99,#0050cc);border:2px solid #3388ff;
    border-radius:10px;padding:6px 10px;text-align:center;min-width:88px;}
.logo-nepra .la{color:#aaccff;font-size:0.42rem;font-weight:800;letter-spacing:2px;}
.logo-nepra .lb{color:#fff;font-size:1.2rem;font-weight:900;line-height:1;letter-spacing:2px;text-shadow:0 0 12px #44aaff;}
.logo-nepra .lc{color:#66aaff;font-size:0.36rem;line-height:1.4;}
.logo-qesco{background:linear-gradient(135deg,#002800,#005500);border:2px solid #00bb44;
    border-radius:10px;padding:6px 10px;text-align:center;min-width:88px;}
.logo-qesco .la{color:#88ffaa;font-size:0.42rem;font-weight:800;letter-spacing:2px;}
.logo-qesco .lb{color:#fff;font-size:1.2rem;font-weight:900;line-height:1;letter-spacing:2px;text-shadow:0 0 12px #22ff66;}
.logo-qesco .lc{color:#33ff77;font-size:0.36rem;line-height:1.4;}
.hdr-mid{flex:1;text-align:center;padding:0 10px;}
.hdr-mid h2{color:#fff;font-size:0.9rem;font-weight:800;margin:0 0 2px;line-height:1.25;}
.hdr-mid p{color:#55aaff;font-size:0.58rem;margin:0;}
.hdr-mid .badges{display:flex;gap:4px;justify-content:center;margin-top:4px;flex-wrap:wrap;}
.bd-b{background:rgba(0,70,180,0.45);border:1px solid #0055dd;border-radius:20px;padding:1px 7px;font-size:0.58rem;color:#66aaff;display:inline-block;}
.bd-g{background:rgba(0,130,40,0.35);border:1px solid #00aa33;border-radius:20px;padding:1px 7px;font-size:0.58rem;color:#33ee66;display:inline-block;}

/* ── STATUS ── */
.sbar{background:linear-gradient(90deg,#001628,#002244);border:1px solid #004acc;
    border-radius:7px;padding:5px 12px;margin-bottom:7px;font-size:0.68rem;color:#55aaff;}
.sbar b{color:#00ff88;}

/* ── CHAT AREA ── */
.chat-wrap{max-height:65vh;overflow-y:auto;padding-right:2px;margin-bottom:5px;
    display:flex;flex-direction:column-reverse;}

/* User bubble */
.umsg{background:linear-gradient(135deg,#002d99,#003dbb);border:1px solid #2277ff;
    border-radius:14px 14px 3px 14px;padding:8px 14px;margin:4px 0 4px 25%;
    color:#fff;font-size:0.84rem;line-height:1.55;box-shadow:0 2px 10px rgba(0,60,180,0.3);}

/* ── ANSWER CARD ── */
.acard{background:#05110a;border:1px solid #008833;border-radius:14px 14px 14px 3px;
    padding:0;margin:4px 0 4px 0;overflow:hidden;box-shadow:0 2px 16px rgba(0,130,50,0.25);}

/* Answer top — AI response */
.atop{padding:10px 14px;background:linear-gradient(135deg,#031a0a,#042210);}
.atop .alabel{color:#33dd66;font-size:0.65rem;font-weight:700;letter-spacing:1px;margin-bottom:4px;}
.atop .atext{color:#ccffdd;font-size:0.82rem;line-height:1.65;}

/* Rule section */
.arules{border-top:1px solid #004422;padding:8px 14px;background:#030f06;}
.arules .rlabel{color:#00cc55;font-size:0.63rem;font-weight:700;letter-spacing:1px;
    margin-bottom:6px;display:flex;align-items:center;gap:6px;}
.ritem{background:rgba(0,50,20,0.6);border:1px solid #006633;border-radius:8px;
    padding:6px 10px;margin-bottom:5px;}
.ritem:last-child{margin-bottom:0;}
.rnum{display:inline-block;background:#004422;border:1px solid #00aa44;
    border-radius:4px;padding:1px 7px;color:#00ff88;font-size:0.65rem;
    font-weight:800;margin-right:6px;}
.rhead{color:#aaffcc;font-size:0.72rem;font-weight:700;}
.rpage{color:#44aa66;font-size:0.62rem;margin-left:6px;}
.rtext{color:#88ccaa;font-size:0.68rem;line-height:1.55;margin-top:4px;
    padding-top:4px;border-top:1px dashed #004422;}

/* Source footer */
.asrc{border-top:1px solid #003322;padding:5px 14px;background:#030f06;
    font-size:0.62rem;display:flex;flex-wrap:wrap;gap:4px;align-items:center;}
.stg{display:inline-block;background:rgba(0,50,100,0.4);border:1px solid #005599;
    border-radius:5px;padding:1px 7px;color:#55aadd;font-size:0.6rem;}
.stg-r{display:inline-block;background:rgba(0,80,30,0.4);border:1px solid #007733;
    border-radius:5px;padding:1px 7px;color:#33cc66;font-size:0.6rem;font-weight:600;}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#001628,#002244)!important;
    border:1px solid #004acc!important;color:#55aaff!important;
    border-radius:7px!important;font-size:0.7rem!important;
    padding:4px 8px!important;width:100%!important;transition:all 0.2s!important;}
.stButton>button:hover{background:linear-gradient(135deg,#002244,#003366)!important;
    border-color:#2277ff!important;color:#fff!important;
    box-shadow:0 0 8px rgba(0,80,220,0.4)!important;}

/* Chat input */
.stChatInput>div{background:#001628!important;border:2px solid #0055dd!important;
    border-radius:10px!important;box-shadow:0 0 14px rgba(0,80,200,0.3)!important;}
.stChatInput textarea{color:#fff!important;font-size:0.83rem!important;background:#001628!important;}
.stChatInput textarea::placeholder{color:#4477aa!important;}

/* Sample question bar */
.sq-bar{display:flex;align-items:center;gap:8px;margin-bottom:6px;flex-wrap:wrap;}
.sq-label{color:#55aaff;font-size:0.68rem;font-weight:700;white-space:nowrap;}
.stSelectbox>div>div{background:#001628!important;border:1px solid #0055dd!important;
    border-radius:7px!important;}
.stSelectbox [data-baseweb=select]{background:#001628!important;}
.stSelectbox [data-baseweb=selected-option]{color:#aaddff!important;font-size:0.78rem!important;}
.stSelectbox [data-baseweb=input]{color:#aaddff!important;font-size:0.78rem!important;}
.sq-go button{background:linear-gradient(135deg,#001628,#002244)!important;
    border:1px solid #0055dd!important;color:#55aaff!important;
    border-radius:7px!important;font-size:0.75rem!important;
    padding:4px 14px!important;}

/* Disclaimer */
.disc{background:linear-gradient(135deg,#140e00,#201500);border:1px solid #bb7700;
    border-radius:8px;padding:5px 12px;margin-top:5px;font-size:0.65rem;color:#ffaa33;line-height:1.5;}
.disc b{color:#ffcc00;}
.footer{text-align:center;color:#223344;font-size:0.58rem;
    padding:3px 0 1px;border-top:1px solid #001628;margin-top:4px;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important;}
</style>
""", unsafe_allow_html=True)

# ── RULE LOOKUP ───────────────────────────────
@st.cache_resource(show_spinner="📋 Loading rules index...")
def build_rule_lookup():
    rule_dict = {}
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    if not pdf_files:
        return rule_dict
    for pdf_path in pdf_files:
        fname = pdf_path.name.lower()
        if not any(x in fname for x in ["nepra","csm","consumer","qesco"]):
            continue
        try:
            reader = PdfReader(str(pdf_path))
            for pg in range(4, len(reader.pages)):
                raw = reader.pages[pg].extract_text() or ""
                text = _clean(raw)
                if not text.strip():
                    continue
                printed = max(1, pg + 1 - PAGE_OFFSET)
                # Extract sections: "6.1.1 HEADING\nbody text" or "6.1.1 Heading — body"
                for m in re.finditer(
                    r'(\d+\.\d+(?:\.\d+)*)\s+([A-Z][A-Z\s/&,()\-]{4,60}?)(?:\n|[—\-])((?:[\s\S]{10,600}?))(?=\n\d+\.\d+(?:\.\d+)*\s+[A-Z]|\nCHAPTER|\Z)',
                    text
                ):
                    rnum  = m.group(1).strip()
                    rhead = m.group(2).strip()
                    rbody = re.sub(r'\s+', ' ', m.group(3)).strip()[:400]
                    if rnum not in rule_dict:
                        rule_dict[rnum] = {"heading": rhead, "text": rbody, "page": printed}
                # Chapters
                ch = re.search(r'(CHAPTER\s+\d+)\s*\n([A-Z][A-Z\s/&,\-]+)', text)
                if ch:
                    key = ch.group(1).strip()
                    if key not in rule_dict:
                        rule_dict[key] = {"heading": ch.group(2).strip(), "text": "", "page": printed}
        except Exception as e:
            st.warning(f"⚠️ Rule lookup — {pdf_path.name}: {e}")
    if not rule_dict:
        st.warning("⚠️ No rules indexed. Ensure NEPRA CSM PDF is in the `pdfs/` folder.")
    return rule_dict

# ── RAG FUNCTIONS ─────────────────────────────
@st.cache_resource(show_spinner="⚡ Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

def find_rules_in_chunk(text):
    refs = set()
    for m in re.finditer(r'\b(\d+\.\d+(?:\.\d+)*)\b', text):
        refs.add(m.group(1))
    for m in re.finditer(r'(CHAPTER\s+\d+)', text, re.IGNORECASE):
        refs.add(m.group(1).upper())
    return list(refs)[:6]

def _clean(text):
    return text.encode("ascii", "replace").decode("ascii")

def load_pdfs_from_folder():
    chunks, metas = [], []
    for pdf_path in sorted(Path(PDF_FOLDER).glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            fname  = pdf_path.name.lower()
            offset = PAGE_OFFSET if any(x in fname for x in ["nepra","csm","consumer","qesco"]) else 0
            for pg, page in enumerate(reader.pages):
                try:
                    raw = page.extract_text() or ""
                    text = _clean(raw)
                    if len(text.strip()) < 50:
                        continue
                    printed = max(1, pg + 1 - offset)
                    rules   = find_rules_in_chunk(text)
                    for i in range(0, max(1, len(text) - 500 + 1), 400):
                        chunk = text[i:i+500].strip()
                        if len(chunk) > 80:
                            chunks.append(chunk)
                            metas.append({"file": pdf_path.name, "page": printed, "rules": rules})
                except Exception:
                    continue
        except Exception as e:
            st.warning(f"⚠️ {pdf_path.name}: {e}")
    for txt_path in sorted(Path(PDF_FOLDER).glob("*.txt")):
        try:
            text = _clean(txt_path.read_text(encoding="utf-8", errors="ignore"))
            for i in range(0, max(1, len(text) - 500 + 1), 400):
                chunk = text[i:i+500].strip()
                if len(chunk) > 80:
                    chunks.append(chunk)
                    metas.append({"file": txt_path.name, "page": 1, "rules": []})
        except Exception:
            continue
    return chunks, metas

def build_index(chunks, metas):
    vecs = np.array(load_embedder().encode(chunks, show_progress_bar=False), dtype="float32")
    faiss.normalize_L2(vecs)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    faiss.write_index(idx, f"{VECTORSTORE_PATH}/index.faiss")
    pickle.dump({"chunks":chunks,"metas":metas}, open(f"{VECTORSTORE_PATH}/data.pkl","wb"))
    return idx, chunks, metas

def load_index():
    idx = faiss.read_index(f"{VECTORSTORE_PATH}/index.faiss")
    d   = pickle.load(open(f"{VECTORSTORE_PATH}/data.pkl","rb"))
    return idx, d["chunks"], d["metas"]

def retrieve(q, idx, chunks, metas, k=8):
    qv = np.array(load_embedder().encode([q]), dtype="float32")
    faiss.normalize_L2(qv)
    _, ids = idx.search(qv, k)
    return [{"chunk":chunks[i],"meta":metas[i]} for i in ids[0] if i!=-1]

def ask_groq(question, hits):
    ctx = "\n---\n".join(
        "[Printed Page {page} | Sections: {rules}]\n{chunk}".format(
            page=h["meta"]["page"],
            rules=", ".join(h["meta"].get("rules",[])) or "General",
            chunk=h["chunk"]
        ) for h in hits
    )
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role":"user","content":
            "You are a NEPRA Consumer Service Manual (CSM NOV-2025) expert for QESCO.\n"
            "Answer using ONLY the provided context.\n\n"
            "Format your answer EXACTLY like this:\n"
            "SUMMARY: [One clear sentence answer]\n\n"
            "DETAILS:\n"
            "• [Point 1]\n"
            "• [Point 2]\n"
            "• [Point 3]\n\n"
            "REFERENCE: Section [X.X], Page [N] of NEPRA CSM NOV-2025\n\n"
            "If information not found: write 'Not found in NEPRA CSM NOV-2025'\n\n"
            f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"}],
        temperature=0, max_tokens=800,
    )
    return resp.choices[0].message.content

def format_answer(raw):
    """Format the AI answer into structured HTML sections."""
    html = ""
    summary = re.search(r'SUMMARY:\s*(.+?)(?=\nDETAILS:|$)', raw, re.DOTALL)
    details  = re.search(r'DETAILS:\s*([\s\S]+?)(?=\nREFERENCE:|$)', raw)
    ref      = re.search(r'REFERENCE:\s*(.+?)$', raw, re.DOTALL | re.MULTILINE)

    if summary:
        html += (
            f'<div style="background:linear-gradient(135deg,#002d1a,#003d11);'
            f'border:1px solid #00bb44;border-radius:10px;padding:10px 14px;margin-bottom:8px;">'
            f'<div style="color:#00ff88;font-size:0.6rem;font-weight:800;letter-spacing:1.5px;margin-bottom:4px;">'
            f'ANSWER SUMMARY</div>'
            f'<div style="color:#eeffee;font-size:0.88rem;font-weight:600;line-height:1.6;">'
            f'{summary.group(1).strip()}</div></div>'
        )

    if details:
        det_text = details.group(1).strip()
        bullets = [b.strip().lstrip("•-* ") for b in det_text.split("\n") if b.strip().lstrip("•-* ")]
        if bullets:
            html += (
                f'<div style="background:#030f06;border:1px solid #004422;'
                f'border-radius:8px;padding:10px 14px;margin-bottom:8px;">'
                f'<div style="color:#33dd66;font-size:0.6rem;font-weight:800;letter-spacing:1.5px;margin-bottom:8px;">'
                f'DETAILS</div>'
            )
            for b in bullets:
                html += (
                    f'<div style="display:flex;gap:8px;margin-bottom:7px;align-items:flex-start;">'
                    f'<span style="color:#00ff88;font-weight:900;font-size:0.85rem;flex-shrink:0;padding-top:1px;">▸</span>'
                    f'<span style="color:#ccffdd;font-size:0.82rem;line-height:1.6;">{b}</span>'
                    f'</div>'
                )
            html += '</div>'

    if ref:
        html += (
            f'<div style="background:rgba(0,60,120,0.35);border:1px solid #005599;'
            f'border-radius:8px;padding:7px 12px;display:flex;align-items:center;gap:6px;">'
            f'<span style="color:#4499dd;font-size:0.85rem;">&#128196;</span>'
            f'<span style="color:#88ccff;font-size:0.75rem;">{ref.group(1).strip()}</span>'
            f'</div>'
        )

    if not html:
        html = f'<div style="color:#ccffdd;font-size:0.84rem;line-height:1.65;padding:8px;">{raw}</div>'
    return html

def build_rule_boxes(hits, rule_lookup):
    """Build rule explanation boxes from matched rules in hits."""
    shown = set()
    items = []
    for h in hits:
        for rnum in h["meta"].get("rules", []):
            if rnum in shown or rnum not in rule_lookup:
                continue
            shown.add(rnum)
            info  = rule_lookup[rnum]
            head  = info["heading"][:65]
            body  = info["text"][:320]
            page  = info["page"]
            body_html = (
                f'<div class="rtext">{body}{"..." if len(info["text"])>320 else ""}</div>'
                if body else ""
            )
            items.append(
                f'<div class="ritem">'
                f'<span class="rnum">{rnum}</span>'
                f'<span class="rhead">{head}</span>'
                f'<span class="rpage">— p.{page}</span>'
                f'{body_html}'
                f'</div>'
            )
    if not items:
        return ""
    return (
        f'<div class="arules">'
        f'<div class="rlabel">📋 NEPRA CSM Rules &amp; Provisions Referenced</div>'
        + "".join(items) +
        f'</div>'
    )

def build_src(hits):
    seen, tags = [], []
    for h in hits:
        k = (h["meta"]["file"], h["meta"]["page"])
        if k not in seen:
            seen.append(k)
            tags.append(f'<span class="stg">📄 {h["meta"]["file"]} — p.<b>{h["meta"]["page"]}</b></span>')
        for r in h["meta"].get("rules", []):
            if f"r:{r}" not in seen:
                seen.append(f"r:{r}")
                tags.append(f'<span class="stg-r">§ {r}</span>')
    return '<div class="asrc">' + " ".join(tags) + '</div>' if tags else ""

def render_answer(answer_text, hits, rule_lookup):
    answer_html = format_answer(answer_text)
    rule_html   = build_rule_boxes(hits, rule_lookup)
    src_html    = build_src(hits)
    return (
        f'<div class="acard">'
        f'<div class="atop">'
        f'{answer_html}'
        f'</div>'
        f'{rule_html}'
        f'{src_html}'
        f'</div>'
    )

# ── SESSION STATE ─────────────────────────────
for k,v in [("msgs",[]),("idx",None),("chunks",None),("metas",None),("ready",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ── AUTO LOAD ─────────────────────────────────
if not st.session_state.ready:
    has_index = Path(f"{VECTORSTORE_PATH}/index.faiss").exists() and Path(f"{VECTORSTORE_PATH}/data.pkl").exists()
    pdfs = list(Path(PDF_FOLDER).glob("*.pdf")) if Path(PDF_FOLDER).exists() else []
    if has_index:
        with st.spinner("⚡ Loading NEPRA knowledge base..."):
            i,c,m = load_index()
            st.session_state.idx,st.session_state.chunks,st.session_state.metas,st.session_state.ready=i,c,m,True
    elif pdfs:
        with st.spinner(f"⚡ Indexing {len(pdfs)} PDF(s)..."):
            c,m = load_pdfs_from_folder()
            i,c,m = build_index(c,m)
            st.session_state.idx,st.session_state.chunks,st.session_state.metas,st.session_state.ready=i,c,m,True
        st.rerun()

rule_lookup = build_rule_lookup()

# ── HEADER ────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="logo-nepra">
    <div class="la">NATIONAL</div>
    <div class="lb">NEPRA</div>
    <div class="lc">Electric Power<br>Regulatory Authority</div>
  </div>
  <div class="hdr-mid">
    <h2>&#9889; Consumer Service Manual Assistant<br>CSM &mdash; NOV 2025</h2>
    <p>New Connection &middot; Billing &middot; Detection &middot; Complaints &middot; Net Metering</p>
    <div class="badges">
      <span class="bd-b">&#9889; Groq LLaMA RAG</span>
      <span class="bd-g">&#9670; QESCO Balochistan</span>
    </div>
  </div>
  <div class="logo-qesco">
    <div class="la">QUETTA ELECTRIC</div>
    <div class="lb">QESCO</div>
    <div class="lc">Supply Company<br>Balochistan</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── STATUS BAR ────────────────────────────────
if st.session_state.ready:
    files = sorted(set(m["file"] for m in st.session_state.metas))
    st.markdown(
        f'<div class="sbar">&#9989; <b>{len(st.session_state.chunks)} chunks</b>'
        f' &nbsp;&middot;&nbsp; {" &middot; ".join(files)}'
        f' &nbsp;&middot;&nbsp; &#128214; <b>{len(rule_lookup)} rules loaded</b></div>',
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ Add NEPRA CSM PDF to `pdfs/` folder and restart.")
    st.stop()

# ── SAMPLE QUESTIONS ──────────────────────────
# (label, full_question)
SAMPLES = [
    # New Connection — Chapter 2 Categories
    ("Ch-2: Cat-1: 230/400V ≤15kW",        "What are the requirements for Category 1 new connection including voltage, load, tariff category, other types of connections, and sanctioning officer?"),
    ("Ch-2: Cat-2: 230/400V >15-70kW",     "What are the requirements for Category 2 new connection including voltage, load, tariff category, other types of connections, and sanctioning officer?"),
    ("Ch-2: Cat-3: 230/400V >70-500kW",   "What are the requirements for Category 3 new connection including voltage, load, tariff category, and sanctioning officer?"),
    ("Ch-2: Cat-4: 1kV-33kV >500kW-5MW",  "What are the requirements for Category 4 new connection including voltage, load, tariff category, and sanctioning officer?"),
    ("Ch-2: Cat-5: 66kV+ All Loads",        "What are the requirements for Category 5 new connection including voltage, load, tariff category, and sanctioning officer?"),
    # Chapter 4–16
    ("Ch-4: Metering Installation Procedure",       "What is the procedure for metering installation according to NEPRA CSM?"),
    ("Ch-5: Security Deposit Rates",                "What are the security deposit rates in NEPRA CSM NOV-2025?"),
    ("Ch-6: Meter Reading & Billing",               "Explain the meter reading and billing procedure under NEPRA CSM."),
    ("Ch-8: Disconnection & Reconnection",          "When can QESCO disconnect a service? What is the reconnection procedure?"),
    ("Ch-9: Theft / Detection Penalties",          "What are the penalties for dishonest abstraction or theft of electricity?"),
    ("Ch-10: Consumer Complaint & Redressal",       "How should a consumer file a complaint? What is the redressal timeframe?"),
    ("Ch-13: Net Metering Facility",               "What is the net metering facility and how does it work?"),
    ("Ch-14: Consumer Rights & DISCO Obligations",  "What are the consumer rights and DISCO obligations under Chapter 14?"),
    ("Ch-16: EV Charging Stations",                 "What are the requirements for Public Electric Vehicle Charging Stations?"),
]

# Only show picker when chat is empty
if not st.session_state.msgs:
    # Dropdown — label shown, full question stored as value
    opts        = [q for _, q in SAMPLES]
    disp_labels = [f"▸ {lbl}" for lbl, _ in SAMPLES]
    opt_map     = dict(zip(disp_labels, opts))

    st.markdown('<div class="sq-bar">', unsafe_allow_html=True)
    cols = st.columns([1, 5, 1])
    with cols[0]:
        st.markdown('<span class="sq-label">&#9654; ASK</span>', unsafe_allow_html=True)
    with cols[1]:
        chosen = st.selectbox(
            "Choose a topic...",
            options=[""] + disp_labels,
            index=0,
            format_func=lambda x: "— Select a topic —" if x == "" else x,
            label_visibility="collapsed", key="sq_select"
        )
    with cols[2]:
        go = st.button("Go", key="sq_go")
    st.markdown('</div>', unsafe_allow_html=True)

    if go and chosen:
        q = opt_map[chosen]
        st.session_state.msgs.append({"role":"user","content":q})
        with st.spinner("⚡ Searching NEPRA CSM..."):
            hits   = retrieve(q, st.session_state.idx, st.session_state.chunks, st.session_state.metas)
            answer = ask_groq(q, hits)
            card   = render_answer(answer, hits, rule_lookup)
        st.session_state.msgs.append({"role":"assistant","card":card})
        st.rerun()

# ── CHAT HISTORY ──────────────────────────────
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
for msg in reversed(st.session_state.msgs):
    if msg["role"] == "user":
        st.markdown(f'<div class="umsg">&#x1F9D1; {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(msg.get("card",""), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.msgs:
    if st.button("&#128465; Clear Chat"):
        st.session_state.msgs = []
        st.rerun()

# ── TEXT INPUT ────────────────────────────────
if prompt := st.chat_input("⚡ Ask about NEPRA CSM — Connection / Billing / Detection / Complaints / Net Metering..."):
    st.session_state.msgs.append({"role":"user","content":prompt})
    with st.spinner("⚡ Searching NEPRA CSM..."):
        hits   = retrieve(prompt, st.session_state.idx, st.session_state.chunks, st.session_state.metas)
        answer = ask_groq(prompt, hits)
        card   = render_answer(answer, hits, rule_lookup)
    st.session_state.msgs.append({"role":"assistant","card":card})
    st.rerun()

# ── LOGOUT ────────────────────────────────────
if st.button("🔒 Logout"):
    st.session_state.authenticated = False
    st.rerun()

# ── FOOTER ────────────────────────────────────
st.markdown("""
<div class="disc">&#9888; <b>Disclaimer:</b> This AI-based app provides answers generated from NEPRA CSM NOV-2025 for reference only. While accuracy is aimed for, answers may be incomplete or incorrect. Always verify from official NEPRA CSM documents and consult QESCO/NEPRA for official decisions. Page and rule references are indicative — cross-check with original document.</div>
<div class="footer">AI for Everyone &mdash; Batch-05 &nbsp;|&nbsp; NEPRA CSM NOV-2025
&nbsp;|&nbsp; QESCO Balochistan &nbsp;|&nbsp; Groq LLaMA + FAISS</div>
<div class="footer" style="font-size:0.65rem; margin-top:4px;">
Programmer: <b>Faruk Ali Khan</b> &nbsp;|&nbsp; AI Engineer: <b>Faruk Ali Khan</b>
</div>
""", unsafe_allow_html=True)
