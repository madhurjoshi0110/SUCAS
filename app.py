"""
╔══════════════════════════════════════════════════════════╗
║   SMART URBAN COMPLAINT ANALYSIS SYSTEM — app.py         ║
║   Streamlit Frontend + Backend                           ║
╚══════════════════════════════════════════════════════════╝

Place rf_model.pkl, tfidf_vectorizer.pkl, and Employees.csv in the same
folder as this file. For the AI Chat Assistant page, also set up a .env
file (see .env.example) with a HuggingFace API token.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib, time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from text_utils import preprocess
from priority_engine import (
    DEPT_PRIORITY, MONTH_PRIORITY,
    hour_priority, compute_priority, assign_workers, ticket_id, build_role_map,
)
from llm_workflow import llm_available, run_chat_turn, new_chat_state, get_chat_model

# ──────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────
st.set_page_config(
    page_title="SUCAS — Smart Urban Complaint Analysis",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────
# CUSTOM CSS — dark civic theme
# ──────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root{
  --bg:#0d1117;
  --surface:#161b22;
  --surface2:#21262d;
  --accent:#00e5ff;
  --accent2:#ff6b35;
  --accent3:#a8ff3e;
  --high:#ff4d4d;
  --med:#ffa94d;
  --low:#69db7c;
  --text:#e6edf3;
  --muted:#8b949e;
  --border:#30363d;
}

/* ── Global ── */
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg) !important;
  font-family:'DM Sans',sans-serif;
  color:var(--text);
}
[data-testid="stSidebar"]{
  background:var(--surface) !important;
  border-right:1px solid var(--border);
}
h1,h2,h3{font-family:'Syne',sans-serif;}

/* ── Hero banner ── */
.hero{
  background: linear-gradient(135deg,#0d1117 0%,#0a2332 50%,#0d1117 100%);
  border:1px solid var(--border);
  border-radius:16px;
  padding:2.5rem 3rem;
  margin-bottom:2rem;
  position:relative;
  overflow:hidden;
}
.hero::before{
  content:'';
  position:absolute;
  top:-60px;right:-60px;
  width:280px;height:280px;
  background:radial-gradient(circle,rgba(0,229,255,.12),transparent 70%);
  border-radius:50%;
}
.hero-title{
  font-family:'Syne',sans-serif;
  font-size:2.4rem;
  font-weight:800;
  background:linear-gradient(90deg,var(--accent),var(--accent3));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  margin:0 0 .4rem;
  letter-spacing:-1px;
}
.hero-sub{color:var(--muted);font-size:1.05rem;margin:0;}

/* ── Ticket card ── */
.ticket{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:12px;
  padding:1.4rem 1.6rem;
  margin:.5rem 0;
  animation: fadeUp .5s ease both;
}
.ticket-id{
  font-family:'Syne',sans-serif;
  font-size:.75rem;
  color:var(--accent);
  letter-spacing:2px;
  text-transform:uppercase;
  margin-bottom:.5rem;
}
.ticket-title{
  font-size:1.3rem;
  font-weight:600;
  margin:.2rem 0 .8rem;
}

/* ── Priority badge ── */
.badge{
  display:inline-block;
  padding:.25rem .75rem;
  border-radius:20px;
  font-size:.78rem;
  font-weight:600;
  font-family:'Syne',sans-serif;
  letter-spacing:.5px;
}
.badge-high{background:rgba(255,77,77,.15);color:var(--high);border:1px solid rgba(255,77,77,.4);}
.badge-med {background:rgba(255,169,77,.15);color:var(--med);border:1px solid rgba(255,169,77,.4);}
.badge-low {background:rgba(105,219,124,.15);color:var(--low);border:1px solid rgba(105,219,124,.4);}

/* ── Stat cards ── */
.stat-row{display:flex;gap:1rem;margin:1rem 0;}
.stat-card{
  flex:1;
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:12px;
  padding:1.1rem 1.4rem;
  text-align:center;
  transition:border-color .25s;
}
.stat-card:hover{border-color:var(--accent);}
.stat-num{
  font-family:'Syne',sans-serif;
  font-size:2rem;
  font-weight:800;
  color:var(--accent);
}
.stat-lbl{font-size:.8rem;color:var(--muted);margin-top:.2rem;}

/* ── Worker card ── */
.worker-card{
  background:var(--surface2);
  border:1px solid var(--border);
  border-left:4px solid var(--accent);
  border-radius:10px;
  padding:1rem 1.3rem;
  margin:.6rem 0;
  animation: fadeUp .4s ease both;
}
.worker-name{font-weight:600;font-size:1.05rem;}
.worker-role{font-size:.8rem;color:var(--accent);font-family:'Syne',sans-serif;letter-spacing:.5px;}
.worker-meta{font-size:.78rem;color:var(--muted);margin-top:.3rem;display:flex;gap:1rem;flex-wrap:wrap;}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:.35rem;}
.status-Available .status-dot{background:#69db7c;}
.status-Busy .status-dot{background:#ffa94d;}
.status-On.Leave .status-dot,.status-On-Leave .status-dot{background:#ff6b6b;}

/* ── Step pill ── */
.step-pill{
  display:inline-flex;align-items:center;gap:.5rem;
  background:var(--surface2);border:1px solid var(--border);
  border-radius:30px;padding:.4rem 1rem;margin:.3rem .2rem;
  font-size:.85rem;
}
.step-num{
  background:var(--accent);color:#000;
  border-radius:50%;width:22px;height:22px;
  display:inline-flex;align-items:center;justify-content:center;
  font-weight:700;font-size:.7rem;font-family:'Syne',sans-serif;
}

/* ── Keyframes ── */
@keyframes fadeUp{
  from{opacity:0;transform:translateY(16px);}
  to  {opacity:1;transform:translateY(0);}
}
@keyframes pulse{
  0%,100%{box-shadow:0 0 0 0 rgba(0,229,255,.4);}
  50%{box-shadow:0 0 0 8px rgba(0,229,255,0);}
}
.pulse{animation:pulse 2s infinite;}

/* ── Streamlit overrides ── */
.stTextArea textarea{
  background:var(--surface2) !important;
  border:1px solid var(--border) !important;
  color:var(--text) !important;
  border-radius:10px !important;
  font-family:'DM Sans',sans-serif !important;
  font-size:.95rem !important;
}
.stTextArea textarea:focus{border-color:var(--accent) !important;}
.stButton>button{
  background:linear-gradient(135deg,var(--accent),#0097a7) !important;
  color:#000 !important;
  font-family:'Syne',sans-serif !important;
  font-weight:700 !important;
  border:none !important;
  border-radius:10px !important;
  padding:.6rem 2rem !important;
  letter-spacing:.5px !important;
  transition:opacity .2s !important;
}
.stButton>button:hover{opacity:.85 !important;}

/* ── Sidebar links ── */
.sidebar-link{
  display:block;
  padding:.5rem .8rem;
  color:var(--muted);
  text-decoration:none;
  border-radius:8px;
  font-size:.9rem;
  transition:background .2s,color .2s;
}
.sidebar-link:hover{background:var(--surface2);color:var(--accent);}

div[data-testid="stMetricValue"]{color:var(--accent) !important;font-family:'Syne',sans-serif;}
.stSelectbox>div>div{background:var(--surface2) !important;border-color:var(--border) !important;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# LOAD MODELS & DATA
# ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    # rf_model.pkl / tfidf_vectorizer.pkl were saved with joblib.dump(), so
    # they must be loaded with joblib.load() (plain pickle.load() will
    # fail on sklearn's numpy-array internals).
    model = joblib.load("rf_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    # No stop.pkl / lemma.pkl on disk — these are cheap to build at
    # startup and don't need to be pickled.
    stop = set(stopwords.words("english"))
    lemma = WordNetLemmatizer()
    return model, tfidf, stop, lemma

@st.cache_data(show_spinner=False)
def load_employees():
    df = pd.read_csv("Employees.csv")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Employee_ID": "employee_id",
        "Name": "name",
        "Department": "department",
        "Designation": "role",
        "Ward": "ward",
        "Current_Workload": "workload",
        "Max_Workload": "max_workload",
        "Status": "status",
        "Phone": "phone",
        "Experience": "experience",
    })
    for col in ("name", "department", "role", "ward", "status"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    for col in ("workload", "max_workload", "experience"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

# ── Priority logic lives in priority_engine.py (imported above) so the
#    manual complaint form and the AI chat assistant share one set of rules.

def render_worker_card(row):
    status = row.get("status", "—")
    dot_class = f"status-{str(status).replace(' ', '-')}"
    st.markdown(f"""
    <div class="worker-card">
      <div class="worker-name">👤 {row.get('name','—')}</div>
      <div class="worker-role">{row.get('role','—')}</div>
      <div class="worker-meta">
        <span class="{dot_class}"><span class="status-dot"></span>{status}</span>
        <span>📍 {row.get('ward','—')}</span>
        <span>📊 {row.get('workload','—')}/{row.get('max_workload','—')} active</span>
        <span>📞 {row.get('phone','—')}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem;'>
      <div style='font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;
                  background:linear-gradient(90deg,#00e5ff,#a8ff3e);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        🏙️ SUCAS
      </div>
      <div style='font-size:.72rem;color:#8b949e;letter-spacing:2px;text-transform:uppercase;'>
        Smart Urban Complaint Analysis
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", ["🏠 Home", "📋 File Complaint", "🤖 AI Chat Assistant", "📊 Dashboard", "👷 Workers", "ℹ️ About"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Quick Links**", unsafe_allow_html=True)
    links = {
        "BBMP Official Portal":"https://bbmp.gov.in",
        "Bangalore City Services":"https://bbmp.gov.in/services",
        "Complaint Helpline":"https://bbmp.gov.in/contact",
        "Emergency Services":"https://112.gov.in",
    }
    for label, url in links.items():
        st.markdown(f'<a href="{url}" target="_blank" class="sidebar-link">🔗 {label}</a>', unsafe_allow_html=True)

    st.markdown("---")
    now = datetime.now()
    st.markdown(f"""
    <div style='font-size:.78rem;color:#8b949e;padding:.5rem 0;'>
      🕐 <b>{now.strftime("%H:%M")}</b>  &nbsp;|&nbsp;
      📅 {now.strftime("%d %b %Y")}
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────
# LOAD RESOURCES
# ──────────────────────────────────────────
try:
    model, tfidf, stop_obj, lemma_obj = load_models()
    employees = load_employees()
    # Reconstruct stop words set & lemmatizer from pickled objects
    stop_words = stop_obj if isinstance(stop_obj, (set, list, frozenset)) else set(stopwords.words("english"))
    lemmatizer = lemma_obj if hasattr(lemma_obj, "lemmatize") else WordNetLemmatizer()
    # Department -> typical worker designation, derived straight from
    # Employees.csv (used for dashboard displays & the priority simulator's
    # department picker) — not a hand-typed guess.
    role_map = build_role_map(employees)
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ══════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════
if "Home" in page:
    st.markdown("""
    <div class="hero">
      <div class="hero-title">Smart Urban Complaint Analysis System</div>
      <div class="hero-sub">AI-powered civic complaint routing · Real-time priority scoring · Intelligent worker assignment</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    if models_loaded:
        total_roles = employees["role"].nunique()
        total_depts = employees["department"].nunique()
        total_workers = len(employees)
    else:
        total_roles, total_depts, total_workers = "—", "—", "—"

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="stat-card pulse">
          <div class="stat-num">{total_depts}</div>
          <div class="stat-lbl">Departments Covered</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-num">{total_workers}</div>
          <div class="stat-lbl">Active Workers</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-num">{total_roles}</div>
          <div class="stat-lbl">Worker Roles</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-num">3</div>
          <div class="stat-lbl">Priority Levels</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### How It Works")

    steps = [
        ("1","Describe Complaint","Write your civic issue in plain language."),
        ("2","AI Classification","Our ML model identifies the department."),
        ("3","Priority Scoring","Date, time & department determine urgency."),
        ("4","Worker Assignment","Best-matched workers are auto-assigned."),
        ("5","Ticket Generated","Track your complaint with a unique ID."),
    ]
    cols = st.columns(5)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="ticket" style="text-align:center;min-height:130px;">
              <div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                          color:var(--accent);margin-bottom:.4rem;">{num}</div>
              <div style="font-weight:600;margin-bottom:.4rem;font-size:.95rem;">{title}</div>
              <div style="font-size:.8rem;color:#8b949e;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Department coverage
    st.markdown("### Department Intelligence Map")
    dept_df = pd.DataFrame([
        {"Department": d, "Role": r, "Priority Weight": DEPT_PRIORITY.get(d,1)}
        for d,r in role_map.items()
    ])
    fig = px.treemap(
        dept_df, path=["Role","Department"], values="Priority Weight",
        color="Priority Weight",
        color_continuous_scale=["#1a3a4a","#00e5ff","#a8ff3e"],
        title="Departments by Role & Priority Weight",
    )
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="DM Sans"),
        margin=dict(t=50,l=0,r=0,b=0),
        title_font=dict(family="Syne",size=16),
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════
#  PAGE: FILE COMPLAINT
# ══════════════════════════════════════════
elif "Complaint" in page:
    st.markdown("## 📋 File a Complaint")
    st.markdown("<small style='color:#8b949e;'>Your complaint is auto-classified by AI and routed to the right department instantly.</small>", unsafe_allow_html=True)

    if not models_loaded:
        st.error(f"⚠️ Models could not be loaded. Ensure all .pkl files are in the app directory.\n\n`{load_error}`")
        st.stop()

    col_form, col_info = st.columns([3,2], gap="large")

    with col_form:
        with st.form("complaint_form"):
            complaint_text = st.text_area(
                "Describe your complaint",
                placeholder="e.g. The street light near my house has not been working for the past 3 days. It is a safety concern at night...",
                height=140,
            )
            loc_col, area_col = st.columns(2)
            with loc_col:
                location = st.text_input("Ward / Area", placeholder="e.g. Koramangala")
            with area_col:
                locality = st.text_input("Landmark (optional)", placeholder="e.g. Near Metro Station")

            submitted = st.form_submit_button("🚀 Analyse & Submit Complaint")

        if submitted and not complaint_text.strip():
            st.warning("Please describe your complaint before submitting.")

        if submitted and complaint_text.strip():
            with st.spinner("Running AI classification…"):
                time.sleep(0.6)  # brief animation pause
                processed = preprocess(complaint_text, stop_words, lemmatizer)
                vec = tfidf.transform([processed])
                dept = model.predict(vec)[0]
                proba = model.predict_proba(vec)[0]
                dept_idx = list(model.classes_).index(dept)
                confidence = round(proba[dept_idx]*100, 1)

            now = datetime.now()
            priority_label, total_score, ds, ms, hs = compute_priority(dept, now)
            workers, req_role = assign_workers(dept, priority_label, employees)
            tid = ticket_id()

            # Store in session
            st.session_state["last_ticket"] = {
                "id": tid, "dept": dept, "priority": priority_label,
                "score": total_score, "ds": ds, "ms": ms, "hs": hs,
                "workers": workers, "role": req_role, "confidence": confidence,
                "text": complaint_text, "location": location,
                "submitted_at": now.strftime("%d %b %Y, %H:%M:%S"),
            }

        if "last_ticket" in st.session_state:
            tk = st.session_state["last_ticket"]
            badge_cls = {"High":"badge-high","Medium":"badge-med","Low":"badge-low"}[tk["priority"]]

            st.markdown(f"""
            <div class="ticket" style="border-left:4px solid var(--accent);margin-top:1.5rem;">
              <div class="ticket-id">✅ COMPLAINT REGISTERED · {tk['id']}</div>
              <div class="ticket-title">{tk['dept']}</div>
              <span class="badge {badge_cls}">{tk['priority']} Priority</span>
              &nbsp;
              <span style="font-size:.8rem;color:#8b949e;">Submitted: {tk['submitted_at']}</span>
            </div>
            """, unsafe_allow_html=True)

            # Priority breakdown
            st.markdown("#### 🎯 Priority Score Breakdown")
            sc1,sc2,sc3,sc4 = st.columns(4)
            sc1.metric("Total Score", tk["score"], help="Max=9 → High")
            sc2.metric("Dept Weight", tk["ds"])
            sc3.metric("Month Weight", tk["ms"])
            sc4.metric("Hour Weight", tk["hs"])

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=tk["score"],
                domain={"x":[0,1],"y":[0,1]},
                title={"text":"Priority Score","font":{"family":"Syne","size":16,"color":"#e6edf3"}},
                number={"font":{"color":"#00e5ff","family":"Syne","size":40}},
                gauge={
                    "axis":{"range":[0,9],"tickcolor":"#8b949e","tickfont":{"color":"#8b949e"}},
                    "bar":{"color":"#00e5ff"},
                    "steps":[
                        {"range":[0,4],"color":"rgba(105,219,124,.15)"},
                        {"range":[4,7],"color":"rgba(255,169,77,.15)"},
                        {"range":[7,9],"color":"rgba(255,77,77,.25)"},
                    ],
                    "threshold":{"line":{"color":"#ff4d4d","width":3},"thickness":.75,"value":7},
                    "bgcolor":"#161b22",
                    "bordercolor":"#30363d",
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#0d1117",font=dict(color="#e6edf3"),
                height=260, margin=dict(t=40,b=20,l=20,r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # AI confidence
            st.markdown(f"""
            <div class="ticket">
              <div style="font-size:.8rem;color:#8b949e;">AI Classification Confidence</div>
              <div style="font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;color:#a8ff3e;">
                {tk['confidence']}%
              </div>
              <div style="background:#21262d;border-radius:6px;height:8px;overflow:hidden;margin-top:.5rem;">
                <div style="height:100%;width:{tk['confidence']}%;
                            background:linear-gradient(90deg,#00e5ff,#a8ff3e);border-radius:6px;
                            transition:width 1s ease;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Assigned workers
            st.markdown("#### 👷 Assigned Workers")
            for _, row in tk["workers"].iterrows():
                render_worker_card(row)

    with col_info:
        st.markdown("""
        <div class="ticket">
          <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
                      color:#00e5ff;margin-bottom:1rem;">⚡ Priority Criteria</div>
          <div style="font-size:.85rem;color:#8b949e;line-height:1.8;">
            <b style="color:#e6edf3;">Department Weight (1–3)</b><br>
            Critical infrastructure = 3<br>Civic services = 2<br>Admin = 1<br><br>
            <b style="color:#e6edf3;">Month Weight (1–3)</b><br>
            Monsoon/Summer = 3<br>Moderate = 2<br>Winter = 1<br><br>
            <b style="color:#e6edf3;">Time of Day (1–3)</b><br>
            Late night (10pm–6am) = 3<br>Peak hours = 2<br>Daytime = 1<br><br>
            <b style="color:#a8ff3e;">Total ≥ 8 → High</b><br>
            <b style="color:#ffa94d;">Total 5–7 → Medium</b><br>
            <b style="color:#69db7c;">Total ≤ 4 → Low</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="ticket" style="margin-top:1rem;">
          <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
                      color:#00e5ff;margin-bottom:.8rem;">📞 Emergency Contacts</div>
          <div style="font-size:.85rem;line-height:2;">
            🚨 <a href="tel:112" style="color:#ff6b35;">112</a> — National Emergency<br>
            🏙️ <a href="tel:1533" style="color:#ff6b35;">1533</a> — BBMP Helpline<br>
            💧 <a href="tel:1916" style="color:#ff6b35;">1916</a> — Water Emergency<br>
            ⚡ <a href="tel:1912" style="color:#ff6b35;">1912</a> — BESCOM Electricity
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════
#  PAGE: AI CHAT ASSISTANT
# ══════════════════════════════════════════
elif "Chat" in page:
    st.markdown("## 🤖 AI Chat Assistant")
    st.markdown("<small style='color:#8b949e;'>Chat with SUCAS to register a complaint — it asks for the details it needs, then classifies, prioritizes, and routes it automatically.</small>", unsafe_allow_html=True)

    if not models_loaded:
        st.error(f"⚠️ Models could not be loaded. Ensure rf_model.pkl and tfidf_vectorizer.pkl are in the app directory.\n\n`{load_error}`")
        st.stop()

    if not llm_available:
        st.warning(
            "The AI Chat Assistant needs an LLM API connection that isn't configured yet.\n\n"
            "1. `pip install langgraph langchain langchain-huggingface python-dotenv`\n"
            "2. Create a `.env` file with `HUGGINGFACEHUB_API_TOKEN=your_token_here`\n"
            "3. (Optional) set `SUCAS_LLM_REPO_ID` to choose a different hosted chat model\n\n"
            "Nothing here gets pickled — the LLM is called live over the API, same as any "
            "other web request the app makes."
        )
        st.info("In the meantime, you can register a complaint from **📋 File Complaint**.")
        st.stop()

    @st.cache_resource(show_spinner=False)
    def get_chat_model_cached():
        return get_chat_model()

    if "chat_state" not in st.session_state:
        st.session_state["chat_state"] = new_chat_state()
        st.session_state["chat_display"] = []  # [(role, text), ...] for rendering
        st.session_state["chat_ticket"] = None

    for role, text in st.session_state["chat_display"]:
        with st.chat_message(role):
            st.markdown(text)

    if user_msg := st.chat_input("Describe your issue, or answer SUCAS's question…"):
        st.session_state["chat_display"].append(("user", user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("SUCAS is thinking…"):
                try:
                    chat_model = get_chat_model_cached()
                    new_state, reply, ticket = run_chat_turn(
                        st.session_state["chat_state"], user_msg, chat_model,
                        model, tfidf, stop_words, lemmatizer, employees,
                    )
                    st.session_state["chat_state"] = new_state
                except Exception as e:
                    reply = f"⚠️ The AI service ran into an error: `{e}`"
                    ticket = None
            st.markdown(reply)
        st.session_state["chat_display"].append(("assistant", reply))
        if ticket:
            st.session_state["chat_ticket"] = ticket
        st.rerun()

    if st.session_state.get("chat_ticket"):
        tk = st.session_state["chat_ticket"]
        prio = tk["priority"].get("priority", "Low")
        badge_cls = {"High":"badge-high","Medium":"badge-med","Low":"badge-low"}.get(prio, "badge-low")
        st.markdown(f"""
        <div class="ticket" style="border-left:4px solid var(--accent);margin-top:1rem;">
          <div class="ticket-id">✅ COMPLAINT REGISTERED VIA AI CHAT</div>
          <div class="ticket-title">{tk['department']}</div>
          <span class="badge {badge_cls}">{prio} Priority</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"**Reason:** {tk['priority'].get('reason','—')}  \n**Estimated time:** {tk['priority'].get('estimated_time','—')}")
        st.markdown("#### 👷 Assigned Workers")
        for w in tk["worker"]["workers"]:
            render_worker_card(w)
        if st.button("Start a new complaint"):
            st.session_state["chat_state"] = new_chat_state()
            st.session_state["chat_display"] = []
            st.session_state["chat_ticket"] = None
            st.rerun()

# ══════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════
elif "Dashboard" in page:
    st.markdown("## 📊 Analytics Dashboard")

    if not models_loaded:
        st.warning(f"⚠️ Models could not be loaded, so the dashboard has nothing to plot yet.\n\n`{load_error}`")
        st.stop()

    dept_df = pd.DataFrame([
        {"Department": d, "Role": r, "Priority Weight": DEPT_PRIORITY.get(d,1)}
        for d,r in role_map.items()
    ])

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        # Department by priority weight bar
        fig1 = px.bar(
            dept_df.sort_values("Priority Weight", ascending=True),
            x="Priority Weight", y="Department", orientation="h",
            color="Priority Weight",
            color_continuous_scale=["#69db7c","#ffa94d","#ff4d4d"],
            title="Department Priority Weights",
        )
        fig1.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3",family="DM Sans"),
            title_font=dict(family="Syne",size=15),
            yaxis=dict(tickfont=dict(size=10)),
            coloraxis_showscale=False,
            margin=dict(l=10,r=10,t=40,b=10),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with r1c2:
        # Role distribution pie
        role_counts = dept_df["Role"].value_counts().reset_index()
        role_counts.columns = ["Role","Count"]
        fig2 = px.pie(
            role_counts, names="Role", values="Count",
            title="Departments by Worker Role",
            color_discrete_sequence=px.colors.sequential.Teal,
            hole=.45,
        )
        fig2.update_layout(
            paper_bgcolor="#0d1117",
            font=dict(color="#e6edf3",family="DM Sans"),
            title_font=dict(family="Syne",size=15),
            legend=dict(font=dict(size=10)),
            margin=dict(l=0,r=0,t=40,b=0),
        )
        st.plotly_chart(fig2, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        # Month priority heatmap
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        month_scores = [MONTH_PRIORITY[i] for i in range(1,13)]
        fig3 = go.Figure(go.Bar(
            x=months, y=month_scores,
            marker_color=["#69db7c" if s==1 else "#ffa94d" if s==2 else "#ff4d4d" for s in month_scores],
            text=["Low" if s==1 else "Medium" if s==2 else "High" for s in month_scores],
            textposition="outside",
        ))
        fig3.update_layout(
            title="Monthly Priority Weights",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3",family="DM Sans"),
            title_font=dict(family="Syne",size=15),
            yaxis=dict(range=[0,4]),
            margin=dict(l=10,r=10,t=40,b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with r2c2:
        # Hour of day priority
        hours = list(range(24))
        h_scores = [hour_priority(h) for h in hours]
        h_labels = [f"{h:02d}:00" for h in hours]
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=h_labels, y=h_scores, mode="lines+markers",
            line=dict(color="#00e5ff",width=2.5),
            marker=dict(color=["#69db7c" if s==1 else "#ffa94d" if s==2 else "#ff4d4d" for s in h_scores], size=8),
            fill="tozeroy", fillcolor="rgba(0,229,255,.07)",
        ))
        fig4.update_layout(
            title="Hour-of-Day Priority Weights",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3",family="DM Sans"),
            title_font=dict(family="Syne",size=15),
            xaxis=dict(tickangle=-45,tickfont=dict(size=9)),
            yaxis=dict(range=[0,4]),
            margin=dict(l=10,r=10,t=40,b=10),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Worker role distribution
    st.markdown("### 👷 Worker Role Distribution")
    if models_loaded:
        worker_role_counts = employees["role"].value_counts().reset_index()
        worker_role_counts.columns = ["Role","Workers"]
        fig5 = px.bar(
            worker_role_counts.sort_values("Workers",ascending=True),
            x="Workers", y="Role", orientation="h",
            color="Workers",
            color_continuous_scale=["#0a2332","#00e5ff"],
            title="Workers Available per Role",
        )
        fig5.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3",family="DM Sans"),
            title_font=dict(family="Syne",size=15),
            coloraxis_showscale=False,
            margin=dict(l=10,r=10,t=40,b=10),
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Priority score simulator
    st.markdown("### 🔬 Priority Score Simulator")
    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_dept = st.selectbox("Department", list(role_map.keys()))
    with sim_c2:
        sim_month = st.selectbox("Month", list(range(1,13)), format_func=lambda x: datetime(2024,x,1).strftime("%B"))
    with sim_c3:
        sim_hour = st.slider("Hour of Day", 0, 23, 14)

    sim_dt = datetime(2024, sim_month, 1, sim_hour)
    sim_prio, sim_total, sim_ds, sim_ms, sim_hs = compute_priority(sim_dept, sim_dt)
    prio_color = {"High":"#ff4d4d","Medium":"#ffa94d","Low":"#69db7c"}[sim_prio]

    sc1,sc2,sc3,sc4,sc5 = st.columns(5)
    sc1.metric("Dept Score", sim_ds)
    sc2.metric("Month Score", sim_ms)
    sc3.metric("Hour Score", sim_hs)
    sc4.metric("Total", sim_total)
    sc5.metric("Priority", sim_prio)

    prio_bg = {"High":"rgba(255,77,77,.08)","Medium":"rgba(255,169,77,.08)","Low":"rgba(105,219,124,.08)"}[sim_prio]
    st.markdown(f"""
    <div style='background:{prio_bg};border:1px solid {prio_color};border-radius:10px;
                padding:.8rem 1.2rem;margin-top:-.5rem;'>
      <span style='color:{prio_color};font-family:Syne,sans-serif;font-weight:700;font-size:1.2rem;'>
        ● {sim_prio} Priority — Score {sim_total}/9
      </span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════
#  PAGE: WORKERS
# ══════════════════════════════════════════
elif "Workers" in page:
    st.markdown("## 👷 Worker Directory")

    if not models_loaded:
        st.warning("Employees data not loaded.")
        st.stop()

    st.markdown(f"<small style='color:#8b949e;'>{len(employees)} workers across {employees['department'].nunique()} departments</small>", unsafe_allow_html=True)

    # Filters
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        depts = ["All"] + sorted(employees["department"].unique().tolist())
        sel_dept = st.selectbox("Filter by Department", depts)
    with fcol2:
        roles = ["All"] + sorted(employees["role"].unique().tolist())
        sel_role = st.selectbox("Filter by Role", roles)
    with fcol3:
        statuses = ["All"] + sorted(employees["status"].unique().tolist())
        sel_status = st.selectbox("Filter by Status", statuses)

    filtered = employees
    if sel_dept != "All":
        filtered = filtered[filtered["department"] == sel_dept]
    if sel_role != "All":
        filtered = filtered[filtered["role"] == sel_role]
    if sel_status != "All":
        filtered = filtered[filtered["status"] == sel_status]

    search = st.text_input("Search worker name", placeholder="e.g. Karan")
    if search:
        filtered = filtered[filtered["name"].str.contains(search, case=False, na=False)]

    st.markdown(f"<small style='color:#8b949e;'>Showing {len(filtered)} workers</small>", unsafe_allow_html=True)

    # Display in grid
    cols = st.columns(3)
    for i, (_, row) in enumerate(filtered.iterrows()):
        with cols[i % 3]:
            render_worker_card(row)

    # Status breakdown
    st.markdown("---")
    st.markdown("### Status Breakdown")
    sc = filtered["status"].value_counts().reset_index()
    sc.columns = ["Status","Count"]
    fig_w = px.bar(sc, x="Status", y="Count", color="Count",
                   color_continuous_scale=["#0a2332","#00e5ff"],
                   title="Workers by Status (filtered view)")
    fig_w.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3",family="DM Sans"),
        title_font=dict(family="Syne",size=15),
        xaxis_tickangle=-30, coloraxis_showscale=False,
        margin=dict(l=10,r=10,t=40,b=10),
    )
    st.plotly_chart(fig_w, use_container_width=True)

# ══════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════
elif "About" in page:
    st.markdown("## ℹ️ About SUCAS")

    st.markdown("""
    <div class="ticket" style="margin-bottom:1rem;">
      <div style="font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;color:#00e5ff;margin-bottom:.8rem;">
        Project Overview
      </div>
      <div style="font-size:.92rem;line-height:1.9;color:#c9d1d9;">
        <b>SUCAS</b> (Smart Urban Complaint Analysis System) is an AI-powered civic technology platform
        that automates the routing and prioritisation of public complaints. Citizens describe their issue
        in plain text — either via a form or the AI chat assistant — and a Random Forest classifier trained
        on historical BBMP complaint data identifies the responsible department, a three-factor scoring
        engine computes urgency, and the best-available worker is automatically dispatched.
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="ticket">
          <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#00e5ff;margin-bottom:.8rem;">
            🤖 ML Pipeline
          </div>
          <div style="font-size:.85rem;line-height:1.9;color:#c9d1d9;">
            • <b>Model:</b> Random Forest (sklearn)<br>
            • <b>Features:</b> TF-IDF vectorisation on cleaned complaint text<br>
            • <b>Preprocessing:</b> Stopword removal, WordNet lemmatisation<br>
            • <b>Classes:</b> 31 urban departments<br>
            • <b>Artefacts:</b> rf_model.pkl, tfidf_vectorizer.pkl<br>
            • <b>Chat layer:</b> LLM (HuggingFace endpoint) collects complaint details conversationally, called live — not pickled
          </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="ticket">
          <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#a8ff3e;margin-bottom:.8rem;">
            ⚡ Priority Engine
          </div>
          <div style="font-size:.85rem;line-height:1.9;color:#c9d1d9;">
            • <b>Dept Score (1–3):</b> Pre-computed from historical frequency<br>
            • <b>Month Score (1–3):</b> Seasonal urgency mapping<br>
            • <b>Hour Score (1–3):</b> Night-time complaints escalated<br>
            • <b>Total Score:</b> Sum → High / Medium / Low threshold<br>
            • <b>Worker Assignment:</b> Role-matched from Employees.csv
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ticket" style="margin-top:1rem;">
      <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#ff6b35;margin-bottom:.8rem;">
        🔗 Data Sources & References
      </div>
      <div style="font-size:.85rem;line-height:2;color:#c9d1d9;">
        • <a href="https://bbmp.gov.in" target="_blank" style="color:#00e5ff;">BBMP Official Portal</a> — Bruhat Bengaluru Mahanagara Palike<br>
        • <a href="https://data.gov.in" target="_blank" style="color:#00e5ff;">data.gov.in</a> — India Open Government Data Platform<br>
        • <a href="https://scikit-learn.org" target="_blank" style="color:#00e5ff;">scikit-learn</a> — Machine Learning library<br>
        • <a href="https://www.nltk.org" target="_blank" style="color:#00e5ff;">NLTK</a> — Natural Language Toolkit<br>
        • <a href="https://streamlit.io" target="_blank" style="color:#00e5ff;">Streamlit</a> — App framework<br>
        • <a href="https://plotly.com" target="_blank" style="color:#00e5ff;">Plotly</a> — Interactive visualisations
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Tech stack pills
    st.markdown("<br>**Tech Stack**", unsafe_allow_html=True)
    techs = ["Python 3.x","Streamlit","scikit-learn","NLTK","Plotly","pandas","joblib","LangGraph","LangChain"]
    pills_html = "".join([f'<span class="step-pill"><span class="step-num">✓</span>{t}</span>' for t in techs])
    st.markdown(pills_html, unsafe_allow_html=True)

# ──────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;color:#30363d;font-size:.78rem;'>
  SUCAS · Smart Urban Complaint Analysis System ·
  <a href="https://bbmp.gov.in" target="_blank" style="color:#30363d;">BBMP</a> ·
  Built with Streamlit & scikit-learn
</div>
""", unsafe_allow_html=True)
