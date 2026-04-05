import streamlit as st
import uuid
import os
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv
import sys
import time
import html
# --- PAGE CONFIG ---
st.set_page_config(
    page_title="پاکستان قانونی AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DYNAMIC PATH RESOLUTION ---
# Get the absolute path of the directory containing main.py (Frontend)
frontend_dir = os.path.dirname(os.path.abspath(__file__))
# Get the root directory (RAG-e-Qanoon)
root_dir = os.path.dirname(frontend_dir)
# 2. ADD ROOT TO SYS.PATH
# Now Python can "see" sibling folders like rag_backend
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 3. NOW PERFORM THE IMPORT
# Since the root is in the path, Python can find the rag_backend folder
from rag_backend.rag_pipeline import RAGPipeline

env_path = os.path.join(root_dir, '.env')
load_dotenv(dotenv_path=env_path)

# 2. Fetch the URI
MONGO_URL = os.getenv("MONGO_URL")

# --- INITIALIZE RAG BACKEND ---
@st.cache_resource(show_spinner=False)
def init_rag():
    rag = RAGPipeline(chunking_strategy="fixed")
    # Correct path to your JSON in the scrapper folder
    json_path = os.path.join(root_dir, "scrapper", "cleaned_ocr_output.json")
    if os.path.exists(json_path):
        rag.load_bm25_from_json(json_path)
    else:
        st.warning(f"JSON not found at: {json_path}")
    return rag

# ... [Place the Import & Path Fix code from Step 1 here] ...

# --- SESSION STATE ---
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"
# ... [Keep your existing session state logic for chat_sessions etc.] ...

# --- SIDEBAR ---

# --- MAIN CONTENT AREA ---
# if st.session_state.current_page == "dashboard":
#     # --- PASTE CONTENT OF dashboard.py HERE ---
#     st.title("📊 System Evaluation & Ablation Study")
#     # ... (Tables, metrics from your dashboard.py) ...

# else:
#     # --- YOUR ORIGINAL CHAT UI CODE ---
#     # Welcome screen, Chips, Chat bubbles, and Chat Input
#     # ...
    
#     if prompt := st.chat_input("اپنا قانونی سوال یہاں لکھیں..."):
#         # The logic we built to call rag_pipeline.query()
#         with st.spinner("جواب تیار کیا جا رہا ہے..."):
#             result = rag_pipeline.query(prompt, run_evaluation=False)
#             # append result to messages and st.rerun()
# --- MONGODB CONNECTION ---
@st.cache_resource 
def init_connection():
    if not MONGO_URL:
        st.error("MONGO_URL not found in .env file.")
        return None
    return MongoClient(MONGO_URL, tlsCAFile=certifi.where())

client = init_connection()
if client:
    db = client["UrduLegalAI"]          
    collection = db["chat_histories"]   

# --- HELPER FUNCTIONS FOR MEMORY ---
def get_all_past_chats():
    if not client: return {}
    cursor = collection.find({}, {"chat_id": 1, "messages": 1})
    return {doc["chat_id"]: doc["messages"] for doc in cursor}

def save_chat(chat_id, messages):
    if not client: return
    collection.update_one(
        {"chat_id": chat_id}, 
        {"$set": {"messages": messages}}, 
        upsert=True
    )
def render_user_bubble(text: str):
    safe_text = html.escape(text).replace("\n", "<br>")
    st.markdown(f"""
    <div class="msg-row-user animate-in">
        <div class="user-bubble">{safe_text}</div>
        <div class="av av-user">👤</div>
    </div>
    """, unsafe_allow_html=True)

def render_typing_indicator(container):
    container.markdown("""
    <div class="typing-wrap">
        <div class="av av-bot">⚖️</div>
        <div class="typing-bubble">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def stream_assistant_bubble(container, full_text: str, delay: float = 0.012):
    words = full_text.split()
    shown = ""

    for word in words:
        shown = (shown + " " + word).strip()
        safe_text = html.escape(shown).replace("\n", "<br>")
        container.markdown(f"""
        <div class="msg-row-bot animate-in">
            <div class="av av-bot">⚖️</div>
            <div class="bot-bubble">{safe_text}</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(delay)


# --- GLOBAL CSS (YOUR EXACT CSS) ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600;700&family=Cormorant+Garamond:wght@500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
header { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

:root {
    --green-dark:   #01411C;
    --green-mid:    #025a27;
    --green-light:  #e8f5ed;
    --green-accent: #2e7d52;
    --gold:         #c9a84c;
    --cream:        #faf8f3;
    --white:        #ffffff;
    --gray-200:     #e5e7eb;
    --gray-400:     #9ca3af;
    --gray-500:     #6b7280;
    --gray-700:     #374151;
    --gray-900:     #111827;
    --shadow-sm:    0 1px 4px rgba(0,0,0,0.07);
    --shadow-md:    0 4px 20px rgba(0,0,0,0.09);
}

.stApp { background-color: var(--cream) !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #01411C 0%, #012a12 100%) !important;
    border-right: 1px solid rgba(201,168,76,0.15) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.88) !important; }

[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(201,168,76,0.2) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.85) !important;
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl !important;
    padding: 10px 14px !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
    text-align: right !important;
    margin-bottom: 5px !important;
    font-size: 0.9rem !important;
    line-height: 1.8 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(201,168,76,0.12) !important;
    border-color: rgba(201,168,76,0.45) !important;
}
[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: linear-gradient(135deg, var(--gold) 0%, #b8922e 100%) !important;
    border: none !important;
    color: #01411C !important;
    font-weight: 700 !important;
    box-shadow: 0 3px 12px rgba(201,168,76,0.35) !important;
    margin-bottom: 20px !important;
}/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #d7e7dc !important;
    border-radius: 16px !important;
    background: #ffffff !important;
    margin: 0 clamp(20px, 6vw, 100px) 14px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}

[data-testid="stExpander"] details {
    background: #ffffff !important;
}

[data-testid="stExpander"] summary {
    background: #f3f8f5 !important;
    color: var(--green-dark) !important;
    border-bottom: 1px solid #d7e7dc !important;
    padding: 12px 16px !important;
    font-weight: 600 !important;
    border-radius: 16px 16px 0 0 !important;
}

[data-testid="stExpander"] summary:hover {
    background: #ebf4ee !important;
}

/* Score cards */
.score-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 8px 0 16px;
}

.score-card {
    background: #f7fbf8;
    border: 1px solid #d8e9dd;
    border-radius: 14px;
    padding: 12px 16px;
    min-width: 170px;
    box-shadow: var(--shadow-sm);
}

.score-label {
    font-size: 0.82rem;
    color: var(--gray-500);
    margin-bottom: 4px;
}

.score-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--green-dark);
}

/* Chunk cards */
.context-card {
    background: #f8fcf9;
    border: 1px solid #d8e9dd;
    border-right: 4px solid var(--green-accent);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: var(--shadow-sm);
    color: var(--gray-900) !important;
    direction: rtl;
    text-align: right;
    line-height: 2.1;
}

.context-title {
    font-weight: 700;
    color: var(--green-dark);
    margin-bottom: 6px;
}

.context-score {
    display: inline-block;
    background: #e8f5ed;
    color: var(--green-dark);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.8rem;
    margin-bottom: 10px;
}
.context-card {
    background: #f7fbf8;
    border: 1px solid #d8e9dd;
    border-right: 4px solid var(--green-accent);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: var(--shadow-sm);
    direction: rtl;
    text-align: right;
}

.context-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.9rem;
    color: var(--green-dark);
    margin-bottom: 8px;
    font-weight: 700;
}

.context-score {
    display: inline-block;
    background: var(--green-light);
    color: var(--green-dark);
    border: 1px solid rgba(1,65,28,0.12);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.8rem;
    margin-bottom: 10px;
}

.score-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 14px;
}

.score-card {
    background: #f7fbf8;
    border: 1px solid #d8e9dd;
    border-radius: 14px;
    padding: 12px 16px;
    min-width: 180px;
    box-shadow: var(--shadow-sm);
}

.score-label {
    font-size: 0.82rem;
    color: var(--gray-500);
    margin-bottom: 4px;
}

.score-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--green-dark);
}
.sidebar-header {
    padding: 28px 18px 18px;
    border-bottom: 1px solid rgba(201,168,76,0.18);
    margin-bottom: 18px;
    text-align: center;
}
.sidebar-appname {
    font-family: 'Noto Nastaliq Urdu', serif;
    font-size: 1.3rem;
    color: white !important;
    direction: rtl;
    line-height: 1.7;
    margin-bottom: 4px;
}
.sidebar-tag {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.68rem;
    color: var(--gold) !important;
    letter-spacing: 2.5px;
    text-transform: uppercase;
}
.history-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35) !important;
    padding: 0 4px;
    margin-bottom: 8px;
    display: block;
}

/* ── Welcome ── */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 72vh;
    text-align: center;
}
.welcome-eyebrow {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--green-accent);
    background: var(--green-light);
    border: 1px solid rgba(1,65,28,0.15);
    border-radius: 50px;
    padding: 5px 18px;
    margin-bottom: 22px;
    display: inline-block;
}
.welcome-title {
    font-family: 'Noto Nastaliq Urdu', serif;
    font-size: clamp(1.9rem, 3.5vw, 2.8rem);
    color: var(--green-dark);
    direction: rtl;
    line-height: 1.55;
    margin-bottom: 22px;
    font-weight: 700;
}
.welcome-quote-box {
    background: var(--white);
    border-right: 3px solid var(--gold);
    border-radius: 14px;
    padding: 18px 28px;
    max-width: 500px;
    margin-bottom: 36px;
    box-shadow: var(--shadow-sm);
}
.welcome-quote-text {
    font-family: 'Noto Nastaliq Urdu', serif;
    font-size: 1rem;
    color: var(--gray-500);
    direction: rtl;
    line-height: 2.5;
}

/* ── Chip buttons ── */
.chips-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    width: 100%;
    max-width: 680px;
    margin: 0 auto;
}
.chip-btn-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    direction: rtl;
    width: 100%;
}
.chip-btn-row .stButton > button {
    background: var(--white) !important;
    border: 1px solid var(--gray-200) !important;
    border-radius: 50px !important;
    padding: 9px 18px !important;
    font-family: 'Noto Nastaliq Urdu', serif !important;
    font-size: 0.92rem !important;
    color: var(--gray-700) !important;
    box-shadow: var(--shadow-sm) !important;
    direction: rtl !important;
    cursor: pointer !important;
    transition: box-shadow 0.15s, border-color 0.15s, transform 0.1s !important;
    width: auto !important;
    display: inline-block !important;
    margin: 0 !important;
}
.chip-btn-row .stButton > button:hover {
    box-shadow: var(--shadow-md) !important;
    border-color: rgba(1,65,28,0.3) !important;
    transform: translateY(-2px) !important;
    color: var(--green-dark) !important;
}

/* ── Chat messages ── */
.msg-row-user {
    display: flex; justify-content: flex-end;
    align-items: flex-end; gap: 10px; margin: 12px 0;
    padding: 0 clamp(20px, 6vw, 100px);
}
.msg-row-bot {
    display: flex; justify-content: flex-start;
    align-items: flex-end; gap: 10px; margin: 12px 0;
    padding: 0 clamp(20px, 6vw, 100px);
}
.user-bubble {
    background: linear-gradient(135deg, #01411C, #025a27);
    color: #fff; padding: 13px 20px;
    border-radius: 22px 22px 4px 22px;
    max-width: 68%; font-family: 'Noto Nastaliq Urdu', serif;
    font-size: 1rem; line-height: 2.2; direction: rtl; text-align: right;
    box-shadow: 0 4px 14px rgba(1,65,28,0.22); word-wrap: break-word;
}
.bot-bubble {
    background: var(--white); color: var(--gray-900);
    padding: 14px 20px; border-radius: 22px 22px 22px 4px;
    max-width: 72%; font-family: 'Noto Nastaliq Urdu', serif;
    font-size: 1rem; line-height: 2.4; direction: rtl; text-align: right;
    box-shadow: var(--shadow-sm); border: 1px solid var(--gray-200); word-wrap: break-word;
}
.av { width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.95rem; flex-shrink: 0; }
.av-user { background: linear-gradient(135deg, #01411C, #2e7d52); }
.av-bot  { background: linear-gradient(135deg, #c9a84c, #a8873a); }

/* ── Chat input strip ── */
[data-testid="stBottom"] {
    background-color: var(--cream) !important;
    background: var(--cream) !important;
    border-top: 1px solid var(--gray-200) !important;
    padding: 14px clamp(60px, 14vw, 260px) 18px !important;
}
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottom"] section,
[data-testid="stBottom"] .stChatInput {
    background-color: var(--cream) !important;
    background: var(--cream) !important;
}
[data-testid="stChatInputContainer"] {
    background: var(--white) !important;
    background-color: var(--white) !important;
    border: 1.5px solid var(--gray-200) !important;
    border-radius: 30px !important;
    box-shadow: var(--shadow-md) !important;
    padding: 8px 18px !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: var(--green-accent) !important;
    box-shadow: 0 0 0 3px rgba(46,125,82,0.1), var(--shadow-md) !important;
}
[data-testid="stChatInputContainer"] textarea {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    font-size: 1rem !important; direction: rtl !important;
    color: var(--gray-900) !important; line-height: 1.8 !important;
    background: transparent !important;
    background-color: transparent !important;
}
[data-testid="stChatInputContainer"] textarea::placeholder {
    color: var(--gray-400) !important;
    font-family: 'Noto Nastaliq Urdu', serif !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #01411C, #025a27) !important;
    border-radius: 50% !important; border: none !important;
    box-shadow: 0 2px 8px rgba(1,65,28,0.28) !important;
}

/* ── Scrollbar & Expanders ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(1,65,28,0.15); border-radius: 10px; }
[data-testid="stExpander"] { background-color: var(--white); border-radius: 10px; border: 1px solid var(--gray-200); margin: 0 clamp(20px, 6vw, 100px) 12px; }
/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes blinkDots {
    0%, 20%   { opacity: 0.25; transform: translateY(0); }
    50%       { opacity: 1; transform: translateY(-2px); }
    100%      { opacity: 0.25; transform: translateY(0); }
}

.animate-in {
    animation: fadeUp 0.35s ease-out;
}

.typing-wrap {
    display: flex;
    justify-content: flex-start;
    align-items: flex-end;
    gap: 10px;
    margin: 12px 0;
    padding: 0 clamp(20px, 6vw, 100px);
    animation: fadeUp 0.25s ease-out;
}

.typing-bubble {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: 22px 22px 22px 4px;
    padding: 14px 18px;
    box-shadow: var(--shadow-sm);
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 72px;
}

.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green-accent);
    display: inline-block;
    animation: blinkDots 1.2s infinite ease-in-out;
}

.typing-dot:nth-child(2) { animation-delay: 0.15s; }
.typing-dot:nth-child(3) { animation-delay: 0.3s; }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE PIPELINE ---
rag_pipeline = init_rag()

# --- SESSION STATE ---
if "db_synced" not in st.session_state:
    st.session_state.chat_sessions = get_all_past_chats()
    st.session_state.db_synced = True

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chip_prompt" not in st.session_state:
    st.session_state.chip_prompt = None

if not st.session_state.current_chat_id:
    if len(st.session_state.chat_sessions) > 0:
        st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[-1]
    else:
        init_id = str(uuid.uuid4())
        st.session_state.chat_sessions[init_id] = []
        st.session_state.current_chat_id = init_id

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <img src="https://pakistancode.gov.pk/english/images/pakcodelogo.png"
             style="width:70px; margin:0 auto 15px; display:block; filter:drop-shadow(0px 2px 4px rgba(0,0,0,0.2));">
        <div class="sidebar-appname">پاکستان قانونی AI</div>
        <div class="sidebar-tag">Pakistan Legal · AI</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕  نیا سوال", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = []
        st.session_state.current_chat_id = new_id
        st.rerun()

    sessions_except_current = [
        (cid, msgs) for cid, msgs in st.session_state.chat_sessions.items()
        if cid != st.session_state.current_chat_id
    ]
    if sessions_except_current:
        st.markdown('<span class="history-label">Recent</span>', unsafe_allow_html=True)
        for chat_id, msgs in reversed(sessions_except_current):
            label = next(
                (m["content"][:30] + "…" for m in msgs if m["role"] == "user"),
                f"گفتگو {chat_id[:6]}…"
            )
            if st.button(label, key=chat_id, use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.rerun()

# --- MAIN AREA ---
current_messages = st.session_state.chat_sessions[st.session_state.current_chat_id]

CHIPS = [
    "پاکستان کا ریاستی مذہب کیا ہے؟",
    "پاکستان میں قومی اسمبلی کی مدت کتنی ہے؟",
    "اگر کسی شخص کو غلط طریقے سے گرفتار کیا جائے تو اسے کیا حقوق حاصل ہیں؟",
    "انسانی عزت کے بارے میں پاکستان کے آئین میں کیا کہا گیا ہے؟",
    "وزیر اعظم کو برطرف کرنے کے لیے کیا طریقہ کار ہے؟",
    "پاکستان میں جنگ کی صورت میں کیا ہوتا ہے؟ حکومت کو کیا اختیارات مل جاتے ہیں؟",
    "عدالت عظمیٰ براہ راست کوئی مقدمہ سن سکتی ہے یا پہلے نچلی عدالت میں جانا ضروری ہے؟",
    "وفاقی شرعی عدالت کیا کام کرتی ہے؟",
    "پاکستان کا ریاستی مذہب کیا ہے اور یہ آئین میں کہاں لکھا ہے؟",
    "عدالت عظمیٰ کے جج کب ریٹائر ہوتے ہیں؟"
]

if len(current_messages) == 0:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-eyebrow">Pakistan Legal AI &nbsp;•&nbsp; پاکستانی قانون</div>
        <div class="welcome-title">آپ کے قانونی سوالات کا جواب</div>
        <div class="welcome-quote-box">
            <div class="welcome-quote-text">
                "اپنے حقوق جاننا آپ کی طاقت ہے  اور ان کی حفاظت آپ کی شہری ذمہ داری ہے"
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chip-btn-row">', unsafe_allow_html=True)
    for i in range(0, len(CHIPS), 2):
        cols = st.columns(2)
        with cols[0]:
            if st.button(CHIPS[i], key=f"chip_{i}", use_container_width=True):
                st.session_state.chip_prompt = CHIPS[i]
                st.rerun()
        if i + 1 < len(CHIPS):
            with cols[1]:
                if st.button(CHIPS[i+1], key=f"chip_{i+1}", use_container_width=True):
                    st.session_state.chip_prompt = CHIPS[i+1]
                    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    for message in current_messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="msg-row-user animate-in">
                <div class="user-bubble">{message["content"]}</div>
                <div class="av av-user">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="msg-row-bot animate-in">
                <div class="av av-bot">⚖️</div>
                <div class="bot-bubble">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show chunks and scores if they exist
            if message.get("chunks"):
                with st.expander("📊 Evaluation Scores & Retrieved Context (تفصیلات)", expanded=False):
                    f_score = message.get("faithfulness", "N/A")
                    r_score = message.get("relevancy", "N/A")

                    st.markdown(f"""
                    <div class="score-row">
                        <div class="score-card">
                            <div class="score-label">Faithfulness</div>
                            <div class="score-value">{f_score}</div>
                        </div>
                        <div class="score-card">
                            <div class="score-label">Relevancy</div>
                            <div class="score-value">{r_score}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    for i, chunk in enumerate(message["chunks"]):
                        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", 0)))
                        chunk_text = html.escape(chunk["text"]).replace("\n", "<br>")
                        st.markdown(f"""
                        <div class="context-card">
                            <div class="context-title">Document {i+1}</div>
                            <div class="context-score">Score: {score:.3f}</div>
                            <div>{chunk_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

# --- EXECUTE RAG PIPELINE FUNCTION ---
def execute_rag(user_input):
    current_messages.append({"role": "user", "content": user_input})
    save_chat(st.session_state.current_chat_id, current_messages)

    # show user message immediately
    render_user_bubble(user_input)

    # assistant typing placeholder
    assistant_placeholder = st.empty()
    render_typing_indicator(assistant_placeholder)

    history_for_rag = [{"role": m["role"], "content": m["content"]} for m in current_messages[:-1]]

    try:
        result = rag_pipeline.query(
            user_query=user_input,
            conversation_history=history_for_rag,
            run_evaluation=True
        )

        answer = result["answer"]
        chunks = result.get("retrieved_chunks", [])

        f_val = result.get("faithfulness", {}).get("score")
        r_val = result.get("relevancy", {}).get("score")
        faithfulness = f"{f_val:.2%}" if isinstance(f_val, float) else "N/A"
        relevancy = f"{r_val:.2%}" if isinstance(r_val, float) else "N/A"

    except Exception as e:
        answer = f"معذرت، ایک تکنیکی خرابی پیش آ گئی ہے۔ برائے مہربانی دوبارہ کوشش کریں۔\n\n(System Error: {str(e)})"
        chunks, faithfulness, relevancy = [], "N/A", "N/A"

    # replace typing indicator with streamed answer
    stream_assistant_bubble(assistant_placeholder, answer, delay=0.01)

    current_messages.append({
        "role": "assistant",
        "content": answer,
        "chunks": chunks,
        "faithfulness": faithfulness,
        "relevancy": relevancy
    })

    save_chat(st.session_state.current_chat_id, current_messages)
    time.sleep(0.2)
    st.rerun()
# --- HANDLE INPUTS ---
if st.session_state.chip_prompt:
    prompt = st.session_state.chip_prompt
    st.session_state.chip_prompt = None
    execute_rag(prompt)

if prompt := st.chat_input("اپنا قانونی سوال یہاں لکھیں..."):
    execute_rag(prompt)