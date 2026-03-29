import streamlit as st
import uuid
import os
import certifi
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv

# --- LOAD ENVIRONMENT VARIABLES ---
# 1. Get the directory where main.py is located (Frontend folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Go up one level to the main project folder (RAG-e-Qanoon)
root_dir = os.path.dirname(current_dir)
# 3. Point directly to the .env file
env_path = os.path.join(root_dir, '.env')

# Load the .env file from that specific path
load_dotenv(dotenv_path=env_path)

# Fetch the URI
MONGO_URL = os.getenv("MONGO_URL")

# Print to terminal to verify (Remove or comment this out before final submission for security!)
print(f"DEBUG - Loaded MONGO_URI: {MONGO_URL}")

# --- MONGODB CONNECTION ---
@st.cache_resource # This ensures Streamlit only connects once to save resources
def init_connection():
    if not MONGO_URL:
        raise ValueError("MONGO_URI not found. Check your .env file and path.")
    return MongoClient(MONGO_URL, tlsCAFile=certifi.where())

client = init_connection()
db = client["UrduLegalAI"]          # Name of your database
collection = db["chat_histories"]   # Name of the collection (table)

# --- HELPER FUNCTIONS FOR MEMORY ---
def get_all_past_chats():
    # Grabs all past chats from DB so we can put them in the sidebar
    cursor = collection.find({}, {"chat_id": 1, "messages": 1})
    return {doc["chat_id"]: doc["messages"] for doc in cursor}

def load_chat(chat_id):
    chat = collection.find_one({"chat_id": chat_id})
    return chat["messages"] if chat else []

def save_chat(chat_id, messages):
    collection.update_one(
        {"chat_id": chat_id}, 
        {"$set": {"messages": messages}}, 
        upsert=True
    )

# --- SESSION STATE (With Database Sync) ---
# 1. Pull everything from MongoDB when the app first loads
if "db_synced" not in st.session_state:
    st.session_state.chat_sessions = get_all_past_chats()
    st.session_state.db_synced = True

# 2. Setup standard variables
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chip_prompt" not in st.session_state:
    st.session_state.chip_prompt = None

# 3. If no chat is active, pick the most recent one or start a new one
if not st.session_state.current_chat_id:
    if len(st.session_state.chat_sessions) > 0:
        # Load the last active chat
        st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[-1]
    else:
        # Create a brand new one
        init_id = str(uuid.uuid4())
        st.session_state.chat_sessions[init_id] = []
        st.session_state.current_chat_id = init_id
# --- PAGE CONFIG ---
st.set_page_config(
    page_title="پاکستان قانونی AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL CSS ---
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

/* ── Chip buttons (Streamlit buttons styled as chips) ── */
.chips-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0;
    width: 100%;
    max-width: 680px;
    margin: 0 auto;
}
/* Target the chip button area specifically via a wrapper class */
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

/* ── Chat input strip — force cream background on every layer ── */
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

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(1,65,28,0.15); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "db_synced" not in st.session_state:
    st.session_state.chat_sessions = get_all_past_chats()
    st.session_state.db_synced = True

# 2. Setup standard variables
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "chip_prompt" not in st.session_state:
    st.session_state.chip_prompt = None

# 3. If no chat is active, pick the most recent one or start a new one
if not st.session_state.current_chat_id:
    if len(st.session_state.chat_sessions) > 0:
        # Load the last active chat
        st.session_state.current_chat_id = list(st.session_state.chat_sessions.keys())[-1]
    else:
        # Create a brand new one
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
    "🏛️ آئین پاکستان کے بنیادی حقوق",
    "⚖️ ایف آئی آر کیسے درج کریں؟",
    "🏠 کرایہ دار کے حقوق",
    "👨‍👩‍👧 خاندانی قانون",
    "💼 ملازمت کے حقوق",
    "📄 دستاویزات اور تصدیق",
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

    # Clickable chip buttons rendered in a flex row via custom class
    st.markdown('<div class="chip-btn-row">', unsafe_allow_html=True)
    cols = st.columns(len(CHIPS))
    for i, chip in enumerate(CHIPS):
        with cols[i]:
            if st.button(chip, key=f"chip_{i}"):
                st.session_state.chip_prompt = chip
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    for message in current_messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="msg-row-user">
                <div class="user-bubble">{message["content"]}</div>
                <div class="av av-user">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row-bot">
                <div class="av av-bot">⚖️</div>
                <div class="bot-bubble">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

# --- # --- Handle chip click (fires after rerun) ---
if st.session_state.chip_prompt:
    prompt = st.session_state.chip_prompt
    st.session_state.chip_prompt = None
    
    # We grab the memory from Streamlit's synced state
    current_messages = st.session_state.chat_sessions[st.session_state.current_chat_id]
    
    current_messages.append({"role": "user", "content": prompt})
    current_messages.append({"role": "assistant", "content": "جواب یہاں آئے گا — آپ کا RAG سسٹم یہاں جڑے گا۔"})
    
    # SAVE IT!
    save_chat(st.session_state.current_chat_id, current_messages)
    st.rerun()
# --- CHAT INPUT ---
if prompt := st.chat_input("اپنا قانونی سوال یہاں لکھیں..."):
    current_messages.append({"role": "user", "content": prompt})
    conversation_history = ""
    for msg in current_messages[-5:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation_history += f"{role}: {msg['content']}\n"
    # TODO: Pass 'conversation_history' + 'prompt' to your RAG pipeline
    current_messages.append({"role": "assistant", "content": "جواب یہاں آئے گا — آپ کا RAG سسٹم یہاں جڑے گا۔"})
    save_chat(st.session_state.current_chat_id, current_messages)
    st.rerun()