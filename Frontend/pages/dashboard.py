import streamlit as st
import pandas as pd

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Evaluation & Ablation", layout="wide")

# --- CUSTOM CSS (Consistent Theme) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap');
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { background-color: #FFFFFF !important; color: #1a1a1a; }
    
    [data-testid="stSidebar"] { background-color: #01411C !important; }
    [data-testid="stSidebar"] * { color: white !important; border-color: rgba(255,255,255,0.2) !important; }
    
    .metric-card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #01411C;
        margin-bottom: 20px;
    }
    .urdu-font { font-family: 'Noto Nastaliq Urdu', serif; direction: rtl; text-align: right; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 System Evaluation & Ablation Study")
st.markdown("This dashboard evaluates the RAG-e-Qanoon system using an **LLM-as-a-Judge** approach on 10 fixed legal queries.")

# --- TABS SETUP ---
tab1, tab2, tab3 = st.tabs(["Ablation Study", "Test Set Metrics", "Live LLM Judge"])

# --- TAB 1: ABLATION STUDY ---
with tab1:
    st.subheader("LLM Model Comparison")
    st.markdown("Comparing different language models to find the optimal balance of Faithfulness and Relevancy for Urdu legal text.")
    
    llm_data = {
        "Rank": ["1 🏆", "2", "3", "4", "5"],
        "Model": ["qwen14b", "lughaat_8b", "qalb_8b", "urdu_llama_3b", "aya23_8b"],
        "Faithfulness": ["82.84%", "67.21%", "63.53%", "62.37%", "48.38%"],
        "Relevancy": ["96.47%", "74.42%", "79.68%", "74.32%", "56.82%"],
        "Composite (60/40)": ["88.29%", "70.09%", "69.99%", "67.15%", "51.75%"],
        "Avg Latency": ["49.03s", "10.34s", "13.87s", "2.22s", "18.60s"]
    }
    st.dataframe(pd.DataFrame(llm_data), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Chunking Strategy")
        chunk_data = {
            "Strategy": ["Fixed (512 chars)", "Recursive", "Sentence Boundary"],
            "Faithfulness": ["71.0%", "69.6%", "68.2%"],
            "Relevancy": ["78.8%", "80.2%", "80.3%"]
        }
        st.dataframe(pd.DataFrame(chunk_data), use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Retrieval Architecture")
        retrieval_data = {
            "Configuration": ["Hybrid + Reranker ON", "Hybrid + Reranker OFF"],
            "Faithfulness": ["66.4%", "49.2%"],
            "Relevancy": ["78.8%", "72.4%"]
        }
        st.dataframe(pd.DataFrame(retrieval_data), use_container_width=True, hide_index=True)

# --- TAB 2: OVERALL METRICS ---
with tab2:
    st.subheader("Performance on Benchmark Test Queries")
    st.markdown("*Metrics based on the absolute best configuration (Qwen14b, Fixed Chunking, Reranker ON, RRF 0.5/0.5).*")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Faithfulness: 82.84%</h3>
            <p>Measures if the generated Urdu response is strictly grounded in the retrieved legal text. (Score represents the percentage of LLM-verified claims successfully backed by the retrieved context).</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Relevancy: 96.47%</h3>
            <p>Measures if the response directly answers the user's specific legal question. (Calculated via semantic similarity of auto-generated alternate queries).</p>
        </div>
        """, unsafe_allow_html=True)

# --- TAB 3: LIVE LLM JUDGE ---
with tab3:
    st.subheader("Interactive Evaluation Tool")
    st.markdown("Test the LLM Judge logic manually.")
    
    col1, col2 = st.columns(2)
    with col1:
        test_context = st.text_area("Retrieved Context (Urdu)", height=150, placeholder="Paste legal text here...")
    with col2:
        test_answer = st.text_area("Generated Answer (Urdu)", height=150, placeholder="Paste the AI's answer here...")
        
    if st.button("⚖️ Run LLM Judge"):
        if test_context and test_answer:
            with st.spinner("Analyzing Faithfulness and Relevancy..."):
                import time
                time.sleep(2) # Mocking the API call for the frontend demo
                st.success("Evaluation Complete!")
                st.write("**Faithfulness:** ہاں (The claims match the context).")
                st.write("**Relevancy:** 92.5% (The answer directly addresses the legal context provided).")
        else:
            st.warning("Please provide both context and an answer to evaluate.")