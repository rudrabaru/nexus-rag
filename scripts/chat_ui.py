import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv(override=True)

from ui.chat import render_chat_tab
from ui.dashboard import render_dashboard_tab
from ui.documents import render_documents_tab
from ui.playground import render_playground_tab

st.set_page_config(page_title="Nexus RAG", page_icon="🤖", layout="wide")

st.markdown(
    """
<style>
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stChatMessage.user {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-left: 4px solid #3b82f6;
    }
    .stChatMessage.assistant {
        background: linear-gradient(145deg, #0f172a, #020617);
        border-left: 4px solid #10b981;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .streamlit-expanderHeader {
        font-size: 0.95em;
        font-weight: 600;
        color: #94a3b8;
        background-color: rgba(15, 23, 42, 0.5);
        border-radius: 6px;
    }
    .streamlit-expanderContent {
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: none;
        border-bottom-left-radius: 6px;
        border-bottom-right-radius: 6px;
        padding-top: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"

if "api_key" not in st.session_state:
    st.session_state.api_key = "demo" if DEMO_MODE else ""

# Sidebar Settings
with st.sidebar:
    st.header("Nexus RAG Configuration")
    if not DEMO_MODE:
        st.subheader("Workspace Authentication")
        if st.session_state.api_key:
            st.success("Authenticated with your private workspace")
            
            if st.session_state.get("newly_generated_key"):
                st.warning("⚠️ Please save your new API key now! You will need it to login later.")
                st.code(st.session_state.newly_generated_key, language=None)
                if st.button("I have saved my key", use_container_width=True):
                    st.session_state.newly_generated_key = None
                    st.rerun()
            
            if st.button("Sign Out", use_container_width=True):
                st.session_state.api_key = ""
                st.session_state.newly_generated_key = None
                st.rerun()
        else:
            st.info("Enter an existing API key or generate a new one to get your own private document workspace.")
            
            existing_key = st.text_input("Existing API Key", type="password", placeholder="Paste your key here...")
            if st.button("Login", use_container_width=True) and existing_key:
                st.session_state.api_key = existing_key
                st.rerun()
                
            st.markdown("---")
            
            if st.button("Generate New API Key", use_container_width=True):
                try:
                    res = requests.post(f"{API_BASE_URL}/register", timeout=10)
                    if res.status_code == 200:
                        new_key = res.json()["api_key"]
                        st.session_state.api_key = new_key
                        st.session_state.newly_generated_key = new_key
                        st.rerun()
                    else:
                        st.error(f"Failed to generate API Key: {res.status_code} - {res.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")
    else:
        st.info("Running in **DEMO MODE**. Multi-tenancy is disabled. All users share the 'demo' workspace.")

    api_headers = {}
    if st.session_state.api_key:
        api_headers["X-API-Key"] = st.session_state.api_key
        api_headers["RAG-API-KEY"] = st.session_state.api_key
        
    st.markdown("---")
    st.subheader("Query Parameters")
    top_k = st.slider("Sources (Top K)", min_value=1, max_value=10, value=5)
    use_reranker = st.toggle("Use Cross-Encoder Reranker", value=False)
    stream_response = st.toggle("Stream LLM Response", value=True)
    
    st.markdown("---")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

tab_query, tab_playground, tab_dashboard, tab_docs = st.tabs(["💬 Query Assistant", "🔬 Retrieval Playground", "📊 Pipeline Dashboard", "📄 Document Management"])

with tab_query:
    render_chat_tab(API_BASE_URL, api_headers, top_k, use_reranker, stream_response)

with tab_playground:
    render_playground_tab(API_BASE_URL, api_headers, top_k)

with tab_dashboard:
    render_dashboard_tab(API_BASE_URL, api_headers)

with tab_docs:
    render_documents_tab(API_BASE_URL, api_headers)
