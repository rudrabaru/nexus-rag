"""
The admin UI: a thin Streamlit client of the Nexus API. It holds no retrieval or ingestion logic and
imports nothing from src/, so it can run (and ship) on its own.

    streamlit run admin_ui/app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from chat_tab import render_chat_tab
from client import NexusClient
from dashboard_tab import render_dashboard_tab
from documents_tab import render_documents_tab
from playground_tab import render_playground_tab
from style import PAGE_CSS

MAX_TOP_K = 20  # the API's limit

st.set_page_config(page_title="Nexus RAG", page_icon="🤖", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
st.session_state.setdefault("api_key", "")

with st.sidebar:
    st.header("Nexus RAG")
    st.subheader("Workspace key")
    if st.session_state.api_key:
        st.success("Signed in to your workspace")
        if st.button("Sign out", use_container_width=True):
            st.session_state.api_key = ""
            st.rerun()
    else:
        st.info("Enter the API key your administrator issued for your workspace.")
        key = st.text_input("API key", type="password", placeholder="Paste your key here...")
        if st.button("Sign in", use_container_width=True) and key:
            st.session_state.api_key = key
            st.rerun()

    st.markdown("---")
    st.subheader("Query parameters")
    top_k = st.slider("Sources (top k)", min_value=1, max_value=MAX_TOP_K, value=5)
    use_reranker = st.toggle("Use the workspace's reranker", value=False)
    stream_response = st.toggle("Stream the answer", value=True)
    judge = st.toggle("Check faithfulness (one extra model call per answer)", value=False)

    st.markdown("---")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

client = NexusClient(API_BASE_URL, st.session_state.api_key)
signed_in = bool(st.session_state.api_key)

tab_chat, tab_playground, tab_dashboard, tab_docs = st.tabs(["Chat", "Retrieval playground", "Dashboard", "Documents"])
with tab_chat:
    render_chat_tab(client, signed_in, top_k, use_reranker, stream_response, judge)
with tab_playground:
    render_playground_tab(client, top_k)
with tab_dashboard:
    render_dashboard_tab(client)
with tab_docs:
    render_documents_tab(client)
