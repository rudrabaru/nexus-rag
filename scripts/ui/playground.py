import streamlit as st
import requests

def render_playground_tab(API_BASE_URL, api_headers, top_k):
    st.title("🔬 Retrieval Playground (Dev Tool)")
    st.markdown("Test retrieval quality with and without the Cross-Encoder Reranker. **No LLM generation is performed.**")

    query = st.text_input("Enter a test query:")
    
    if st.button("Run Retrieval Comparison") and query:
        with st.spinner("Running parallel retrieval..."):
            try:
                payload = {"query": query, "top_k": top_k}
                res = requests.post(f"{API_BASE_URL}/query/compare", json=payload, headers=api_headers, timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    baseline = data.get("baseline", [])
                    reranked = data.get("reranked", [])
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(f"Baseline (Vector Search Only)")
                        st.caption(f"Latency: {data.get('baseline_latency_ms', 0):.0f}ms")
                        for idx, src in enumerate(baseline):
                            label = src.get("section") or "Source"
                            sim = src.get("similarity_score", 0)
                            st.markdown(f"**[{idx+1}] [{label}]({src.get('url', '#')})** &nbsp;&nbsp; `<Score: {sim:.3f}>`", unsafe_allow_html=True)
                            preview = src.get("chunk_preview", "").replace('\n', ' ').strip()
                            with st.container(border=True):
                                st.caption(f'"{preview[:300]}..."' if len(preview) > 300 else f'"{preview}"')
                                
                    with col2:
                        st.subheader(f"Reranked (Hybrid + Cross-Encoder)")
                        st.caption(f"Latency: {data.get('reranked_latency_ms', 0):.0f}ms")
                        for idx, src in enumerate(reranked):
                            label = src.get("section") or "Source"
                            sim = src.get("similarity_score", 0)
                            st.markdown(f"**[{idx+1}] [{label}]({src.get('url', '#')})** &nbsp;&nbsp; `<Score: {sim:.3f}>`", unsafe_allow_html=True)
                            preview = src.get("chunk_preview", "").replace('\n', ' ').strip()
                            with st.container(border=True):
                                st.caption(f'"{preview[:300]}..."' if len(preview) > 300 else f'"{preview}"')
                                
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect: {e}")
