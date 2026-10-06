import streamlit as st

from client import ApiError, NexusClient
from render import render_sources


def render_playground_tab(client: NexusClient, top_k: int):
    st.title("Retrieval Playground")
    st.markdown("Compare retrieval with and without the workspace's reranker. **No answer is generated.**")

    query = st.text_input("Enter a test query:")
    if not (st.button("Run comparison") and query):
        return
    with st.spinner("Retrieving..."):
        try:
            data = client.compare_retrieval(query, top_k)
        except ApiError as e:
            st.error(str(e))
            return

    reranker = data.get("reranker") or "no reranker"
    for column, title, key in zip(st.columns(2), ("First-stage order", f"Reranked ({reranker})"), ("baseline", "reranked")):
        with column:
            st.subheader(title)
            st.caption(f"Latency: {data.get(key + '_latency_ms', 0):.0f} ms")
            render_sources(data.get(key, []), limit=300)
    if data.get("degraded"):
        st.warning("Degraded: " + "; ".join(data["degraded"]))
