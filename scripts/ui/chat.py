import streamlit as st
import requests

def render_chat_tab(API_BASE_URL, api_headers, top_k, use_reranker):
    st.title("Enterprise RAG Assistant")
    if not st.session_state.api_key:
        st.warning("Please generate an API key in the sidebar.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        chat_container = st.container()

        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander(f"View Sources ({len(message['sources'])})"):
                            st.caption("Scores represent relative retrieval rank via RRF fusion, not raw semantic similarity.")
                            for idx, src in enumerate(message["sources"]):
                                st.markdown(f"**{idx+1}. [{src.get('section') or 'Link'}]({src['url']})**")
                                st.caption(f"Score: {src['similarity_score']:.3f}")
                                if src.get("chunk_preview"):
                                    st.markdown(f"> {src['chunk_preview']}")

        if prompt := st.chat_input("E.g., What is Cloud Run?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                    payload = {"query": prompt, "top_k": top_k, "use_reranker": use_reranker, "history": history, "evaluate_faithfulness": True}

                try:
                    response = requests.post(f"{API_BASE_URL}/query", json=payload, headers=api_headers, timeout=120)
                    if response.status_code == 200:
                        data = response.json()
                        full_response = data["answer"]
                        sources = data.get("sources", [])

                        message_placeholder.markdown(full_response)

                        with st.expander(f"🧠 RAG Reasoning & Sources ({len(sources)})"):
                            score = data.get("faithfulness_score")
                            reasoning = data.get("faithfulness_reasoning")
                            if score is not None:
                                color = "green" if score >= 0.8 else "orange" if score >= 0.6 else "red"
                                emoji = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "🚫"
                                label = "High" if score >= 0.8 else "Partial" if score >= 0.6 else "Low"
                                st.markdown(f"**{emoji} Faithfulness: :{color}[{score:.0%} ({label})]**")
                                if reasoning:
                                    st.caption(reasoning)
                            
                            latency = data.get("latency_ms", 0)
                            breakdown = data.get("latency_breakdown", {})
                            st.markdown(f"**Total Latency:** {latency:.1f}ms")
                            if breakdown:
                                st.markdown(f"- Retrieval: {breakdown.get('retrieval', 0):.1f}ms")
                                st.markdown(f"- Generation: {breakdown.get('generation', 0):.1f}ms")
                            
                            st.markdown("---")
                            if sources:
                                st.caption("Scores represent relative retrieval rank via RRF fusion, not raw semantic similarity.")
                                for idx, src in enumerate(sources):
                                    label = src.get("section") or "Source"
                                    sim = src["similarity_score"]
                                    st.markdown(f"**{idx+1}. [{label}]({src['url']})**")
                                    st.caption(f"Relevance: {sim:.3f}")
                                    if src.get("chunk_preview"):
                                        st.markdown(f"> {src['chunk_preview']}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "sources": sources,
                        })
                    else:
                        error_msg = f"Error {response.status_code}: {response.text[:200]}"
                        message_placeholder.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"Failed to connect to backend: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
