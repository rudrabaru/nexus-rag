import streamlit as st
import requests
import json

def render_chat_tab(API_BASE_URL, api_headers, top_k, use_reranker, stream_response):
    st.title("Nexus RAG Assistant")
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
                    if ("sources" in message and message["sources"]) or message.get("faithfulness_data"):
                        with st.expander(f"View Sources ({len(message.get('sources', []))})"):
                            if message.get("faithfulness_data"):
                                f_data = message["faithfulness_data"]
                                score = f_data.get("score")
                                reasoning = f_data.get("reasoning")
                                if score is not None:
                                    color = "green" if score >= 0.8 else "orange" if score >= 0.6 else "red"
                                    emoji = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "🚫"
                                    label = "High" if score >= 0.8 else "Partial" if score >= 0.6 else "Low"
                                    st.markdown(f"**{emoji} Faithfulness: :{color}[{score:.0%} ({label})]**")
                                    if reasoning:
                                        st.caption(reasoning)
                                st.markdown("---")
                            
                            st.caption("Scores represent relative retrieval rank via RRF fusion, not raw semantic similarity.")
                            for idx, src in enumerate(message.get("sources", [])):
                                label = src.get("section") or "Source"
                                sim = src.get("similarity_score", 0)
                                st.markdown(f"**[{idx+1}] [{label}]({src.get('url', '#')})** &nbsp;&nbsp; `<Relevance: {sim:.3f}>`", unsafe_allow_html=True)
                                if src.get("chunk_preview"):
                                    preview = src["chunk_preview"].replace('\n', ' ').strip()
                                    with st.container(border=True):
                                        st.caption(f'"{preview[:400]}..."' if len(preview) > 400 else f'"{preview}"')

        if prompt := st.chat_input("E.g., What is Cloud Run?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                    
                    evaluate_faithfulness = True
                    payload = {"query": prompt, "top_k": top_k, "use_reranker": use_reranker, "history": history, "evaluate_faithfulness": evaluate_faithfulness}

                    try:
                        if stream_response:
                            response = requests.post(f"{API_BASE_URL}/query/stream", json=payload, headers=api_headers, timeout=120, stream=True)
                            if response.status_code == 200:
                                full_response = ""
                                sources = []
                                faithfulness_data = None
                                for line in response.iter_lines():
                                    if line:
                                        decoded_line = line.decode('utf-8')
                                        if decoded_line.startswith("data: "):
                                            try:
                                                data = json.loads(decoded_line[6:])
                                                if data.get("type") == "token":
                                                    full_response += data["content"]
                                                    message_placeholder.markdown(full_response + "▌")
                                                elif data.get("type") == "sources":
                                                    sources = data["content"]
                                                elif data.get("type") == "faithfulness":
                                                    faithfulness_data = data["content"]
                                                elif data.get("type") == "done":
                                                    message_placeholder.markdown(full_response)
                                            except json.JSONDecodeError:
                                                pass
                                
                                if sources or faithfulness_data:
                                    with st.expander(f"🧠 RAG Reasoning & Sources ({len(sources)})"):
                                        if faithfulness_data:
                                            score = faithfulness_data.get("score")
                                            reasoning = faithfulness_data.get("reasoning")
                                            if score is not None:
                                                color = "green" if score >= 0.8 else "orange" if score >= 0.6 else "red"
                                                emoji = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "🚫"
                                                label = "High" if score >= 0.8 else "Partial" if score >= 0.6 else "Low"
                                                st.markdown(f"**{emoji} Faithfulness: :{color}[{score:.0%} ({label})]**")
                                                if reasoning:
                                                    st.caption(reasoning)
                                            st.markdown("---")
                                            
                                        st.caption("Scores represent relative retrieval rank via RRF fusion, not raw semantic similarity.")
                                        for idx, src in enumerate(sources):
                                            label = src.get("section") or "Source"
                                            sim = src.get("similarity_score", 0)
                                            st.markdown(f"**[{idx+1}] [{label}]({src.get('url', '#')})** &nbsp;&nbsp; `<Relevance: {sim:.3f}>`", unsafe_allow_html=True)
                                            if src.get("chunk_preview"):
                                                preview = src["chunk_preview"].replace('\n', ' ').strip()
                                                with st.container(border=True):
                                                    st.caption(f'"{preview[:400]}..."' if len(preview) > 400 else f'"{preview}"')
                                
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": full_response,
                                    "sources": sources,
                                    "faithfulness_data": faithfulness_data
                                })
                            else:
                                error_msg = f"Error {response.status_code}: {response.text[:200]}"
                                message_placeholder.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        else:
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
                                            sim = src.get("similarity_score", 0)
                                            st.markdown(f"**[{idx+1}] [{label}]({src.get('url', '#')})** &nbsp;&nbsp; `<Relevance: {sim:.3f}>`", unsafe_allow_html=True)
                                            if src.get("chunk_preview"):
                                                preview = src["chunk_preview"].replace('\n', ' ').strip()
                                                with st.container(border=True):
                                                    st.caption(f'"{preview[:400]}..."' if len(preview) > 400 else f'"{preview}"')

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
