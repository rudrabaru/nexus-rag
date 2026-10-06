import streamlit as st

from client import ApiError, NexusClient
from render import render_faithfulness, render_sources

MAX_HISTORY_TURNS = 20  # the API's limits on `history`: turns and characters per turn
MAX_TURN_CHARS = 4000


def _show_answer_details(message: dict) -> None:
    sources, faithfulness = message.get("sources") or [], message.get("faithfulness") or {}
    if not sources and faithfulness.get("score") is None and not message.get("latency_ms"):
        return
    with st.expander(f"Sources ({len(sources)})"):
        render_faithfulness(faithfulness.get("score"), faithfulness.get("reasoning"))
        if message.get("latency_ms"):
            breakdown = message.get("latency_breakdown") or {}
            parts = ", ".join(f"{name} {ms:.0f} ms" for name, ms in breakdown.items())
            st.markdown(f"**Latency:** {message['latency_ms']:.0f} ms" + (f" ({parts})" if parts else ""))
        render_sources(sources)


def _stream(client: NexusClient, payload: dict, placeholder) -> dict:
    message = {"role": "assistant", "content": ""}
    for event in client.chat_events(payload):
        kind = event.get("type")
        if kind == "token":
            message["content"] += event["content"]
            placeholder.markdown(message["content"] + "▌")
        elif kind == "sources":
            message["sources"] = event["content"]
        elif kind == "faithfulness":
            message["faithfulness"] = event["content"]
        elif kind == "done":
            message["latency_ms"] = event.get("latency_ms")
        elif kind == "error":
            message["content"] += f"\n\n:red[{event.get('message', 'The answer could not be completed.')}]"
    placeholder.markdown(message["content"])
    return message


def _complete(client: NexusClient, payload: dict, placeholder) -> dict:
    data = client.chat(payload)
    placeholder.markdown(data["answer"])
    return {
        "role": "assistant", "content": data["answer"], "sources": data.get("sources", []),
        "latency_ms": data.get("latency_ms"), "latency_breakdown": data.get("latency_breakdown"),
    }


def render_chat_tab(client: NexusClient, signed_in: bool, top_k: int, use_reranker: bool, stream: bool, judge: bool):
    st.title("Nexus RAG Assistant")
    if not signed_in:
        st.warning("Enter your API key in the sidebar.")
        return

    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _show_answer_details(message)

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return
    history = [{"role": m["role"], "content": m["content"][:MAX_TURN_CHARS]} for m in messages[-MAX_HISTORY_TURNS:]]
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        payload = {
            "query": prompt, "top_k": top_k, "use_reranker": use_reranker, "history": history,
            "evaluate_faithfulness": judge,
        }
        try:
            reply = (_stream if stream else _complete)(client, payload, placeholder)
        except ApiError as e:
            placeholder.error(str(e))
            reply = {"role": "assistant", "content": str(e)}
        else:
            _show_answer_details(reply)
        messages.append(reply)
