import pandas as pd
import streamlit as st

from client import ApiError, NexusClient

RECENT_QUERIES = 20


def render_dashboard_tab(client: NexusClient):
    st.title("Pipeline Dashboard")
    st.subheader("Cost & Latency Summary")
    try:
        data = client.usage()
    except ApiError as e:
        st.error(f"Failed to load dashboard data: {e}")
        return

    summary = data.get("summary", {})
    cols = st.columns(4)
    cols[0].metric("Total Queries", summary.get("total_queries", 0))
    cols[1].metric("Total Cost", f"${summary.get('total_cost_usd', 0.0):.4f}")
    cols[2].metric("Avg Cost/Query", f"${summary.get('avg_cost_per_query_usd', 0.0):.4f}")
    cols[3].metric("Avg Latency", f"{summary.get('avg_latency_ms', 0.0):.1f} ms")

    st.markdown("---")
    st.subheader("Recent Queries")
    queries = data.get("queries", [])
    if not queries:
        st.info("No queries found.")
        return
    frame = pd.DataFrame(queries)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp", ascending=False).head(RECENT_QUERIES)
    table = frame[["timestamp", "query", "latency_ms", "tokens_used"]].copy()
    if "total_cost_usd" in frame.columns:
        table["cost_usd"] = frame["total_cost_usd"]
    if "faithfulness_score" in frame.columns:
        table["faith_score"] = frame["faithfulness_score"]
    st.dataframe(table, use_container_width=True)
    st.markdown(f"**Latency Trend (Last {RECENT_QUERIES} Queries)**")
    st.line_chart(frame.set_index("timestamp")["latency_ms"])
