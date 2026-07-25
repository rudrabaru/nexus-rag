import streamlit as st
import requests
import pandas as pd

def render_dashboard_tab(API_BASE_URL, api_headers):
    st.title("Pipeline Dashboard")
    
    st.subheader("Cost & Latency Summary")
    try:
        res = requests.get(f"{API_BASE_URL}/logs", headers=api_headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            summary = data.get("summary", {})
            chat_logs = data.get("queries", [])
            
            cols = st.columns(4)
            cols[0].metric("Total Queries", summary.get("total_queries", 0))
            cols[1].metric("Total Cost", f"${summary.get('total_cost_usd', 0.0):.4f}")
            cols[2].metric("Avg Cost/Query", f"${summary.get('avg_cost_per_query_usd', 0.0):.4f}")
            cols[3].metric("Avg Latency", f"{summary.get('avg_latency_ms', 0.0):.1f} ms")
            
            st.markdown("---")
            st.subheader("Recent Queries")
            if chat_logs:
                df_logs = pd.DataFrame(chat_logs)
                df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
                df_logs = df_logs.sort_values("timestamp", ascending=False).head(20)
                
                display_df = df_logs[["timestamp", "query", "latency_ms", "tokens_used"]].copy()
                if "total_cost_usd" in df_logs.columns:
                    display_df["cost_usd"] = df_logs["total_cost_usd"]
                if "faithfulness_score" in df_logs.columns:
                    display_df["faith_score"] = df_logs["faithfulness_score"]
                    
                st.dataframe(display_df, use_container_width=True)
                
                st.markdown("**Latency Trend (Last 20 Queries)**")
                st.line_chart(df_logs.set_index("timestamp")["latency_ms"])
            else:
                st.info("No queries found.")
                
    except Exception as e:
        st.error(f"Failed to load dashboard data: {e}")
