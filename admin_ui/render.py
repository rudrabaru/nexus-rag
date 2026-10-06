"""The one way sources and faithfulness scores are drawn, whichever tab or answer mode shows them."""
from typing import List, Optional

import streamlit as st

PREVIEW_CHARS = 400
FAITHFUL_AT = 0.8
PARTIAL_AT = 0.6
SCORE_CAPTION = "A score only compares sources within this one list; it is not a similarity percentage."


def _clean(preview: str, limit: int) -> str:
    text = " ".join((preview or "").split())
    return f'"{text[:limit]}..."' if len(text) > limit else f'"{text}"'


def render_faithfulness(score: Optional[float], reasoning: Optional[str] = None) -> None:
    if score is None:
        return
    if score >= FAITHFUL_AT:
        color, emoji, label = "green", "✅", "High"
    elif score >= PARTIAL_AT:
        color, emoji, label = "orange", "⚠️", "Partial"
    else:
        color, emoji, label = "red", "🚫", "Low"
    st.markdown(f"**{emoji} Faithfulness: :{color}[{score:.0%} ({label})]**")
    if reasoning:
        st.caption(reasoning)


def render_sources(sources: List[dict], limit: int = PREVIEW_CHARS) -> None:
    if sources:
        st.caption(SCORE_CAPTION)
    for position, source in enumerate(sources, start=1):
        label = source.get("section") or "Source"
        score = source.get("similarity_score", 0)
        st.markdown(f"**[{position}] [{label}]({source.get('url') or '#'})** &nbsp;&nbsp; `<Score: {score:.3f}>`", unsafe_allow_html=True)
        if source.get("chunk_preview"):
            with st.container(border=True):
                st.caption(_clean(source["chunk_preview"], limit))
