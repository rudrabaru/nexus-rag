import time

import streamlit as st

from client import ApiError, NexusClient

MAX_FILES_PER_BATCH = 5
MAX_FILE_BYTES = 20 * 1024 * 1024  # the API's upload limit
POLL_SECONDS = 2
MAX_POLL_SECONDS = 600  # ingestion is queued: with no worker running a job stays queued, which should read as a message, not a frozen bar
FINISHED = ("complete", "failed", "partial_success")
STATUS_COLORS = {"complete": "green", "processing": "orange", "failed": "red"}


def _with_filter(url: str, sitemap_filter: str) -> str:
    if not sitemap_filter.strip():
        return url
    return f"{url}{'&' if '?' in url else '?'}filter={sitemap_filter.strip()}"


def _final_message(name: str, job: dict) -> None:
    meta = job.get("metadata") or {}
    chunks, total, fetched, skipped = job.get("chunk_count", "?"), meta.get("total_pages"), meta.get("fetched_pages") or 0, meta.get("failed_pages") or 0
    scope = f"{fetched}/{total} pages, " if total is not None else ""
    reason = job.get("error") or meta.get("error_reason") or "Some chunks or pages failed processing."
    status = job["status"]
    if status == "complete":
        st.success(f"✅ [{name}] Indexed {scope}{chunks} chunks. {skipped} skipped.")
    elif status == "partial_success":
        st.warning(f"⚠️ [{name}] Partial success: {scope}{chunks} chunks. Reason: {reason}")
    else:
        st.error(f"❌ [{name}] {job.get('error', 'Unknown error')}")


def _watch(client: NexusClient, name: str, job_id: str) -> None:
    progress = st.empty()
    for _ in range(MAX_POLL_SECONDS // POLL_SECONDS):
        time.sleep(POLL_SECONDS)
        try:
            job = client.job(job_id)
        except ApiError:
            return
        meta = job.get("metadata") or {}
        pages = f" ({meta.get('fetched_pages') or 0}/{meta['total_pages']} pages)" if meta.get("total_pages") is not None else ""
        progress.progress(job.get("progress_pct", 0) / 100, text=f"[{name}] {job['status'].upper()} — {job.get('progress_pct', 0)}%{pages}")
        if job["status"] in FINISHED:
            progress.empty()
            _final_message(name, job)
            return
    progress.warning(
        f"⏳ [{name}] Still queued after {MAX_POLL_SECONDS}s: are the workers running? "
        "(`python -m src.jobs.workers ingest`; URLs also need `python -m src.jobs.workers fetch`)"
    )


def _submit(client: NexusClient, url: str, sitemap_filter: str, files) -> None:
    jobs = []
    if url:
        target = _with_filter(url.strip(), sitemap_filter)
        jobs.append((target, lambda: client.ingest_url(target)))
    for f in files or []:
        jobs.append((f.name, lambda f=f: client.ingest_file(f.name, f.getvalue(), f.type)))

    queued = []
    for name, send in jobs:
        try:
            queued.append((name, send()))
        except ApiError as e:
            st.error(f"Ingest error ({name}): {e.message}")
    if queued:
        st.success("Jobs queued.")
    for name, job_id in queued:
        if job_id:
            _watch(client, name, job_id)


def _document_row(client: NexusClient, doc: dict) -> None:
    with st.container(border=True):
        title, chunks, status, delete = st.columns([4, 2, 2, 1])
        title.markdown(f"**{doc.get('title') or doc.get('url')}**")
        chunks.markdown(f"Chunks: `{doc.get('chunks', 0)}`")
        state = doc.get("status", "unknown")
        status.markdown(f"Status: :{STATUS_COLORS.get(state, 'blue')}[{state.upper()}]")
        if doc.get("error"):
            (st.error if state == "failed" else st.warning)(doc["error"])
        if delete.button("Delete", key=f"del_{doc['id']}"):
            client.delete_document(doc["id"])
            st.rerun()


def render_documents_tab(client: NexusClient):
    st.title("Document Management")
    st.markdown("Upload documents or provide URLs to index them.")

    left, right = st.columns(2)
    with left:
        url = st.text_input("Web page or sitemap URL (https)")
        sitemap_filter = st.text_input("Sitemap URL filter (optional, e.g. /docs/)")
    with right:
        files = st.file_uploader("Or upload files", type=["pdf", "docx", "md", "txt"], accept_multiple_files=True)

    if st.button("Process & Index", type="primary", use_container_width=True):
        if files and len(files) > MAX_FILES_PER_BATCH:
            st.error(f"At most {MAX_FILES_PER_BATCH} files at a time.")
        elif files and any(f.size > MAX_FILE_BYTES for f in files):
            st.error(f"Each file may be at most {MAX_FILE_BYTES // (1024 * 1024)} MB.")
        elif not url and not files:
            st.warning("Provide a URL or upload a file.")
        else:
            _submit(client, url, sitemap_filter, files)

    st.markdown("---")
    header, refresh = st.columns([4, 1])
    header.subheader("Indexed Documents")
    if refresh.button("Refresh", use_container_width=True):
        st.rerun()
    try:
        documents = client.documents()
    except ApiError as e:
        st.error(f"Error loading documents: {e}")
        return
    if not documents:
        st.info("No documents found.")
    for doc in documents:
        _document_row(client, doc)
