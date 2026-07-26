import streamlit as st
import requests
import pandas as pd
import time

def render_documents_tab(API_BASE_URL, api_headers):
    st.title("Document Management")
    st.markdown("Upload documents or provide URLs to index them.")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        ingest_url = st.text_input("URL or Sitemap to crawl")
        sitemap_filter = st.text_input("Sitemap URL Prefix Filter (optional, e.g. /payment-methods/google-pay/)")
        extract_visuals = st.toggle("Extract Visuals (Images/Charts)", value=False)
    with col_input2:
        ingest_files = st.file_uploader("Or upload files", type=["pdf", "docx", "md", "txt"], accept_multiple_files=True)
    
    if st.button("Process & Index", type="primary", use_container_width=True):
        if ingest_files and len(ingest_files) > 5:
            st.error("Upload limit exceeded: You can only upload a maximum of 5 files at a time.")
        elif ingest_files and any(f.size > 20 * 1024 * 1024 for f in ingest_files):
            st.error("File size limit exceeded: Maximum file size is 20MB per file.")
        elif not ingest_url and not ingest_files:
            st.warning("Please provide a URL or upload a file.")
        else:
            with st.spinner("Queuing ingestion jobs..."):
                try:
                    data = {"extract_visuals": extract_visuals}
                    success = True
                    job_ids = []
                    if ingest_url:
                        url_data = data.copy()
                        url_to_send = ingest_url.strip()
                        if sitemap_filter and sitemap_filter.strip() and ("sitemap" in url_to_send.lower() or url_to_send.lower().endswith(".xml")):
                            sep = "&" if "?" in url_to_send else "?"
                            url_to_send = f"{url_to_send}{sep}filter={sitemap_filter.strip()}"
                        url_data["url"] = url_to_send
                        res = requests.post(f"{API_BASE_URL}/ingest", data=url_data, headers=api_headers)
                        if res.status_code != 200:
                            success = False
                            st.error(f"Ingest Error (URL): {res.json().get('detail', res.text)}")
                        else:
                            job_ids.append((url_to_send, res.json().get("job_id")))
                    if ingest_files:
                        for f in ingest_files:
                            files = {"file": (f.name, f.getvalue(), f.type)}
                            res = requests.post(f"{API_BASE_URL}/ingest", files=files, data=data, headers=api_headers)
                            if res.status_code != 200:
                                success = False
                                st.error(f"Ingest Error ({f.name}): {res.json().get('detail', res.text)}")
                            else:
                                job_ids.append((f.name, res.json().get("job_id")))
                    if success:
                        st.success("Jobs queued successfully.")
                        for src_name, jid in job_ids:
                            if not jid: continue
                            prog_ph = st.empty()
                            while True:
                                time.sleep(2)
                                s_res = requests.get(f"{API_BASE_URL}/ingest/{jid}", headers=api_headers, timeout=10)
                                if s_res.status_code == 200:
                                    j_data = s_res.json()
                                    pct = j_data.get("progress_pct", 0)
                                    status = j_data.get("status", "processing")
                                    meta = j_data.get("metadata") or {}
                                    total_p = meta.get("total_pages")
                                    idx_p = meta.get("indexed_pages")
                                    fail_p = meta.get("failed_pages")

                                    if total_p is not None:
                                        msg = f"[{src_name}] {status.upper()} — {pct}% ({idx_p or 0}/{total_p} pages, {fail_p or 0} skipped)"
                                    else:
                                        msg = f"[{src_name}] {status.upper()} — {pct}%"

                                    prog_ph.progress(pct / 100, text=msg)
                                    if status in ("complete", "failed", "partial_success"):
                                        if status == "complete":
                                            if total_p is not None:
                                                prog_ph.success(f"✅ [{src_name}] Indexed {idx_p or 0} pages from sitemap ({j_data.get('chunk_count', '?')} chunks). {fail_p or 0} skipped.")
                                            else:
                                                prog_ph.success(f"✅ [{src_name}] Indexed {j_data.get('chunk_count', '?')} chunks.")
                                        elif status == "partial_success":
                                            prog_ph.warning(f"⚠️ [{src_name}] Partial success: Indexed {idx_p or 0}/{total_p or '?'} pages ({j_data.get('chunk_count', '?')} chunks). {fail_p or 0} failed.")
                                        elif status == "failed":
                                            prog_ph.error(f"❌ [{src_name}] {j_data.get('error', 'Unknown error')}")
                                        break
                                else:
                                    break
                except Exception as e:
                    st.error(f"Ingestion error: {str(e)}")
                    
    st.markdown("---")
    
    col_hdr1, col_hdr2 = st.columns([4,1])
    with col_hdr1:
        st.subheader("Indexed Documents")
    with col_hdr2:
        if st.button("Refresh Documents", use_container_width=True):
            st.rerun()
            
    try:
        res = requests.get(f"{API_BASE_URL}/documents", headers=api_headers)
        if res.status_code == 200:
            docs = res.json()
            if docs:
                df_docs = pd.DataFrame(docs)
                for index, doc in df_docs.iterrows():
                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns([4, 2, 2, 1])
                        with c1:
                            st.markdown(f"**{doc.get('title') or doc.get('url')}**")
                        with c2:
                            st.markdown(f"Chunks: `{doc.get('chunks', 0)}`")
                        with c3:
                            status = doc.get('status', 'unknown')
                            color = "green" if status == "complete" else "orange" if status == "processing" else "red"
                            st.markdown(f"Status: :{color}[{status.upper()}]")
                            if status == "failed" and doc.get("error"):
                                st.error(f"Error: {doc.get('error')}")
                        with c4:
                            if st.button("Delete", key=f"del_{doc['id']}"):
                                requests.delete(f"{API_BASE_URL}/documents/{doc['id']}", headers=api_headers)
                                st.rerun()
            else:
                st.info("No documents found.")
    except Exception as e:
        st.error(f"Error loading documents: {e}")
