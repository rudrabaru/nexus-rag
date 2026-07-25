import streamlit as st
import requests
import pandas as pd

def render_documents_tab(API_BASE_URL, api_headers):
    st.title("Document Management")
    st.markdown("Upload documents or provide URLs to index them.")
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        ingest_url = st.text_input("URL or Sitemap to crawl")
        extract_visuals = st.toggle("Extract Visuals (Images/Charts)", value=False)
    with col_input2:
        ingest_files = st.file_uploader("Or upload files", type=["pdf", "docx", "md", "txt"], accept_multiple_files=True)
    
    if st.button("Process & Index", type="primary", use_container_width=True):
        if not ingest_url and not ingest_files:
            st.warning("Please provide a URL or upload a file.")
        else:
            with st.spinner("Queuing ingestion jobs..."):
                try:
                    data = {"extract_visuals": extract_visuals}
                    success = True
                    if ingest_url:
                        url_data = data.copy()
                        url_data["url"] = ingest_url
                        res = requests.post(f"{API_BASE_URL}/ingest", data=url_data, headers=api_headers)
                        if res.status_code != 200:
                            success = False
                            st.error(f"Ingest Error (URL): {res.json().get('detail', res.text)}")
                    if ingest_files:
                        for f in ingest_files:
                            files = {"file": (f.name, f.getvalue(), f.type)}
                            res = requests.post(f"{API_BASE_URL}/ingest", files=files, data=data, headers=api_headers)
                            if res.status_code != 200:
                                success = False
                                st.error(f"Ingest Error ({f.name}): {res.json().get('detail', res.text)}")
                    if success:
                        st.success("Jobs queued successfully.")
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
