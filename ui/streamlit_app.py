"""Streamlit chat UI for the RAG assistant.

A thin client: it uploads PDFs and posts questions to the FastAPI backend, then
renders the answer and the source chunks behind it. All logic lives in the API —
this file only does HTTP and display.

Run:
    streamlit run ui/streamlit_app.py
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Industrial Document RAG", page_icon="📄")
st.title("📄 Industrial Document RAG Assistant")

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )
    if st.button("Ingest", disabled=not uploads):
        files = [(u.name, u.getvalue(), "application/pdf") for u in uploads]
        with st.spinner("Ingesting…"):
            resp = requests.post(
                f"{BACKEND_URL}/ingest",
                files=[("files", f) for f in files],
                timeout=300,
            )
        if resp.ok:
            for f in resp.json()["files"]:
                st.success(f"{f['source']}: {f['pages']} pages → {f['chunks']} chunks")
        else:
            st.error(f"Ingest failed ({resp.status_code}): {resp.text}")

question = st.chat_input("Ask a question about your documents…")
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            resp = requests.post(
                f"{BACKEND_URL}/ask", json={"question": question}, timeout=120
            )
        if not resp.ok:
            st.error(f"Request failed ({resp.status_code}): {resp.text}")
        else:
            data = resp.json()
            st.write(data["answer"])
            sources = data["sources"]
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for s in sources:
                        st.markdown(
                            f"**[{s['source']} p.{s['page']}]** "
                            f"· similarity {s['score']:.3f}"
                        )
                        st.caption(s["content"])
