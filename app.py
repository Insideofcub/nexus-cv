import os
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="NexusCV Cloud AI", page_icon="⚡", layout="wide")

st.markdown("<h1 style='text-align: center;'>⚡ NexusCV Cloud Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af;'>Cloud-deployed resume ranker powered by OpenAI.</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar for API Key
st.sidebar.markdown("### 🔑 Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))

if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to proceed.")
else:
    client = OpenAI(api_key=api_key)

    # Main UI Layout
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📋 1. Job Description")
        jd_file = st.file_uploader("Upload JD (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="jd_upload")
        jd_text = ""
        if jd_file:
            if jd_file.name.endswith(".txt"):
                jd_text = jd_file.read().decode("utf-8")
            elif jd_file.name.endswith(".pdf"):
                reader = PdfReader(jd_file)
                jd_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            elif jd_file.name.endswith(".docx"):
                doc = Document(jd_file)
                jd_text = "\n".join([p.text for p in doc.paragraphs])

    with col2:
        st.markdown("### 📂 2. Candidate Resumes")
        resume_files = st.file_uploader("Upload Resumes (Multiple PDFs or Word docs)", type=["pdf", "docx"], accept_multiple_files=True, key="resume_upload")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Run Cloud AI Ranking"):
        if not jd_text.strip():
            st.error("Please provide a job description.")
        elif not resume_files:
            st.error("Please upload at least one resume.")
        else:
            with st.spinner("Analyzing and generating embeddings via OpenAI..."):
                # Extract text from all uploaded resumes
                resume_data = []
                for file in resume_files:
                    text = ""
                    if file.name.endswith(".pdf"):
                        reader = PdfReader(file)
                        text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
                    elif file.name.endswith(".docx"):
                        doc = Document(file)
                        text = "\n".join([p.text for p in doc.paragraphs])
                    
                    if text.strip():
                        resume_data.append({"filename": file.name, "text": text, "file_obj": file})

                if resume_data:
                    # Get embeddings from OpenAI for JD and Resumes
                    texts_to_embed = [jd_text] + [r["text"] for r in resume_data]
                    
                    response = client.embeddings.create(
                        input=texts_to_embed,
                        model="text-embedding-3-small"
                    )
                    
                    embeddings = [item.embedding for item in response.data]
                    jd_embedding = np.array(embeddings[0]).reshape(1, -1)
                    resume_embeddings = np.array(embeddings[1:])
                    
                    # Calculate cosine similarity
                    similarities = cosine_similarity(resume_embeddings, jd_embedding).flatten()
                    ranked_indices = similarities.argsort()[::-1]

                    st.markdown("---")
                    st.subheader("🏆 AI Ranking Results")

                    for rank, idx in enumerate(ranked_indices, 1):
                        candidate = resume_data[idx]
                        score = round(float(similarities[idx]) * 100, 2)
                        
                        with st.expander(f"Rank #{rank}: {candidate['filename']} — Match Score: {score}%"):
                            st.markdown(f"**Snippet:**\n> {candidate['text'][:400]}...")
                            st.download_button(
                                label="📥 Download Resume",
                                data=candidate["file_obj"].getvalue(),
                                file_name=candidate["filename"],
                                mime="application/octet-stream",
                                key=f"dl_{rank}_{candidate['filename']}"
                            )