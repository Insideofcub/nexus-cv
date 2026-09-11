import os
import streamlit as st
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="NexusCV Free", page_icon="⚡", layout="wide")

st.markdown("<h1 style='text-align: center;'>⚡ NexusCV Free Intelligence Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #9ca3af;'>100% Free, zero-API-cost resume ranking in the cloud.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 1. Job Description")
    jd_file = st.file_uploader("Upload JD (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="jd_free")
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
    resume_files = st.file_uploader("Upload Resumes (Multiple PDFs or Word docs)", type=["pdf", "docx"], accept_multiple_files=True, key="res_free")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Run Free Ranking"):
    if not jd_text.strip():
        st.error("Please upload a job description.")
    elif not resume_files:
        st.error("Please upload at least one resume.")
    else:
        with st.spinner("Analyzing resumes locally for free..."):
            resume_data = []
            resume_texts = []
            
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
                    resume_texts.append(text)
            
            if resume_texts:
                vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
                tfidf_matrix = vectorizer.fit_transform(resume_texts + [jd_text])
                
                query_vector = tfidf_matrix[-1]
                resume_vectors = tfidf_matrix[:-1]
                
                similarities = cosine_similarity(resume_vectors, query_vector).flatten()
                ranked_indices = similarities.argsort()[::-1]
                
                st.markdown("---")
                st.subheader("🏆 Free Ranking Results")
                
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
                            key=f"free_dl_{rank}_{candidate['filename']}"
                        )
