import os
import streamlit as st
import zipfile
import tempfile
import re
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="NexusCV Enterprise", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    .candidate-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .summary-box {
        background-color: #fcf5ff;
        border-left: 4px solid #a855f7;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 14px;
        color: #374151;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>⚡ NexusCV Professional Intelligence Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6b7280;'>Search by skills, keywords, or match against a job description.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 1. Job Description (Optional)")
    jd_file = st.file_uploader("Upload Job Description (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="jd_upload")
    jd_text = ""
    if jd_file:
        try:
            if jd_file.name.endswith(".txt"):
                jd_text = jd_file.read().decode("utf-8")
            elif jd_file.name.endswith(".pdf"):
                reader = PdfReader(jd_file)
                jd_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
            elif jd_file.name.endswith(".docx"):
                doc = Document(jd_file)
                jd_text = "\n".join([p.text for p in doc.paragraphs])
            st.success(f"Successfully loaded JD: {jd_file.name}")
        except Exception as e:
            st.error(f"Error reading JD file: {e}")

with col2:
    st.markdown("### 📂 2. Candidate Resumes Folder")
    zip_file = st.file_uploader("Upload Entire Resumes Folder (.zip)", type=["zip"], key="zip_upload")
    if zip_file:
        st.success(f"Successfully loaded Zip: {zip_file.name}")

st.markdown("### 🔍 3. Search Filter / Skill Keyword")
search_keyword = st.text_input("Type any skill, phone number, or keyword (e.g. coder, python, medical):")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Run Candidate Intelligence Engine"):
    if not zip_file:
        st.error("Please upload your zipped resume folder first.")
    else:
        with st.spinner("Processing candidate profiles from zip folder..."):
            resume_data = []
            resume_texts = []
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = os.path.join(tmp_dir, "uploaded_resumes.zip")
                with open(zip_path, "wb") as f:
                    f.write(zip_file.getbuffer())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)
                
                for root, dirs, files in os.walk(tmp_dir):
                    for file in files:
                        if file.lower().endswith(('.pdf', '.docx')) and not file.startswith('._'):
                            file_path = os.path.join(root, file)
                            text = ""
                            try:
                                if file.lower().endswith('.pdf'):
                                    reader = PdfReader(file_path)
                                    text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
                                elif file.lower().endswith('.docx'):
                                    doc = Document(file_path)
                                    text = "\n".join([p.text for p in doc.paragraphs])
                                
                                if text.strip():
                                    # If search keyword is typed, filter by it
                                    if search_keyword.strip():
                                        if search_keyword.lower() not in text.lower() and search_keyword.lower() not in file.lower():
                                            continue
                                            
                                    phones = re.findall(r'[\+\(?[0-9][0-9 .\-\(\)]{8,}[0-9]', text)
                                    phone_str = phones[0].strip() if phones else "Not Provided"
                                    
                                    emails = re.findall(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
                                    email_str = emails[0].strip() if emails else "Not Provided"
                                    
                                    with open(file_path, "rb") as f_obj:
                                        file_bytes = f_obj.read()
                                        
                                    resume_data.append({
                                        "filename": file, 
                                        "text": text, 
                                        "bytes": file_bytes,
                                        "phone": phone_str,
                                        "email": email_str
                                    })
                                    resume_texts.append(text)
                            except Exception:
                                pass

            if not resume_texts:
                st.warning("No matching candidates found matching your keyword criteria inside the zip file.")
            else:
                # If no JD is provided, use the search keyword as a pseudo-query, or rank by file relevance
                target_text = jd_text if jd_text.strip() else (search_keyword if search_keyword.strip() else "resume candidate skills")
                
                vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
                tfidf_matrix = vectorizer.fit_transform(resume_texts + [target_text])
                
                query_vector = tfidf_matrix[-1]
                resume_vectors = tfidf_matrix[:-1]
                
                similarities = cosine_similarity(resume_vectors, query_vector).flatten()
                ranked_indices = similarities.argsort()[::-1]
                
                st.markdown("---")
                st.subheader(f"🏆 Candidate Pool Results ({len(resume_texts)} Matched)")
                
                for rank, idx in enumerate(ranked_indices, 1):
                    candidate = resume_data[idx]
                    score = round(float(similarities[idx]) * 100, 2)
                    
                    display_name = os.path.splitext(candidate['filename'])[0].replace('_', ' ').replace('-', ' ').title()
                    summary_snippet = candidate['text'][:300].replace('\n', ' ')
                    
                    st.markdown(f"""
                        <div class="candidate-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <h3 style="margin: 0; color: #1f2937;">{display_name}</h3>
                                <div>
                                    <span style="background-color: #d1fae5; color: #065f46; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold;">Match: {score}%</span>
                                    <span style="background-color: #e0f2fe; color: #0369a1; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 5px;">Rank #{rank}</span>
                                </div>
                            </div>
                            <p style="margin: 5px 0 10px 0; color: #4b5563; font-size: 14px;">📁 File: {candidate['filename']}</p>
                            <p style="margin: 5px 0; color: #374151; font-size: 14px;">
                                📞 <b>Phone:</b> {candidate['phone']} &nbsp;&nbsp;|&nbsp;&nbsp; ✉️ <b>Email:</b> {candidate['email']}
                            </p>
                            <div class="summary-box">
                                <b>Profile Summary:</b> {summary_snippet}...
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander(f"📂 View Full CV Preview & Actions — {display_name}"):
                        st.text_area("Full Resume Text", candidate['text'], height=250, key=f"preview_{rank}_{candidate['filename']}")
                        st.download_button(
                            label="📥 Download Original CV File",
                            data=candidate["bytes"],
                            file_name=candidate["filename"],
                            mime="application/octet-stream",
                            key=f"free_dl_{rank}_{candidate['filename']}"
                        )
