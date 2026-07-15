from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document

st.set_page_config(page_title="Smart Resume Screening Tool", layout="wide")
st.title("📄 Smart Resume Screening and Candidate Ranking Tool")

def read_pdf(file):
    text = ""
    pdf = PdfReader(file)
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def read_docx(file):
    doc = Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def read_txt(file):
    return file.read().decode("utf-8")

uploaded_resumes = st.file_uploader(
    "Upload Resume(s)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

uploaded_jd = st.file_uploader(
    "Upload Job Description",
    type=["pdf", "docx", "txt"]
)

if uploaded_resumes:
    st.subheader("Uploaded Resume(s)")
    for resume in uploaded_resumes:
        if resume.name.endswith(".pdf"):
            text = read_pdf(resume)
        elif resume.name.endswith(".docx"):
            text = read_docx(resume)

        st.write(f"### {resume.name}")
        st.write(text[:500])

jd_text = ""

if uploaded_jd:
    if uploaded_jd.name.endswith(".pdf"):
        jd_text = read_pdf(uploaded_jd)
    elif uploaded_jd.name.endswith(".docx"):
        jd_text = read_docx(uploaded_jd)
    elif uploaded_jd.name.endswith(".txt"):
        jd_text = read_txt(uploaded_jd)

    st.subheader("📋 Job Description")
    st.write(jd_text)

if uploaded_resumes and uploaded_jd:
    results = []

    for resume in uploaded_resumes:
        if resume.name.endswith(".pdf"):
            resume_text = read_pdf(resume)
        elif resume.name.endswith(".docx"):
            resume_text = read_docx(resume)

        documents = [jd_text, resume_text]
        tfidf = TfidfVectorizer(stop_words="english").fit_transform(documents)
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        score = round(similarity * 100, 2)

        results.append({
            "Candidate": resume.name,
            "Match Score (%)": score
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Match Score (%)", ascending=False)

    st.subheader("🏆 Candidate Ranking")
    st.dataframe(df, use_container_width=True)

    top_candidate = df.iloc[0]
    st.success(
        f"🏆 Top Candidate: {top_candidate['Candidate']} "
        f"({top_candidate['Match Score (%)']}%)"
    )

    st.subheader("🤖 AI Resume Analysis")

    for index, row in df.iterrows():
        score = row["Match Score (%)"]
        st.write(f"### {row['Candidate']}")

        if score >= 20:
            st.success("✅ Excellent Match")
            st.write("- Candidate is highly suitable for this job.")
            st.write("- Recommended for interview.")
        elif score >= 15:
            st.warning("⚠️ Good Match")
            st.write("- Candidate has most of the required skills.")
            st.write("- Can be shortlisted.")
        else:
            st.error("❌ Low Match")
            st.write("- Candidate lacks several required skills.")
            st.write("- Not recommended for this role.")

    st.subheader("📊 Candidate Match Score Chart")
    st.bar_chart(df.set_index("Candidate")["Match Score (%)"])

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Ranking Report",
        data=csv,
        file_name="candidate_ranking.csv",
        mime="text/csv"
    )