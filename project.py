import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Smart Resume Screening Tool", layout="wide")
st.title("📄 Smart Resume Screening and Candidate Ranking Tool")

# ---------------------------------------------------------------
# Load the local AI model (downloads once, then runs offline)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("Loading AI model (first run may take a minute)..."):
    model = load_model()

# ---------------------------------------------------------------
# Skill database
# ---------------------------------------------------------------
skills = [
    "python", "java", "c++", "machine learning",
    "data science", "sql", "html", "css",
    "javascript", "nlp", "deep learning",
    "streamlit", "pandas", "tensorflow",
    "power bi", "excel", "git", "github"
]


def extract_skills(text):
    text = text.lower()
    return [skill for skill in skills if skill.lower() in text]


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
    return "\n".join(para.text for para in doc.paragraphs)


def read_txt(file):
    return file.read().decode("utf-8")


def read_any(file):
    if file.name.endswith(".pdf"):
        return read_pdf(file)
    elif file.name.endswith(".docx"):
        return read_docx(file)
    elif file.name.endswith(".txt"):
        return read_txt(file)
    return ""


def semantic_similarity(text_a, text_b):
    """Compare two texts by meaning, not just shared words."""
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(score * 100, 2)


# ---------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------
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
    st.subheader("📂 Uploaded Resume(s)")
    for resume in uploaded_resumes:
        text = read_any(resume)
        st.write(f"### {resume.name}")
        st.write(text[:500])
        resume.seek(0)  # reset pointer so it can be read again later

jd_text = ""
if uploaded_jd:
    jd_text = read_any(uploaded_jd)
    uploaded_jd.seek(0)
    st.subheader("📋 Job Description")
    st.write(jd_text)

# ---------------------------------------------------------------
# Ranking logic
# ---------------------------------------------------------------
if uploaded_resumes and uploaded_jd:

    results = []
    jd_skills = extract_skills(jd_text)

    with st.spinner("Analyzing resumes with AI..."):
        for resume in uploaded_resumes:
            resume.seek(0)
            resume_text = read_any(resume)

            score = semantic_similarity(jd_text, resume_text)

            resume_skills = extract_skills(resume_text)
            matched_skills = list(set(resume_skills) & set(jd_skills))

            results.append({
                "Candidate": resume.name,
                "Match Score (%)": score,
                "Matched Skills": ", ".join(matched_skills)
            })

    df = pd.DataFrame(results).sort_values(by="Match Score (%)", ascending=False)

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
        st.write(f"Matched Skills: {row['Matched Skills']}")

        if score >= 60:
            st.success("✅ Excellent Match")
            st.write("- Candidate is highly suitable for this job.")
            st.write("- Recommended for interview.")
        elif score >= 45:
            st.warning("⚠️ Good Match")
            st.write("- Candidate has relevant experience and skills.")
            st.write("- Can be shortlisted.")
        else:
            st.error("❌ Low Match")
            st.write("- Candidate's background doesn't align well with this role.")
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