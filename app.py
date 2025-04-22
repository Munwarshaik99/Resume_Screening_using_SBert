
import streamlit as st
import os
import pdfplumber
import docx2txt
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re

@st.cache_resource
def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_bert_model()

st.title("🧠 Advanced Resume Screening Tool")
st.markdown("Upload **Job Description** and **multiple resumes** (PDF, DOCX, CSV), and get dynamic candidate scores and insights.")

jd_file = st.file_uploader("📄 Upload Job Description (PDF/DOCX/CSV)", type=["pdf", "docx", "csv"])
resume_files = st.file_uploader("📂 Upload Resumes (PDF/DOCX/CSV) — Multiple", type=["pdf", "docx", "csv"], accept_multiple_files=True)
process_btn = st.button("🚀 Process")

def extract_text(file):
    if file.name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file.name.endswith('.docx'):
        return docx2txt.process(file)
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)
        return "\n".join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))
    return ""

def extract_years_experience(text):
    patterns = [
        r'(\d{1,2})\+?\s*(years|yrs)[\s\w]{0,15}(experience|exp)',
        r'experience\s*(of)?\s*(\d{1,2})\+?\s*(years|yrs)',
        r'(\d{1,2})\s*(years|yrs)\s*of\s*experience'
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return match[1] if match[1].isdigit() else match[2]
    return "Not Found"

def extract_education(text):
    edu_keywords = ['B.Tech', 'M.Tech', 'BE', 'ME', 'B.Sc', 'M.Sc', 'MBA', 'PhD', 'Bachelor', 'Master', 'Bachelors', 'Masters', 'BS', 'MS']
    return ", ".join([edu for edu in edu_keywords if edu.lower() in text.lower()]) or "Not Found"

def extract_primary_skills(text):
    common_skills = [
        'Python', 'Java', 'C++', 'C#', 'JavaScript', 'SQL', 'R', 'Scala',
        'Machine Learning', 'Data Science', 'Deep Learning', 'Pandas', 'Numpy',
        'PyTorch', 'TensorFlow', 'Keras', 'Hadoop', 'Spark', 'Power BI', 'Tableau',
        'Data Engineering', 'AWS', 'Azure', 'GCP', 'Flask', 'Django'
    ]
    found_skills = [skill for skill in common_skills if skill.lower() in text.lower()]
    return ", ".join(found_skills) or "Not Found"

def calculate_bert_score(jd_text, resume_text):
    jd_embed = model.encode(jd_text, convert_to_tensor=True)
    resume_embed = model.encode(resume_text, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(jd_embed, resume_embed)[0][0]) * 100

if process_btn and jd_file and resume_files:
    jd_text = extract_text(jd_file)
    results = []

    for resume in resume_files:
        resume_text_raw = extract_text(resume)
        score = round(calculate_bert_score(jd_text, resume_text_raw), 2)
        skills = extract_primary_skills(resume_text_raw)
        experience = extract_years_experience(resume_text_raw)
        education = extract_education(resume_text_raw)

        results.append({
            "Candidate File": resume.name,
            "Score": score,
            "Primary Skills": skills,
            "Years of Experience": experience,
            "Education": education
        })

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results CSV", csv, "screening_results.csv", "text/csv")

    st.success("✅ Screening completed successfully!")
