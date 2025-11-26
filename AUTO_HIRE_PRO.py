# phase1_admin_api.py
import streamlit as st
import pandas as pd
import os
from pypdf import PdfReader
from docx import Document

# ---------------- Config ----------------
CSV_FILE = "companies.csv"

# ---------------- Data Handling ----------------
def load_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    else:
        # Create initial dataframe if file doesn't exist
        df = pd.DataFrame(columns=["Company", "JD", "ResumeThreshold", "AptitudeThreshold"])
        return df

def save_data(df):
    df.to_csv(CSV_FILE, index=False)

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# ---------------- Streamlit UI ----------------
def main():
    st.title("📊 Admin Panel - Manage Job Openings")

    # Load current data
    df = load_data()

    with st.form("add_company"):
        company = st.text_input("Company Name")
        
        # File Uploader for JD
        uploaded_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
        
        jd_text = ""
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".pdf"):
                jd_text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.name.endswith(".docx"):
                jd_text = extract_text_from_docx(uploaded_file)
            st.success("✅ Text extracted from file!")

        jd = st.text_area("Job Description", value=jd_text, height=200)
        resume_threshold = st.number_input("Resume Score Threshold", min_value=0, max_value=100, value=60)
        aptitude_threshold = st.number_input("Aptitude Score Threshold", min_value=0, max_value=40, value=25)
        submit = st.form_submit_button("Save / Update Company")

    if submit:
        if not company:
            st.error("⚠️ Company Name is required!")
        else:
            new_data = {
                "Company": company,
                "JD": jd,
                "ResumeThreshold": resume_threshold,
                "AptitudeThreshold": aptitude_threshold
            }
            
            if company in df["Company"].values:
                # Update existing row
                df.loc[df["Company"] == company, ["JD", "ResumeThreshold", "AptitudeThreshold"]] = [jd, resume_threshold, aptitude_threshold]
                st.success(f"✅ Updated thresholds for {company}")
            else:
                # Add new row
                new_row = pd.DataFrame([new_data])
                df = pd.concat([df, new_row], ignore_index=True)
                st.success(f"✅ Added new company: {company}")
            
            # Save to CSV
            save_data(df)

    # Display
    st.subheader("📌 Current Job Openings")
    # Reload data to show updates
    df = load_data()

    if not df.empty:
        st.dataframe(df)
    else:
        st.info("No companies added yet.")

if __name__ == "__main__":
    main()
