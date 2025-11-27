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
    st.set_page_config(page_title="Auto Hire Pro", page_icon="🚀", layout="wide")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio("Go to", ["Candidate View", "Admin Panel"])

    # Load current data
    df = load_data()

    # ---------------- CANDIDATE VIEW ----------------
    if app_mode == "Candidate View":
        st.title("🚀 Job Openings")
        st.markdown("Explore and apply for the best opportunities!")

        if df.empty:
            st.info("No job openings available at the moment.")
        else:
            # Job Selection
            company_list = df["Company"].unique().tolist()
            selected_company = st.selectbox("Select a Company", company_list)

            if selected_company:
                # Get details for selected company
                company_data = df[df["Company"] == selected_company].iloc[0]
                
                st.subheader(f"Job Description - {selected_company}")
                st.markdown("---")
                st.write(company_data["JD"])
                st.markdown("---")

                # Application Form
                st.subheader("📝 Apply Now")
                with st.form("application_form"):
                    candidate_email = st.text_input("Your Email Address")
                    uploaded_resume = st.file_uploader("Upload Your Resume (PDF/DOCX)", type=["pdf", "docx"])
                    
                    apply_btn = st.form_submit_button("Submit Application")

                    if apply_btn:
                        if not candidate_email or not uploaded_resume:
                            st.error("⚠️ Please provide both email and resume.")
                        else:
                            # Create resumes directory if not exists
                            if not os.path.exists("resumes"):
                                os.makedirs("resumes")
                            
                            # Save Resume
                            resume_path = os.path.join("resumes", f"{selected_company}_{candidate_email}_{uploaded_resume.name}")
                            with open(resume_path, "wb") as f:
                                f.write(uploaded_resume.getbuffer())
                            
                            st.success(f"✅ Application Submitted Successfully for {selected_company}!")
                            st.balloons()

    # ---------------- ADMIN PANEL ----------------
    elif app_mode == "Admin Panel":
        st.title("📊 Admin Panel - Manage Job Openings")

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
