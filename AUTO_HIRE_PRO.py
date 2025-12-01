# phase1_admin_api.py
import streamlit as st
import pandas as pd
import os
from pypdf import PdfReader
from docx import Document
import google.generativeai as genai
from datetime import datetime
import time

# ---------------- Config ----------------
CSV_FILE = "companies.csv"
APPS_FILE = "applications.csv"
API_KEY = "AIzaSyBsEF-QP4mHsaimN8CLXQx_z7JxwlK5zoY"  # Replace with env var in production

# Configure Gemini
genai.configure(api_key=API_KEY)

# ---------------- Data Handling ----------------
def load_data():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        if "Role" not in df.columns:
            df["Role"] = "Open Role"
        if "JD_File_Path" not in df.columns:
            df["JD_File_Path"] = ""
        df = df.fillna("")
        return df
    else:
        df = pd.DataFrame(columns=["Company", "Role", "JD", "JD_File_Path", "ResumeThreshold", "AptitudeThreshold"])
        return df

def save_data(df):
    df.to_csv(CSV_FILE, index=False)

def load_apps():
    if os.path.exists(APPS_FILE):
        return pd.read_csv(APPS_FILE)
    else:
        return pd.DataFrame(columns=["Company", "Role", "Email", "Score", "Resume_Path", "Timestamp"])

def save_apps(df):
    df.to_csv(APPS_FILE, index=False)

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

def calculate_score(resume_text, jd_text):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        You are an expert ATS (Applicant Tracking System).
        Compare the following Resume with the Job Description.
        
        JOB DESCRIPTION:
        {jd_text}
        
        RESUME:
        {resume_text}
        
        Task:
        1. Evaluate how well the resume matches the JD.
        2. Provide a single integer score from 0 to 100.
        3. Output ONLY the integer score. Do not output any other text.
        """
        response = model.generate_content(prompt)
        score = int(''.join(filter(str.isdigit, response.text)))
        return score
    except Exception as e:
        st.error(f"AI Error: {e}")
        return 0

# ---------------- Streamlit UI ----------------
def main():
    st.set_page_config(page_title="Auto Hire Pro", page_icon="🚀", layout="wide")

    # Custom CSS
    st.markdown("""
        <style>
        .stApp { background-color: #f8f9fa; }
        [data-testid="stSidebar"] { background-color: #2c3e50; }
        [data-testid="stSidebar"] * { color: #ecf0f1 !important; }
        h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
        .stButton>button { background-color: #2980b9; color: white; border-radius: 5px; border: none; padding: 10px 20px; font-weight: bold; }
        .stButton>button:hover { background-color: #3498db; color: white; }
        .css-1r6slb0 { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .stSuccess { background-color: #d4edda; color: #155724; }
        </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Go to", ["Candidate View", "Admin Panel"])
    st.sidebar.markdown("---")
    st.sidebar.info("Auto Hire Pro v1.0")

    df = load_data()
    apps_df = load_apps()

    # ---------------- CANDIDATE VIEW ----------------
    if app_mode == "Candidate View":
        # Hero Image
        st.image("https://images.unsplash.com/photo-1586281380349-632531db7ed4?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
        
        st.title("🚀 Career Opportunities")
        st.markdown("### Find your dream job and apply today!")
        st.markdown("---")

        if df.empty:
            st.info("No job openings available at the moment.")
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Select a Role")
                company_list = df["Company"].unique().tolist()
                selected_company = st.selectbox("Choose a Company", company_list)
                if selected_company:
                    st.info(f"You are viewing details for **{selected_company}**")

            with col2:
                if selected_company:
                    company_data = df[df["Company"] == selected_company].iloc[0]
                    
                    with st.container():
                        st.subheader(f"📄 {company_data['Role']} at {selected_company}")
                        
                        jd_file_path = company_data.get("JD_File_Path")
                        if isinstance(jd_file_path, str) and os.path.exists(jd_file_path):
                            with open(jd_file_path, "rb") as f:
                                st.download_button("📥 Download Job Description", f, file_name=os.path.basename(jd_file_path))
                        
                        st.markdown("---")
                        st.subheader("📝 Submit Your Application")
                        with st.form("application_form"):
                            c1, c2 = st.columns(2)
                            with c1:
                                candidate_email = st.text_input("Email Address")
                            with c2:
                                uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx"])
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            apply_btn = st.form_submit_button("🚀 Submit Application")

                            if apply_btn:
                                if not candidate_email or not uploaded_resume:
                                    st.error("⚠️ Please provide both email and resume.")
                                else:
                                    with st.spinner("Analyzing your resume with AI..."):
                                        if not os.path.exists("resumes"):
                                            os.makedirs("resumes")
                                        
                                        resume_path = os.path.join("resumes", f"{selected_company}_{candidate_email}_{uploaded_resume.name}")
                                        with open(resume_path, "wb") as f:
                                            f.write(uploaded_resume.getbuffer())
                                        
                                        # Extract Text
                                        resume_text = ""
                                        if uploaded_resume.name.endswith(".pdf"):
                                            resume_text = extract_text_from_pdf(uploaded_resume)
                                        elif uploaded_resume.name.endswith(".docx"):
                                            resume_text = extract_text_from_docx(uploaded_resume)
                                        
                                        # Calculate Score
                                        score = calculate_score(resume_text, company_data["JD"])
                                        
                                        # Save Application
                                        new_app = {
                                            "Company": selected_company,
                                            "Role": company_data["Role"],
                                            "Email": candidate_email,
                                            "Score": score,
                                            "Resume_Path": resume_path,
                                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        }
                                        apps_df = pd.concat([apps_df, pd.DataFrame([new_app])], ignore_index=True)
                                        save_apps(apps_df)
                                        
                                        st.success(f"✅ Application Submitted! Your Resume Match Score: **{score}/100**")
                                        st.balloons()

    # ---------------- ADMIN PANEL ----------------
    elif app_mode == "Admin Panel":
        if 'admin_logged_in' not in st.session_state:
            st.session_state.admin_logged_in = False

        if not st.session_state.admin_logged_in:
            st.title("🔐 Admin Login")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    if username == "admin" and password == "admin123":
                        st.session_state.admin_logged_in = True
                        st.rerun()
                    else:
                        st.error("❌ Invalid Credentials")
        else:
            if st.sidebar.button("Logout"):
                st.session_state.admin_logged_in = False
                st.rerun()

            # Admin Banner
            st.image("https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&w=1200&q=80", use_container_width=True)
            st.title("📊 Admin Dashboard")
            
            # Tabs for better organization
            tab1, tab2 = st.tabs(["📋 Received Applications", "⚙️ Manage Companies"])
            
            with tab1:
                st.subheader("Recent Applications")
                if apps_df.empty:
                    st.info("No applications received yet.")
                else:
                    # Sort by Score descending
                    sorted_apps = apps_df.sort_values(by="Score", ascending=False)
                    st.dataframe(sorted_apps, use_container_width=True)
            
            with tab2:
                action = st.radio("Choose Action", ["Add New Company", "Edit Existing Company", "Delete Company"], horizontal=True)
                st.markdown("---")

                if action == "Add New Company":
                    st.subheader("➕ Add New Company")
                    with st.form("add_company"):
                        col1, col2 = st.columns(2)
                        with col1:
                            company = st.text_input("Company Name")
                            role = st.text_input("Job Role")
                            resume_threshold = st.slider("Resume Score Threshold", 0, 100, 60)
                        with col2:
                            uploaded_file = st.file_uploader("Upload JD", type=["pdf", "docx"])
                            aptitude_threshold = st.slider("Aptitude Score Threshold", 0, 40, 25)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.form_submit_button("💾 Save New Company"):
                            if not company or not uploaded_file:
                                st.error("⚠️ Missing details!")
                            else:
                                if not os.path.exists("job_descriptions"): os.makedirs("job_descriptions")
                                jd_path = os.path.join("job_descriptions", uploaded_file.name)
                                with open(jd_path, "wb") as f: f.write(uploaded_file.getbuffer())
                                
                                jd_text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_docx(uploaded_file)
                                
                                new_data = {"Company": company, "Role": role, "JD": jd_text, "JD_File_Path": jd_path, "ResumeThreshold": resume_threshold, "AptitudeThreshold": aptitude_threshold}
                                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                                save_data(df)
                                st.success("✅ Saved Successfully!")
                                st.balloons()
                                time.sleep(2)
                                st.rerun()

                elif action == "Edit Existing Company":
                    if df.empty:
                        st.info("No companies.")
                    else:
                        company_to_edit = st.selectbox("Select Company", df["Company"].unique())
                        curr = df[df["Company"] == company_to_edit].iloc[0]
                        with st.form("edit"):
                            new_role = st.text_input("Role", value=curr["Role"])
                            new_file = st.file_uploader("New JD (Optional)")
                            if st.form_submit_button("Update"):
                                if new_file:
                                    jd_path = os.path.join("job_descriptions", new_file.name)
                                    with open(jd_path, "wb") as f: f.write(new_file.getbuffer())
                                    jd_text = extract_text_from_pdf(new_file) if new_file.name.endswith(".pdf") else extract_text_from_docx(new_file)
                                    df.loc[df["Company"] == company_to_edit, ["JD", "JD_File_Path"]] = [jd_text, jd_path]
                                df.loc[df["Company"] == company_to_edit, "Role"] = new_role
                                save_data(df)
                                st.success("✅ Updated Successfully!")
                                st.balloons()
                                time.sleep(2)
                                st.rerun()

                elif action == "Delete Company":
                    if not df.empty:
                        to_del = st.selectbox("Delete", df["Company"].unique())
                        if st.button("Confirm Delete"):
                            df = df[df["Company"] != to_del]
                            save_data(df)
                            st.success("✅ Deleted Successfully!")
                            st.balloons()
                            time.sleep(2)
                            st.rerun()

            st.markdown("---")
            st.subheader("📌 Active Job Listings")
            df = load_data()
            if not df.empty:
                st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
