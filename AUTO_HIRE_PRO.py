# phase1_admin_api.py
import streamlit as st
import pandas as pd
import os
from pypdf import PdfReader
from docx import Document
import google.generativeai as genai
from datetime import datetime
import time
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# ---------------- Config ----------------
COMPANIES_FILE = r"C:\Users\advai\.gemini\antigravity\scratch\auto_hire_pro\companies.xlsx"
APPS_FILE = r"C:\Users\advai\.gemini\antigravity\scratch\auto_hire_pro\applications.csv.xlsx"
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except FileNotFoundError:
    st.error("⚠️ Secrets file not found! Please create .streamlit/secrets.toml")
    st.stop()

# Configure Gemini
genai.configure(api_key=API_KEY)

# ---------------- Data Handling ----------------
def load_data():
    if os.path.exists(COMPANIES_FILE):
        try:
            df = pd.read_excel(COMPANIES_FILE)
            # Ensure columns exist
            required_cols = ["Company", "Role", "JD", "JD_File_Path", "ResumeThreshold", "AptitudeThreshold"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = ""
            df = df.fillna("")
            return df
        except Exception:
            return pd.DataFrame(columns=["Company", "Role", "JD", "JD_File_Path", "ResumeThreshold", "AptitudeThreshold"])
    else:
        df = pd.DataFrame(columns=["Company", "Role", "JD", "JD_File_Path", "ResumeThreshold", "AptitudeThreshold"])
        return df

def save_data(df):
    df.to_excel(COMPANIES_FILE, index=False)

def load_apps():
    if os.path.exists(APPS_FILE):
        return pd.read_excel(APPS_FILE)
    else:
        return pd.DataFrame(columns=["Company", "Role", "Email", "Score", "Resume_Path", "Timestamp"])

def save_apps(df):
    df.to_excel(APPS_FILE, index=False)

# ---------------- Email Notification ----------------
def send_email(candidate_email, score, company, role, email_type="success"):
    try:
        sender_email = st.secrets["EMAIL_ADDRESS"]
        password = st.secrets["EMAIL_PASSWORD"]
    except Exception:
        st.warning("⚠️ Email secrets not found. Skipping email.")
        return

    if email_type == "success":
        subject = f"Congratulations! You've been shortlisted for {role} at {company}"
        heading = "Great News! 🎉"
        heading_color = "#FF9F1C"
        score_color = "#2ecc71"
        body_content = f"""
        <p>We are thrilled to inform you that your profile has been <strong>shortlisted</strong> for the <strong>{role}</strong> position at <strong>{company}</strong>!</p>
        <p>Your Resume Score: <span style="font-size: 18px; font-weight: bold; color: {score_color};">{score}/100</span></p>
        <hr>
        <p>As a next step, we invite you to complete a brief Aptitude Test. Please click the link below to proceed:</p>
        <div style="text-align: center; margin: 30px 0;">
            <a href="https://example.com/aptitude-test" style="background-color: #FF9F1C; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">Start Aptitude Test</a>
        </div>
        <p style="font-size: 12px; color: #888;">Note: This link is valid for 48 hours.</p>
        """
    else:  # rejection
        subject = f"Update on your application for {role} at {company}"
        heading = "Application Update"
        heading_color = "#555"
        score_color = "#e74c3c"
        body_content = f"""
        <p>Thank you for giving us the opportunity to review your application for the <strong>{role}</strong> position at <strong>{company}</strong>.</p>
        <p>We were impressed by your skills; however, the competition strictly required a higher match score for this round.</p>
        <p>Your Resume Score: <span style="font-size: 18px; font-weight: bold; color: {score_color};">{score}/100</span></p>
        <hr>
        <p><strong>Don't be discouraged!</strong> We will keep your resume in our talent pool for future openings that better match your profile.</p>
        """

    html_content = f"""
    <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <div style="background-color: #f4f4f4; padding: 20px;">
                <div style="background-color: white; padding: 30px; border-radius: 10px; max-width: 600px; margin: auto; border-top: 5px solid {heading_color};">
                    <h2 style="color: {heading_color};">{heading}</h2>
                    <p>Dear Candidate,</p>
                    {body_content}
                    <br>
                    <p>Best Regards,<br><strong>Auto Hire Pro Team</strong></p>
                </div>
            </div>
        </body>
    </html>
    """
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = candidate_email
    msg.attach(MIMEText(html_content, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, candidate_email, msg.as_string())
        print(f"✅ {email_type.capitalize()} email sent to {candidate_email}")
    except Exception as e:
        st.error(f"❌ Email Delivery Failed: {e}")
        print(f"❌ Failed to send email: {e}")

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
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use Flash Latest model (Stable alias for 1.5 Flash)
            model = genai.GenerativeModel('gemini-flash-latest')
            
            prompt = f"""
            You are a strict and deterministic Applicant Tracking System (ATS).
            Evaluate the Resume against the Job Description (JD) using the following RIGID scoring rubric.
            
            JOB DESCRIPTION:
            {jd_text}
            
            RESUME:
            {resume_text}
            
            SCORING RUBRIC (Total 100 Points):
            1. **Keywords & Hard Skills (Max 30 points)**: 
               - 30 = All mandatory skills present.
               - 15 = Some key skills missing.
               - 0 = No relevant skills.
            2. **Experience & Seniority (Max 40 points)**:
               - 40 = Exact match or exceeds years/role requirements.
               - 20 = Slightly less experience but relevant.
               - 0 = Mismatch in seniority or years.
            3. **Role Relevance (Max 30 points)**:
               - 30 = Resume is perfectly tailored to this specific job title.
               - 15 = Relevant industry but different role.
               - 0 = Completely irrelevant background.
            
            INSTRUCTIONS:
            - Analyze each section and assign points.
            - SUM the points to get the final score.
            - Your output must be ONLY the final integer score (0-100).
            - Do not provide ranges. Do not provide explanations.
            """
            
            # Strict generation config for maximum determinism
            generation_config = {
                "temperature": 0.0,
                "top_p": 0.0,
                "top_k": 1,
            }
            
            response = model.generate_content(prompt, generation_config=generation_config)
            score = int(''.join(filter(str.isdigit, response.text)))
            return score
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            elif attempt == max_retries - 1:
                st.error(f"AI Error: {e}")
                return 0

# ---------------- Streamlit UI ----------------
def main():
    st.set_page_config(page_title="Auto Hire Pro", page_icon="🚀", layout="wide")

    # Custom CSS - Key Finder Theme
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
        }
        
        .stApp { 
            background-image: linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.8)), url('https://images.unsplash.com/photo-1497366811353-6870744d04b2?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] { 
            background-color: #222222; 
            border-right: 1px solid #333;
        }
        [data-testid="stSidebar"] * { color: #ffffff !important; }
        
        /* Buttons */
        .stButton>button { 
            background-color: #FF9F1C; 
            color: white; 
            border-radius: 8px; 
            border: none; 
            padding: 12px 24px; 
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover { 
            background-color: #e0890b; 
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 159, 28, 0.3);
        }
        
        /* Hero Section */
        .hero-container {
            position: relative;
            background-image: url('https://images.unsplash.com/photo-1522202176988-66273c2fd55f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1351&q=80');
            background-size: cover;
            background-position: center;
            height: 400px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify_content: center;
            margin-bottom: 30px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        }
        .hero-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.5);
        }
        .hero-content {
            position: relative;
            z-index: 1;
            text-align: center;
            color: white;
            padding: 20px;
        }
        .hero-title {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .hero-title span { color: #FF9F1C; }
        .hero-subtitle {
            font-size: 1.2rem;
            font-weight: 300;
            margin-bottom: 30px;
            opacity: 0.9;
        }
        
        /* Cards */
        .job-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 5px solid #FF9F1C;
            margin-bottom: 15px;
        }
        
        /* Enterprise Button Styling */
        div.stButton > button {
            background: linear-gradient(135deg, #FF9F1C 0%, #FF7E00 100%);
            color: white;
            border: none;
            padding: 12px 28px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(255, 159, 28, 0.3);
            width: 100%;
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 159, 28, 0.4);
            background: linear-gradient(135deg, #FFB042 0%, #FF8F00 100%);
        }
        div.stButton > button:active {
            transform: translateY(1px);
        }

        /* Interactive Content Cards */
        .content-card {
            background: rgba(255, 255, 255, 0.98);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            margin-bottom: 25px;
            border-top: 4px solid #FF9F1C;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .content-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        }
        
        /* Form Styling */
        [data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.98);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            border-top: 4px solid #FF9F1C;
        }

        /* Typography */
        h1, h2, h3 {
            font-family: 'Poppins', sans-serif;
            color: #222;
            font-weight: 700;
        }
        p, li, label {
            font-family: 'Poppins', sans-serif;
            color: #444;
            line-height: 1.6;
        }
        
        /* Footer */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #222;
            color: white;
            text-align: center;
            padding: 10px;
            font-size: 12px;
            z-index: 1000;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1995/1995515.png", width=50)
        st.title("Auto Hire Pro")
        st.markdown("---")
        app_mode = st.radio("Navigate", ["Candidate View", "Admin Panel"])
        st.markdown("---")
        st.info("Bring the Spark! ✨")

    df = load_data()
    apps_df = load_apps()

    # ---------------- CANDIDATE VIEW ----------------
    if app_mode == "Candidate View":
        # Custom Hero Section
        st.markdown("""
            <div class="hero-container">
                <div class="hero-overlay"></div>
                <div class="hero-content">
                    <div class="hero-title">Drop Resume & <br> Get Your <span>Desired Job!</span></div>
                    <div class="hero-subtitle">Find Jobs, Employment & Career Opportunities</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if df.empty:
            st.info("No job openings available at the moment.")
        else:
            # Search / Filter Section mimicking the "What" and "Where"
            st.markdown("### 🔍 Find your perfect role")
            col1, col2 = st.columns([2, 1])
            with col1:
                company_list = df["Company"].unique().tolist()
                selected_company = st.selectbox("Select Company / Role", company_list, index=None, placeholder="Choose a company...")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True) # Spacer
                find_jobs_btn = st.button("Find Jobs", use_container_width=True)

            st.markdown("---")

            # Session State Logic
            if find_jobs_btn:
                if selected_company:
                    st.session_state['viewing_company'] = selected_company
                else:
                    st.warning("⚠️ Please select a company first.")

            # Display Job Details if a company is "viewing"
            if st.session_state.get('viewing_company'):
                view_company = st.session_state['viewing_company']
                # Ensure the company still exists in data (in case of deletion)
                if view_company in df["Company"].values:
                    company_data = df[df["Company"] == view_company].iloc[0]
                    
                    # Job Details Card
                    st.markdown(f"""
                        <div class="job-card">
                            <h3>{company_data['Role']}</h3>
                            <p style="color: #666;">at <strong>{view_company}</strong></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.markdown('<div class="content-card">', unsafe_allow_html=True)
                    st.subheader("Job Description")
                    jd_file_path = company_data.get("JD_File_Path")
                    if isinstance(jd_file_path, str) and os.path.exists(jd_file_path):
                        with open(jd_file_path, "rb") as f:
                            st.download_button("📥 Download JD File", f, file_name=os.path.basename(jd_file_path))
                    else:
                        st.warning("JD File not available.")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_right:
                    st.subheader("Apply Now")
                    with st.form("application_form"):
                        candidate_email = st.text_input("Email Address", placeholder="you@example.com")
                        uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx"])
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        apply_btn = st.form_submit_button("🚀 Submit Application")

                        if apply_btn:
                            if not candidate_email or not uploaded_resume:
                                st.error("⚠️ Please provide both email and resume.")
                            else:
                                with st.spinner("Analyzing your resume..."):
                                    if not os.path.exists("resumes"):
                                        os.makedirs("resumes")
                                    
                                    resume_path = os.path.join("resumes", f"{view_company}_{candidate_email}_{uploaded_resume.name}")
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
                                        "Company": view_company,
                                        "Role": company_data["Role"],
                                        "Email": candidate_email,
                                        "Score": score,
                                        "Resume_Path": resume_path,
                                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    apps_df = pd.concat([apps_df, pd.DataFrame([new_app])], ignore_index=True)
                                    save_apps(apps_df)
                                    
                                    # Logic for Emails based on Dynamic Threshold
                                    try:
                                        threshold = int(company_data.get("ResumeThreshold", 60))
                                    except:
                                        threshold = 60

                                    if score >= threshold:
                                        send_email(candidate_email, score, view_company, company_data["Role"], email_type="success")
                                        st.success(f"✅ Application Submitted! 🎉 Resume Score: **{score}/100**")
                                        st.balloons()
                                        st.caption("📧 A confirmation email with the Test Link has been sent to your inbox.")
                                    else:
                                        send_email(candidate_email, score, view_company, company_data["Role"], email_type="rejection")
                                        st.warning(f"✅ Application Submitted. Resume Score: **{score}/100**")
                                        st.caption("📧 An update email has been sent to your inbox.")

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
