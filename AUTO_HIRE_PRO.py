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

# ---------------- High-End UI Implementation ----------------
def main():
    st.set_page_config(page_title="Auto Hire Pro", page_icon="⚡", layout="wide")

    # ---------------- GLOBAL CSS INJECTION ----------------
    st.markdown("""
        <style>
        /* Import Google Fonts: Outfit (Modern Geometric) */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* -------- ROOT VARIABLES -------- */
        :root {
            --bg-color: #0F172A;
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --accent-primary: #8B5CF6; /* Violet */
            --accent-secondary: #06B6D4; /* Cyan */
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
        }

        /* -------- GLOBAL RESETS -------- */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
        }

        /* -------- BACKGROUND ANIMATION -------- */
        .stApp {
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.15) 0%, transparent 40%),
                url('https://images.unsplash.com/photo-1620641788421-7f1c3374c752?q=80&w=2070&auto=format&fit=crop'); 
            /* Abstract Dark Grainy Background */
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-size: cover;
        }

        /* -------- SIDEBAR -------- */
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--glass-border);
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        /* -------- GLASSMORPHISM CARDS -------- */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
            margin-bottom: 25px;
        }

        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 45px rgba(139, 92, 246, 0.2);
            border-color: rgba(139, 92, 246, 0.5);
        }

        /* -------- TYPOGRAPHY -------- */
        h1, h2, h3 {
            color: var(--text-primary);
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(to right, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .highlight {
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }

        p, li, label {
            color: var(--text-secondary);
            font-weight: 300;
        }

        /* -------- BUTTONS (CUSTOM GRADIENT) -------- */
        div.stButton > button {
            background: linear-gradient(135deg, var(--accent-primary) 0%, #6366F1 100%);
            color: white;
            border: none;
            padding: 12px 32px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 5px 15px rgba(139, 92, 246, 0.4);
            letter-spacing: 0.5px;
            width: 100%;
        }
        
        div.stButton > button:hover {
            transform: scale(1.02) translateY(-2px);
            box-shadow: 0 10px 25px rgba(139, 92, 246, 0.6);
            background: linear-gradient(135deg, #7C3AED 0%, #4F46E5 100%);
        }

        div.stButton > button:active {
            transform: scale(0.98);
        }

        /* -------- INPUTS -------- */
        [data-testid="stTextInput"], [data-testid="stSelectbox"] {
            border-radius: 12px !important;
        }
        
        div.stTextInput > div > div > input {
            background-color: rgba(255, 255, 255, 0.05);
            color: white;
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            padding: 12px;
        }

        div.stTextInput > div > div > input:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2);
            color: white;
        }

        /* -------- ANIMATIONS -------- */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translate3d(0, 40px, 0);
            }
            to {
                opacity: 1;
                transform: translate3d(0, 0, 0);
            }
        }

        .animate-enter {
            animation-name: fadeInUp;
            animation-duration: 0.8s;
            animation-fill-mode: both;
        }

        .delay-1 { animation-delay: 0.2s; }
        .delay-2 { animation-delay: 0.4s; }
        .delay-3 { animation-delay: 0.6s; }
        
        /* -------- HERO SECTION -------- */
        .hero {
            position: relative;
            padding: 80px 40px;
            border-radius: 24px;
            overflow: hidden;
            background: radial-gradient(circle at top right, rgba(139, 92, 246, 0.2), transparent 50%),
                        linear-gradient(to bottom right, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 40px;
            text-align: center;
        }
        
        .hero h1 {
            font-size: 4rem;
            margin-bottom: 20px;
            line-height: 1.1;
            text-shadow: 0 4px 30px rgba(0,0,0,0.5);
        }
        
        .hero p {
            font-size: 1.4rem;
            max-width: 600px;
            margin: 0 auto;
            color: #cbd5e1;
        }
        
        /* -------- TABS STYLING -------- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 8px;
            color: #94A3B8;
            font-size: 16px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(139, 92, 246, 0.1);
            color: #F8FAFC;
        }

        </style>
    """, unsafe_allow_html=True)

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>⚡ Auto Hire Pro</h2>", unsafe_allow_html=True)
        st.write("") # Spacer
        
        app_mode = st.radio("Navigation", ["Candidate Experience", "Admin Commander"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("""
            <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;'>
                <p style='margin:0; font-size: 12px; color: #64748B;'>SYSTEM STATUS</p>
                <p style='margin:0; color: #10B981; font-weight: 600;'>● Online</p>
                <p style='margin:5px 0 0 0; font-size: 12px; color: #64748B;'>POWERED BY</p>
                <p style='margin:0; color: #F8FAFC; font-weight: 600;'>Gemini 1.5 Flash</p>
            </div>
        """, unsafe_allow_html=True)

    df = load_data()
    apps_df = load_apps()

    # ---------------- CANDIDATE LAYOUT ----------------
    if app_mode == "Candidate Experience":
        # Hero Section
        st.markdown("""
            <div class="hero animate-enter">
                <h1>Unlock Your <br><span class="highlight">Future Career</span></h1>
                <p>Advanced AI-powered matching ensures your resume gets the attention it deserves.</p>
            </div>
        """, unsafe_allow_html=True)

        if df.empty:
            st.info("No active positions detected in the neural network.")
        else:
            # Layout: Search on Top
            col1, col2 = st.columns([3, 1])
            with col1:
                company_options = df["Company"].unique().tolist()
                selected_company = st.selectbox("Search Companies", company_options, index=None, placeholder="Select a target organization...")
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                find_btn = st.button("Initialize Search", use_container_width=True)

            if find_btn:
                if selected_company:
                    st.session_state['viewing_company'] = selected_company
                else:
                    st.toast("⚠️ Select a company to proceed.", icon="⚠️")

            # Main Job View
            if st.session_state.get('viewing_company') and st.session_state['viewing_company'] in df["Company"].values:
                view_company = st.session_state['viewing_company']
                company_data = df[df["Company"] == view_company].iloc[0]

                st.markdown("---")
                
                # Dynamic Grid Layout
                grid_c1, grid_c2 = st.columns([1, 1.5])

                with grid_c1:
                    st.markdown(f"""
                        <div class="glass-card animate-enter delay-1">
                            <p style='font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: var(--accent-secondary);'>Company Profile</p>
                            <h2 style='margin: 5px 0 15px 0;'>{view_company}</h2>
                            <h3 style='font-weight: 400; color: white;'>{company_data['Role']}</h3>
                            <div style='margin-top: 20px; display: flex; gap: 10px;'>
                                <span style='background: rgba(255,255,255,0.1); padding: 5px 12px; border-radius: 20px; font-size: 12px;'>Full Time</span>
                                <span style='background: rgba(255,255,255,0.1); padding: 5px 12px; border-radius: 20px; font-size: 12px;'>Remote Friendly</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # JD Download Button
                    st.markdown('<div class="glass-card animate-enter delay-2">', unsafe_allow_html=True)
                    st.markdown("### Job Asset Info")
                    jd_file = company_data.get("JD_File_Path")
                    if isinstance(jd_file, str) and os.path.exists(jd_file):
                        with open(jd_file, "rb") as f:
                            st.download_button("⬇️ Download Job Spec", f, file_name=os.path.basename(jd_file), mime="application/octet-stream", use_container_width=True)
                    else:
                        st.write("No downloadable assets.")
                    st.markdown('</div>', unsafe_allow_html=True)

                with grid_c2:
                    st.markdown('<div class="glass-card animate-enter delay-1">', unsafe_allow_html=True)
                    st.markdown("### 🚀 Submit Application")
                    st.write("Our AI agents will analyze your profile instantly.")
                    
                    with st.form("application_form_modern"):
                        email_input = st.text_input("Applicant Email", placeholder="you@domain.com")
                        resume_file = st.file_uploader("Upload Resume (PDF/DOCX Only)", type=["pdf", "docx"])
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        submitted = st.form_submit_button("Start Analysis Sequence")
                        
                        if submitted:
                            if not email_input or not resume_file:
                                st.error("❌ Protocols Incomplete: Email and Resume required.")
                            else:
                                with st.status("Initializing Neural Analysis...", expanded=True) as status:
                                    st.write("📂 Securely uploading resume...")
                                    if not os.path.exists("resumes"): os.makedirs("resumes")
                                    resume_path = os.path.join("resumes", f"{view_company}_{email_input}_{resume_file.name}")
                                    with open(resume_path, "wb") as f: f.write(resume_file.getbuffer())
                                    time.sleep(0.5)

                                    st.write("🔍 Extracting skills and experience...")
                                    resume_text = ""
                                    if resume_file.name.endswith(".pdf"):
                                        resume_text = extract_text_from_pdf(resume_file)
                                    else:
                                        resume_text = extract_text_from_docx(resume_file)
                                    
                                    st.write("🧠 AI evaluating against Job Description...")
                                    score = calculate_score(resume_text, company_data["JD"])
                                    
                                    status.update(label="Analysis Complete", state="complete", expanded=False)
                                
                                # Process Result (Same Logic)
                                new_app = {
                                    "Company": view_company,
                                    "Role": company_data["Role"],
                                    "Email": email_input,
                                    "Score": score,
                                    "Resume_Path": resume_path,
                                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                apps_df = pd.concat([apps_df, pd.DataFrame([new_app])], ignore_index=True)
                                save_apps(apps_df)
                                
                                try:
                                    threshold = int(company_data.get("ResumeThreshold", 60))
                                except:
                                    threshold = 60

                                if score >= threshold:
                                    send_email(email_input, score, view_company, company_data["Role"], email_type="success")
                                    st.balloons()
                                    st.markdown(f"""
                                        <div style="background: rgba(46, 204, 113, 0.2); border: 1px solid #2ecc71; padding: 20px; border-radius: 12px; text-align: center; margin-top: 20px;">
                                            <h2 style="color: #2ecc71; margin:0;">MATCH SUCCESS: {score}/100</h2>
                                            <p style="color: white;">You have passed the preliminary screening.</p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    send_email(email_input, score, view_company, company_data["Role"], email_type="rejection")
                                    st.markdown(f"""
                                        <div style="background: rgba(231, 76, 60, 0.2); border: 1px solid #e74c3c; padding: 20px; border-radius: 12px; text-align: center; margin-top: 20px;">
                                            <h2 style="color: #e74c3c; margin:0;">MATCH SCORE: {score}/100</h2>
                                            <p style="color: white;">Profile stored for future matching.</p>
                                        </div>
                                    """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- ADMIN LAYOUT ----------------
    elif app_mode == "Admin Commander":
        if 'admin_logged_in' not in st.session_state:
            st.session_state.admin_logged_in = False

        if not st.session_state.admin_logged_in:
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.markdown('<div class="glass-card animate-enter">', unsafe_allow_html=True)
                st.subheader("🔐 Restricted Access")
                with st.form("login_form"):
                    username = st.text_input("Identity")
                    password = st.text_input("Passcode", type="password")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.form_submit_button("Authenticate"):
                        if username == "admin" and password == "admin123":
                            st.session_state.admin_logged_in = True
                            st.rerun()
                        else:
                            st.error("⛔ Access Denied")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Admin Dashboard
            st.title("Admin Command Center")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                    <div class="glass-card">
                        <h4 style="margin:0; color: var(--text-secondary);">Total Applications</h4>
                        <h1 style="margin:0; font-size: 3rem; color: var(--accent-primary);">{len(apps_df)}</h1>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                avg_score = int(apps_df["Score"].mean()) if not apps_df.empty else 0
                st.markdown(f"""
                    <div class="glass-card">
                        <h4 style="margin:0; color: var(--text-secondary);">Avg. Quality Score</h4>
                        <h1 style="margin:0; font-size: 3rem; color: var(--accent-secondary);">{avg_score}</h1>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                cols = st.columns([1,1])
                with cols[1]:
                    if st.button("Logout"):
                        st.session_state.admin_logged_in = False
                        st.rerun()

            tab1, tab2 = st.tabs(["📋 Application Feed", "⚙️ Company Matrix"])

            with tab1:
                if apps_df.empty:
                    st.info("No incoming data streams.")
                else:
                    st.dataframe(
                        apps_df.sort_values(by="Score", ascending=False),
                        use_container_width=True,
                        column_config={
                            "Score": st.column_config.ProgressColumn(
                                "Match Score",
                                help="AI Calculated Match",
                                format="%d",
                                min_value=0,
                                max_value=100,
                            ),
                        }
                    )
            
            with tab2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                action = st.radio("Operation Mode", ["Add Entity", "Modify Entity", "Delete Entity"], horizontal=True)
                
                if action == "Add Entity":
                    with st.form("add_company"):
                        c_a, c_b = st.columns(2)
                        with c_a:
                            company = st.text_input("Company Name")
                            role = st.text_input("Role Title")
                        with c_b:
                            uploaded_file = st.file_uploader("Upload JD Spec")
                        
                        st.divider()
                        c_c, c_d = st.columns(2)
                        with c_c:
                            resume_threshold = st.slider("Min Resume Score", 0, 100, 60)
                        with c_d:
                            aptitude_threshold = st.slider("Min Aptitude Score", 0, 40, 25)
                        
                        if st.form_submit_button("Deploy New Entity"):
                            if not company or not uploaded_file:
                                st.error("⚠️ Data incomplete.")
                            else:
                                if not os.path.exists("job_descriptions"): os.makedirs("job_descriptions")
                                jd_path = os.path.join("job_descriptions", uploaded_file.name)
                                with open(jd_path, "wb") as f: f.write(uploaded_file.getbuffer())
                                jd_text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_docx(uploaded_file)
                                new_data = {"Company": company, "Role": role, "JD": jd_text, "JD_File_Path": jd_path, "ResumeThreshold": resume_threshold, "AptitudeThreshold": aptitude_threshold}
                                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                                save_data(df)
                                st.success("✅ Entity Active")
                                time.sleep(1)
                                st.rerun()

                elif action == "Modify Entity":
                    if df.empty:
                        st.warning("No entities to modify.")
                    else:
                        company_to_edit = st.selectbox("Select Target", df["Company"].unique())
                        curr = df[df["Company"] == company_to_edit].iloc[0]
                        with st.form("edit_form"):
                            new_role = st.text_input("Role Title", value=curr["Role"])
                            new_file = st.file_uploader("Override JD Spec (Optional)")
                            if st.form_submit_button("Update Entity"):
                                if new_file:
                                    jd_path = os.path.join("job_descriptions", new_file.name)
                                    with open(jd_path, "wb") as f: f.write(new_file.getbuffer())
                                    jd_text = extract_text_from_pdf(new_file) if new_file.name.endswith(".pdf") else extract_text_from_docx(new_file)
                                    df.loc[df["Company"] == company_to_edit, ["JD", "JD_File_Path"]] = [jd_text, jd_path]
                                df.loc[df["Company"] == company_to_edit, "Role"] = new_role
                                save_data(df)
                                st.success("✅ Entity Updated")
                                time.sleep(1)
                                st.rerun()

                elif action == "Delete Entity":
                    if not df.empty:
                        to_del = st.selectbox("Select Target to Purge", df["Company"].unique())
                        if st.button("EXECUTE PURGE", type="primary"):
                            df = df[df["Company"] != to_del]
                            save_data(df)
                            st.success("✅ Entity Purged")
                            time.sleep(1)
                            st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
