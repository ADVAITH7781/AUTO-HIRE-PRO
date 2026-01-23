# phase1_admin_api.py
import streamlit as st
import pandas as pd
import os
from pypdf import PdfReader
from docx import Document
import google.generativeai as genai
import datetime
import time
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------- SAFE IMPORTS FOR CV ----------------
try:
    import cv2
    import mediapipe as mp
    import av
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    import numpy as np
    PROCTORING_AVAILABLE = True
except ImportError as e:
    PROCTORING_AVAILABLE = False
    print(f"⚠️ CV Import Error: {e}")
    # Define a dummy class to prevent NameError later
    class VideoTransformerBase: pass
    webrtc_streamer = None
except Exception as e:
    PROCTORING_AVAILABLE = False
    print(f"⚠️ Unexpected CV Error: {e}")
    class VideoTransformerBase: pass
    webrtc_streamer = None

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# Files moved to data/ folder for better organization & persistence check
COMPANIES_FILE = os.path.join(DATA_DIR, "companies.xlsx")
APPS_FILE = os.path.join(DATA_DIR, "applications.csv.xlsx")
RESUMES_DIR = os.path.join(BASE_DIR, "resumes")
JOBS_DIR = os.path.join(BASE_DIR, "job_descriptions")
QUESTIONS_DIR = os.path.join(BASE_DIR, "questions")

# Ensure directories exist
for d in [RESUMES_DIR, JOBS_DIR, QUESTIONS_DIR, DATA_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except FileNotFoundError:
    st.error("⚠️ Secrets file not found! Please create .streamlit/secrets.toml")

# ---------------- CV PROCTORING LOGIC ----------------
if PROCTORING_AVAILABLE:
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    except Exception as e:
        PROCTORING_AVAILABLE = False
        print(f"⚠️ MediaPipe Init Error: {e}")

class ProctoringProcessor(VideoTransformerBase):
    def __init__(self):
        self.warn_count = 0
        self.last_warn = 0

    def recv(self, frame):
        if not PROCTORING_AVAILABLE:
            # Pass-through if CV is broken
            return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")
            
        img = frame.to_ndarray(format="bgr24")
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            
            h, w, _ = img.shape
            face_count = 0
            status_text = "Secure"
            color = (0, 255, 0)
            
            if results.multi_face_landmarks:
                face_count = len(results.multi_face_landmarks)
                
                if face_count > 1:
                    status_text = "MULTIPLE FACES DETECTED!"
                    color = (0, 0, 255)
                elif face_count == 1:
                    for face_landmarks in results.multi_face_landmarks:
                        # Head Pose Estimation (Simple Nose vs Ear X-coord check)
                        nose = face_landmarks.landmark[1]
                        left_ear = face_landmarks.landmark[234]
                        right_ear = face_landmarks.landmark[454]
                        
                        # Convert to pixel coords
                        nx, ny = int(nose.x * w), int(nose.y * h)
                        lx, _ = int(left_ear.x * w), int(left_ear.y * h)
                        rx, _ = int(right_ear.x * w), int(right_ear.y * h)
                        
                        # Check deviation
                        dist_l = abs(nx - lx)
                        dist_r = abs(nx - rx)
                        
                        ratio = dist_l / (dist_r + 1e-6)
                        
                        if ratio < 0.5: # Looking Left
                            status_text = "LOOKING AWAY (LEFT)"
                            color = (0, 165, 255)
                        elif ratio > 2.0: # Looking Right
                            status_text = "LOOKING AWAY (RIGHT)"
                            color = (0, 165, 255)
                            
                        # Draw Nose
                        cv2.circle(img, (nx, ny), 5, (255, 0, 0), -1)
            else:
                status_text = "NO FACE DETECTED"
                color = (0, 0, 255)
                
            # Draw Status
            cv2.putText(img, f"Status: {status_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Border for warning
            if color != (0, 255, 0):
                cv2.rectangle(img, (0,0), (w,h), color, 10)
                
        except Exception:
            pass # Fail silently during processing
             
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Configure Gemini
genai.configure(api_key=API_KEY)

# ---------------- Data Handling ----------------
def load_data():
    if os.path.exists(COMPANIES_FILE):
        try:
            df = pd.read_excel(COMPANIES_FILE)
            required_cols = ["Company", "Role", "JD", "JD_File_Path", "ResumeThreshold", "AptitudeThreshold", "Job_ID"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = ""
            df = df.fillna("")
            # st.write(f"DEBUG: Loaded data from {COMPANIES_FILE}, Records: {len(df)}") # Uncomment for UI debug
            return df
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return pd.DataFrame(columns=["Company", "Role", "JD", "JD_File_Path", "ResumeThreshold", "AptitudeThreshold", "Job_ID"])
    else:
        print(f"ℹ️ File not found: {COMPANIES_FILE}, creating new.")
        df = pd.DataFrame(columns=["Company", "Role", "JD", "JD_File_Path", "ResumeThreshold", "AptitudeThreshold", "Job_ID"])
        return df

def save_data(df):
    try:
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        df.to_excel(COMPANIES_FILE, index=False)
        st.toast("✅ Jobs Database Saved!", icon="💾") 
    except Exception as e:
        st.error(f"❌ CRITICAL ERROR SAVING DATA: {e}")

def load_apps():
    if os.path.exists(APPS_FILE):
        try:
            df = pd.read_excel(APPS_FILE)
            cols = ["Name", "Email", "Score", "Company", "Role", "Status", "Resume_Text", "TestPassword", "TokenTime", "TestScore", "TestStatus", "Resume_Path", "Timestamp"]
            for col in cols:
                if col not in df.columns: df[col] = ""
            return df
        except Exception:
            return pd.DataFrame(columns=["Name", "Email", "Score", "Company", "Role", "Status", "Resume_Text", "TestPassword", "TokenTime", "TestScore", "TestStatus", "Resume_Path", "Timestamp"])
    else:
        return pd.DataFrame(columns=["Name", "Email", "Score", "Company", "Role", "Status", "Resume_Text", "TestPassword", "TokenTime", "TestScore", "TestStatus", "Resume_Path", "Timestamp"])

def save_apps(df):
    try:
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        df.to_excel(APPS_FILE, index=False)
    except Exception as e:
        st.error(f"❌ Error Saving Applications: {e}")

# ---------------- Email Notification ----------------
def send_email(candidate_email, score, company, role, email_type="success"):
    token = ""
    try:
        sender_email = st.secrets["EMAIL_ADDRESS"]
        password = st.secrets["EMAIL_PASSWORD"]
    except Exception:
        st.warning("⚠️ Email secrets not found. Skipping email.")
        return ""

    if email_type == "success":
        # Generate Secure Token
        chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        token = "".join(random.choices(chars, k=6))
        
        subject = f"Congratulations! You've been shortlisted for {role} at {company}"
        heading = "Great News! 🎉"
        heading_color = "#FF9F1C"
        score_color = "#2ecc71"
        try:
            base_url = st.secrets.get("BASE_URL", "http://localhost:8501")
        except:
            base_url = "http://localhost:8501"
            
        body_content = f"""
        <p>We are thrilled to inform you that your profile has been <strong>shortlisted</strong> for the <strong>{role}</strong> position at <strong>{company}</strong>!</p>
        <p>Your Resume Score: <span style="font-size: 18px; font-weight: bold; color: {score_color};">{score}/100</span></p>
        <hr>
        <p>You have been invited to take the <strong>Proctored Aptitude Test</strong>.</p>
        <div style="background: #fdf2f8; padding: 15px; border-left: 4px solid #db2777; margin: 20px 0;">
            <p style="margin:0; font-weight:bold; color:#be185d;">Your Access Credentials:</p>
            <p style="margin:5px 0 0 0;">Test Password: <span style="font-size: 1.25em; background: #fff; padding: 2px 8px; border: 1px solid #ddd; border-radius: 4px;">{token}</span></p>
            <p style="margin:5px 0 0 0; font-size: 0.85em; color: #666;">⚠️ Valid for 30 Hours only.</p>
        </div>
        <p>Please click the link below to proceed:</p>
        <div style="text-align: center; margin: 30px 0;">
            <a href="{base_url}/?mode=test" style="background-color: #FF9F1C; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">Start Aptitude Test</a>
        </div>
        """
    else:
        subject = f"Update on your application for {role} at {company}"
        heading = "Application Update"
        heading_color = "#555"
        score_color = "#e74c3c"
        body_content = f"""
        <p>Thank you for giving us the opportunity to review your application for the <strong>{role}</strong> position at <strong>{company}</strong>.</p>
        <p>Your Resume Score: <span style="font-size: 18px; font-weight: bold; color: {score_color};">{score}/100</span></p>
        <hr>
        <p><strong>Don't be discouraged!</strong> We will keep your resume in our talent pool for future openings.</p>
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
        return token
    except Exception as e:
        st.error(f"❌ Email Delivery Failed: {e}")
        print(f"❌ Failed to send email: {e}")
        return ""

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            try:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
            except Exception as e:
                print(f"⚠️ Error parsing PDF page: {e}")
                continue
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def calculate_score(resume_text, jd_text):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel('gemini-flash-latest')
            prompt = f"""
            You are a strict ATS. Evaluate Resume against JD.
            JD: {jd_text}
            RESUME: {resume_text}
            SCORING (Max 100):
            1. Skills (30)
            2. Experience (40)
            3. Relevance (30)
            Output ONLY final integer score.
            """
            generation_config = {"temperature": 0.0, "top_p": 0.0, "top_k": 1}
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

import json
import random

# ---------------- Question Bank Logic ----------------
QUESTIONS_DIR = os.path.join(BASE_DIR, "questions")
if not os.path.exists(QUESTIONS_DIR): os.makedirs(QUESTIONS_DIR)

def generate_question_bank(jd_text, job_id):
    """Generates 50 Technical Qs (Job Specific) + reuses 50 General Qs (Common Pool)."""
    
    # 1. SETUP COMMON POOL (Logical + Situational) - Reuse if exists
    common_file = os.path.join(QUESTIONS_DIR, "common_pool_v1.json")
    common_questions = []
    
    if os.path.exists(common_file):
        with open(common_file, "r") as f:
            common_questions = json.load(f)
    else:
        # Generate Common Pool ONCE
        common_topics = [
            "Logical Reasoning & IQ (General)",
            "Situational Judgment (Professional Workplace)",
        ]
        model = genai.GenerativeModel('gemini-flash-latest')
        for topic in common_topics:
            try:
                # Ask for 25 each
                prompt = f"""
                Generate 25 Multiple Choice Questions (MCQs).
                FOCUS AREA: {topic}
                TAG: General
                OUTPUT FORMAT (JSON): [{{"q": "...", "options": [...], "answer": "...", "type": "General"}}]
                """
                response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                batch = json.loads(response.text)
                if isinstance(batch, list):
                    # Ensure type tag exists
                    for b in batch: b["type"] = "General"
                    common_questions.extend(batch)
                time.sleep(1)
            except Exception as e:
                print(f"⚠️ Error gen common pool {topic}: {e}")
        
        # Save Common Pool
        with open(common_file, "w") as f:
            json.dump(common_questions, f)

    # 2. GENERATE JOB SPECIFIC TECHNICAL QUESTIONS (50 Qs)
    tech_questions = []
    tech_topics = [
        "Technical Skills in JD (Hard)",
        "Advanced Role-Specific Scenarios"
    ]
    model = genai.GenerativeModel('gemini-flash-latest')
    
    for topic in tech_topics:
        try:
            prompt = f"""
            Act as a Senior Tech Interviewer. Generate 25 Hard MCQs for this Job Description.
            JD SUMMARY: {jd_text[:1500]}...
            FOCUS AREA: {topic}
            TAG: Technical
            OUTPUT FORMAT (JSON): [{{"q": "...", "options": [...], "answer": "...", "type": "Technical"}}]
            """
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            batch = json.loads(response.text)
            if isinstance(batch, list):
                for b in batch: b["type"] = "Technical"
                tech_questions.extend(batch)
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ Error gen tech batch {topic}: {e}")

    # 3. MERGE & SAVE
    full_bank = tech_questions + common_questions
    
    # Save to JSON
    q_file = os.path.join(QUESTIONS_DIR, f"{job_id}.json")
    with open(q_file, "w") as f:
        json.dump(full_bank, f)
        
    # Save to Word (DOCX)
    try:
        doc = Document()
        doc.add_heading(f'Question Bank: {job_id}', 0)
        
        # Section 1: Technical
        doc.add_heading('Part 1: Technical (Job Specific)', level=1)
        for i, q in enumerate(tech_questions):
            doc.add_paragraph(f"T{i+1}. {q.get('q')}", style='List Number')
            for opt in q.get('options', []): doc.add_paragraph(opt, style='List Bullet')
            doc.add_paragraph(f"Answer: {q.get('answer')}", style='Intense Quote')
            
        # Section 2: General
        doc.add_heading('Part 2: General (Common Pool)', level=1)
        for i, q in enumerate(common_questions):
            doc.add_paragraph(f"G{i+1}. {q.get('q')}", style='List Number')
            for opt in q.get('options', []): doc.add_paragraph(opt, style='List Bullet')
            doc.add_paragraph(f"Answer: {q.get('answer')}", style='Intense Quote')
            
        docx_path = os.path.join(BASE_DIR, f"{job_id}_Question_Bank.docx")
        doc.save(docx_path)
    except Exception as e:
        print(f"⚠️ Error saving DOCX: {e}")

    return len(full_bank)

def get_candidate_questions(job_id, num_questions=40):
    """Samples 25 Technical + 15 General Questions."""
    q_file = os.path.join(QUESTIONS_DIR, f"{job_id}.json")
    if not os.path.exists(q_file): return []
    
    with open(q_file, "r") as f:
        bank = json.load(f)
        
    tech = [q for q in bank if q.get("type") == "Technical"]
    general = [q for q in bank if q.get("type") == "General"]
    
    # Fallback if tags missing (older files)
    if not tech and not general:
        return random.sample(bank, min(len(bank), num_questions))
        
    # Stratified Sampling (25 Tech, 15 General => 40 Total)
    # If not enough, take what we have
    n_tech = min(len(tech), 25)
    n_gen = min(len(general), 15)
    
    exam_set = random.sample(tech, n_tech) + random.sample(general, n_gen)
    random.shuffle(exam_set)
    return exam_set

# ---------------- VIBRANT SAAS UI IMPLEMENTATION ----------------
def main():
    st.set_page_config(page_title="Auto Hire Pro", page_icon="🚀", layout="wide")

    # ---------------- CSS: ORANGE & WHITE SAAS THEME ----------------
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        :root {
            --primary: #FF6B00;
            --primary-hover: #E65A00;
            --secondary: #FFF7ED; 
            --text-main: #1E293B;
            --text-light: #64748B;
            --bg-page: #FFFFFF;
            --bg-card: #FFFFFF;
            --border-color: #E2E8F0;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--text-main);
            background-color: var(--bg-page);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #FAFAFA;
            border-right: 1px solid var(--border-color);
        }
        
        /* Typography */
        h1, h2, h3, h4 {
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.025em;
        }
        
        .hero-title {
            font-size: 3.5rem;
            line-height: 1.1;
            margin-bottom: 2rem;
            background: linear-gradient(to right, #0F172A, #334155);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .highlight-orange {
            color: var(--primary);
            -webkit-text-fill-color: var(--primary);
        }

        /* Buttons (SaaS Style) */
        div.stButton > button {
            background-color: var(--primary) !important;
            color: white !important;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 600;
            border: none;
            box-shadow: 0 4px 6px -1px rgba(255, 107, 0, 0.3);
            transition: all 0.2s ease;
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 15px -3px rgba(255, 107, 0, 0.4);
            background-color: var(--primary-hover) !important;
        }

        /* Cards */
        .saas-card {
            background: white;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
            height: 100%;
        }
        .saas-card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            border-color: #CBD5E1;
            transform: translateY(-2px);
        }

        /* Inputs */
        .stTextInput input, .stSelectbox [data-baseweb="select"] {
            border-radius: 8px;
            border: 1px solid #CBD5E1;
        }
        .stTextInput input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(255, 107, 0, 0.2);
        }

        /* Stats Badge */
        .stat-badge {
            display: inline-block;
            background-color: var(--secondary);
            color: var(--primary);
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.875rem;
            margin-bottom: 1rem;
        }
        
        /* Steps */
        .step-num {
            width: 40px; 
            height: 40px; 
            background: var(--secondary); 
            color: var(--primary); 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            border-radius: 50%;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        
        </style>
    """, unsafe_allow_html=True)

    # ---------------- SIDEBAR ----------------
    if st.query_params.get("mode") == "test":
        mode = "Take Aptitude Test"
        st.sidebar.empty() # Hide Sidebar in Test Mode
    else:
        with st.sidebar:
            st.markdown(f"""
                <div style="padding: 1rem 0;">
                    <h2 style="color: var(--primary); margin:0;">Auto Hire<span style="color:#0F172A">Pro</span></h2>
                    <p style="color: var(--text-light); font-size: 0.875rem;">Intelligent Hiring Platform</p>
                </div>
            """, unsafe_allow_html=True)
            
            mode = st.radio("Workspace", ["Job Seekers", "Admin Dashboard"], label_visibility="collapsed")

    df = load_data()
    apps_df = load_apps()
    
    # ---------------- HELPER: VERIFY TOKEN ----------------
    def verify_token(email, password):
        user = apps_df[apps_df['Email'] == email]
        if user.empty: return False, "Email not found."
        
        user = user.iloc[0]
        if user['Status'] != 'Shortlisted': return False, "You have not been shortlisted yet."
        
        stored_pass = str(user['TestPassword']).strip()
        if stored_pass != password.strip(): return False, "Invalid Password."
        
        # Check Expiry (30 Hours)
        try:
            token_time = pd.to_datetime(user['TokenTime'])
            now = pd.Timestamp.now()
            diff = now - token_time
            if diff.total_seconds() > 30 * 3600:
                return False, "Link Expired (Valid for 30hrs only)."
        except:
             return False, "Invalid Token Data."
             
        return True, user

    # ---------------- TEST PORTAL ----------------
    if mode == "Take Aptitude Test":
        st.markdown("<h1 style='text-align: center; color: #FF6B00;'>🔐 Candidate Test Portal</h1>", unsafe_allow_html=True)
        
        if 'test_session' not in st.session_state:
            st.session_state.test_session = None
            
        if not st.session_state.test_session:
            with st.form("test_login"):
                st.subheader("Secure Login")
                email = st.text_input("Registered Email")
                pwd = st.text_input("Test Password (from Email)")
                
                if st.form_submit_button("Start Test"):
                    valid, res = verify_token(email, pwd)
                    if valid:
                        st.session_state.test_session = res
                        st.success("Verified! Loading Test Environmet...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Access Denied: {res}")
        else:
            # APTITUDE TEST INTERFACE
            user = st.session_state.test_session
            st.title(f"📝 Aptitude Test: {user['Role']}")
            
            # --- JAVASCRIPT PROCTORING ---
            st.components.v1.html("""
                <script>
                document.addEventListener('visibilitychange', function() {
                    if (document.hidden) {
                        window.parent.postMessage({type: 'violation', msg: 'Tab Switch Detected!'}, '*');
                    }
                });
                window.onblur = function() {
                    window.parent.postMessage({type: 'violation', msg: 'Window Focus Lost!'}, '*');
                };
                </script>
            """, height=0)

            # --- LAYOUT ---
            col_q, col_cam = st.columns([3, 1])
            
            with col_cam:
                st.markdown("### 📹 Proctoring Active")
                st.info("Keep your face in the frame. Do not look away.")
                webrtc_streamer(key="proctor", video_processor_factory=ProctoringProcessor)
                
            with col_q:
                # Load Questions
                job_id = user['Role'].replace(" ", "_") # Fallback ID logic if job_id missing in older data
                # Try to fuzzy match job_id from company if possible, or just use what we have
                # ideally we should have saved job_id in apps_df but for now let's try to fetch
                questions = get_candidate_questions(f"{user['Company']}_{user['Role']}".replace(" ", "_"))
                
                if not questions:
                    st.warning("No questions generated for this role yet. Please contact Admin.")
                else:
                    with st.form("exam_form"):
                        answers = {}
                        for i, q in enumerate(questions):
                            st.markdown(f"**Q{i+1}. {q['q']}**")
                            # Encode options carefully
                            opts = q['options']
                            answers[i] = st.radio(f"Select Answer {i}", opts, key=f"q{i}", label_visibility="collapsed")
                            st.markdown("---")
                            
                        if st.form_submit_button("Submit Test"):
                            score = 0
                            for i, q in enumerate(questions):
                                if answers[i] == q['answer']:
                                    score += 1
                            
                            final_score = (score / len(questions)) * 100
                            
                            # Save Results
                            apps_df.loc[apps_df['Email'] == user['Email'], 'TestStatus'] = 'Completed'
                            apps_df.loc[apps_df['Email'] == user['Email'], 'TestScore'] = final_score
                            save_apps(apps_df)
                            
                            st.success(f"Test Submitted! Your Score: {final_score:.1f}%")
                            st.balloons()
                            time.sleep(3)
                            
                            st.session_state.test_session = None
                            st.rerun()

            if st.button("Logout / Quit"):
                st.session_state.test_session = None
                st.rerun()

    # ---------------- CANDIDATE VIEW ----------------
    # ---------------- CANDIDATE VIEW (MODERN JOB BOARD) ----------------
    elif mode == "Job Seekers":
        # Global CSS for this view
        st.markdown("""
        <style>
            /* HIGH END TYPOGRAPHY & LAYOUT */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
            
            h1, h2, h3, h4, h5, h6, p, div {
                font-family: 'Inter', sans-serif;
            }
            
            /* PREMIUM SEARCH BAR */
            .search-container {
                max-width: 700px;
                margin: 0 auto;
                background: white;
                padding: 10px;
                border-radius: 50px;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
                border: 2px solid transparent;
                transition: all 0.3s ease;
            }
            .search-container:focus-within {
                border-color: #FF9F1C;
                box-shadow: 0 20px 25px -5px rgba(255, 159, 28, 0.15), 0 8px 10px -6px rgba(255, 159, 28, 0.1);
                transform: translateY(-2px);
            }
            
            /* BRANDING HEADER */
            .hero-header {
                text-align: center; 
                margin-top: 2rem; 
                margin-bottom: 3rem;
            }
            .brand-title {
                font-size: 3.5rem; 
                font-weight: 800; 
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0;
                letter-spacing: -1px;
            }
            .brand-tagline {
                font-size: 1.2rem; 
                color: #FF9F1C; 
                font-weight: 600; 
                margin-top: 5px;
                text-transform: uppercase;
                letter-spacing: 2px;
            }
            
            /* JOB CARDS */
            .job-card-container {
                padding: 18px;
                border-radius: 12px;
                background: white;
                border: 1px solid #f1f5f9;
                margin-bottom: 12px;
                transition: all 0.2s;
                position: relative;
                overflow: hidden;
            }
            .job-card-container:hover {
                border-color: #FF9F1C;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                background: #fffcf5; /* Subtle orange tint */
            }
            .job-card-container::before {
                content: '';
                position: absolute;
                left: 0;
                top: 0;
                height: 100%;
                width: 4px;
                background: #FF9F1C;
                opacity: 0;
                transition: opacity 0.2s;
            }
            .job-card-container:hover::before {
                opacity: 1;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # 1. CENTERED BRAND HEADER
        st.markdown("""
            <div class="hero-header">
                <h1 class="brand-title">AUTO HIRE PRO</h1>
                <div class="brand-tagline">Your Future, Automated</div>
            </div>
        """, unsafe_allow_html=True)
        
        # 2. PREMIUM SEARCH BAR
        # We wrap the Streamlit input in a styled container to give it that "Floating Search" look
        col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
        with col_s2:
            st.markdown('<div class="search-container">', unsafe_allow_html=True)
            query = st.text_input("Search Jobs", placeholder="Search by Role, Company, or Keywords...", label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown("<br><br>", unsafe_allow_html=True)
            
        # Filter Logic
        if not df.empty:
            if query:
                filtered_df = df[df['Role'].str.contains(query, case=False) | df['Company'].str.contains(query, case=False)]
            else:
                filtered_df = df
        else:
            filtered_df = pd.DataFrame()

        # 3. Master-Detail Layout
        col_list, col_detail = st.columns([1.3, 2])
        
        with col_list:
            if not filtered_df.empty:
                st.markdown(f"#### 🎯 {len(filtered_df)} Jobs Found")
                # Use a radio button to act as the "List Selector"
                job_options = filtered_df.apply(lambda x: f"{x['Role']}  @  {x['Company']}", axis=1).tolist()
                
                # Logic to keep selection if possible
                if 'selected_job_index' not in st.session_state:
                    st.session_state.selected_job_index = 0
                
                # Custom Styling for Radio to look like cards handled by CSS above (mostly) but Streamlit radio is tough.
                # We trust the .stRadio class hooks or just plain look.
                # Actually, standard radio looks acceptable if we style the container.
                
                selected_label = st.radio(
                    "Select a Job", 
                    job_options, 
                    index=0, 
                    label_visibility="collapsed"
                )
            else:
                st.info("No jobs posted yet. Check back soon!")
                selected_label = None

        with col_detail:
            if selected_label and not filtered_df.empty:
                # Find the row based on the selected label
                # (Assuming uniqueness of Role+Company combinator for now)
                # A safer way would be using ID map, but this works for demo
                selected_role = selected_label.split("  @  ")[0]
                selected_company = selected_label.split("  @  ")[1]
                
                job_data = filtered_df[(filtered_df['Role'] == selected_role) & (filtered_df['Company'] == selected_company)].iloc[0]
                
                # -- DETAIL VIEW --
                st.markdown(f"""
                    <div class="saas-card">
                        <div style="display:flex; justify-content:space-between; align-items:start;">
                            <div>
                                <h2 style="color:var(--primary); margin:0;">{job_data['Role']}</h2>
                                <h4 style="margin:5px 0 15px 0; color:#475569;">{job_data['Company']}</h4>
                            </div>
                            <span style="background:#fef3c7; color:#d97706; padding:5px 10px; border-radius:20px; font-size:0.8rem; font-weight:bold;">Active Hiring</span>
                        </div>
                        
                        <div style="display:flex; gap:10px; margin-bottom:20px;">
                            <span class="badge">📍 Remote / Hybrid</span>
                            <span class="badge">💼 Full Time</span>
                            <span class="badge">⭐ Competitive Salary</span>
                        </div>
                        
                        <hr style="border:0; border-top:1px solid #e2e8f0;">
                        
                        <div style="height: 300px; overflow-y: auto; background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #f1f5f9;">
                            <p style="white-space: pre-wrap; color: #334155; font-family: sans-serif;">{job_data['JD']}</p>
                        </div>
                        <br>
                """, unsafe_allow_html=True)
                
                # Check for Download
                path = job_data.get("JD_File_Path")
                if isinstance(path, str) and os.path.exists(path):
                    with open(path, "rb") as f:
                        st.download_button("📄 Download Official JD", f, file_name=os.path.basename(path))
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # -- APPLICATION FORM --
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="saas-card" style="border-top: 5px solid var(--primary);">', unsafe_allow_html=True)
                st.subheader(f"🚀 Apply to {job_data['Company']}")
                
                with st.form("apply_form"):
                    col_a1, col_a2 = st.columns(2)
                    with col_a1:
                        email = st.text_input("Your Email Address")
                    with col_a2:
                        resume = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
                    
                    st.caption("By applying, you agree to our AI processing your resume.")
                    
                    if st.form_submit_button("Send Application", use_container_width=True):
                        if not email or not resume:
                            st.error("Please provide both Email and Resume.")
                        else:
                            with st.spinner("Analyzing Resume & Sending..."):
                                # LOGIC: SAVE RESUME
                                if not os.path.exists(RESUMES_DIR): os.makedirs(RESUMES_DIR)
                                r_path = os.path.join(RESUMES_DIR, f"{job_data['Company']}_{email}_{resume.name}")
                                with open(r_path, "wb") as f: f.write(resume.getbuffer())
                                
                                # LOGIC: SCORE
                                text = extract_text_from_pdf(resume) if resume.name.endswith(".pdf") else extract_text_from_docx(resume)
                                score = calculate_score(text, job_data["JD"])
                                
                                # LOGIC: SAVE APP
                                new_app = {
                                    "Company": job_data['Company'], 
                                    "Role": job_data["Role"], 
                                    "Email": email, 
                                    "Score": score, 
                                    "Resume_Path": r_path, 
                                    "Timestamp": datetime.datetime.now()
                                }
                                apps_df = pd.concat([apps_df, pd.DataFrame([new_app])], ignore_index=True)
                                save_apps(apps_df)
                                
                                # LOGIC: EMAIL
                                thresh = int(job_data.get("ResumeThreshold", 60))
                                email_type = "success" if score >= thresh else "rejection"
                                send_email(email, score, job_data['Company'], job_data["Role"], email_type)
                                
                                if score >= thresh:
                                    st.success(f"🎉 Application Sent! Resume Match: {score}/100")
                                    st.balloons()
                                else:
                                    st.info(f"Application Sent. Resume Match: {score}/100")
                                    
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.image("https://illustrations.popsy.co/amber/working-vacation.svg", width=300)
                st.info("👈 Select a job from the list to see details and apply.")
    
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # "How it Works" Section (MOVED TO BOTTOM)
        st.markdown("<h3 style='text-align:center;'>How AutoHire Pro Works</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#64748B; margin-bottom:3rem;'>Simplicity meets Intelligence.</p>", unsafe_allow_html=True)
        
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("""
                <div class="saas-card">
                    <div class="step-num">1</div>
                    <h4>Upload Profile</h4>
                    <p style="font-size: 0.9rem; color: #64748B;">Drag & drop your resume (PDF/DOCX). Our system parses it instantly.</p>
                </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown("""
                <div class="saas-card">
                    <div class="step-num">2</div>
                    <h4>AI Analysis</h4>
                    <p style="font-size: 0.9rem; color: #64748B;">Advanced AI compares your skills against the Job Description in real-text.</p>
                </div>
            """, unsafe_allow_html=True)
        with s3:
            st.markdown("""
                <div class="saas-card">
                    <div class="step-num">3</div>
                    <h4>Instant Feedback</h4>
                    <p style="font-size: 0.9rem; color: #64748B;">Get a match score and next steps delivered straight to your inbox.</p>
                </div>
            """, unsafe_allow_html=True)

    # ---------------- ADMIN DASHBOARD ----------------
    elif mode == "Admin Dashboard":
        if 'auth' not in st.session_state: st.session_state.auth = False

        if not st.session_state.auth:
            c1, c2, c3 = st.columns([1,1,1])
            with c2:
                # Custom styled Login Card
                st.markdown('<div class="saas-card">', unsafe_allow_html=True)
                with st.form("login"):
                    st.subheader("Admin Login")
                    u = st.text_input("Username")
                    p = st.text_input("Password", type="password")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.form_submit_button("Login"):
                        if u == "admin" and p == "admin123":
                            st.session_state.auth = True
                            st.rerun()
                        else:
                            st.error("Bad credentials")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # KPIS
            st.title("Admin Dashboard")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Total Candidates", len(apps_df))
            with k2:
                avg = int(apps_df["Score"].mean()) if not apps_df.empty else 0
                st.metric("Avg Quality Score", f"{avg}%")
            with k3:
                active_jobs = len(df)
                st.metric("Active Roles", active_jobs)
            with k4:
                if st.button("Logout"):
                    st.session_state.auth = False
                    st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            
            tab_jobs, tab_apps = st.tabs(["Manage Jobs", "View Applications"])
            
            with tab_jobs:
                with st.expander("➕ Create New Job Opening", expanded=False):
                    with st.form("new_job"):
                        jc1, jc2 = st.columns(2)
                        with jc1:
                            co_name = st.text_input("Company Name")
                            role_name = st.text_input("Role Title")
                        with jc2:
                            jd_file = st.file_uploader("Upload JD Spec")
                        
                        sliders = st.columns(2)
                        with sliders[0]: r_th = st.slider("Resume Pass Score", 0, 100, 60)
                        with sliders[1]: a_th = st.slider("Aptitude Pass Score", 0, 100, 25)
                        
                        if st.form_submit_button("Publish Job"):
                            if co_name and jd_file:
                                jp = os.path.join(JOBS_DIR, jd_file.name)
                                with open(jp, "wb") as f: f.write(jd_file.getbuffer())
                                jtxt = extract_text_from_pdf(jd_file) if jd_file.name.endswith(".pdf") else extract_text_from_docx(jd_file)
                                
                                # Generate Question Bank
                                job_id = f"{co_name}_{role_name}".replace(" ", "_")
                                with st.spinner("🤖 AI is reading JD & Generating 100-Question Exam Bank... (This takes ~15s)"):
                                    cnt = generate_question_bank(jtxt, job_id)
                                
                                new = {"Company": co_name, "Role": role_name, "JD": jtxt, "JD_File_Path": jp, "ResumeThreshold": r_th, "AptitudeThreshold": a_th, "Job_ID": job_id}
                                df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
                                save_data(df)
                                
                                st.balloons()
                                st.snow()
                                st.markdown(f"""
                                    <style>
                                        @keyframes slideIn {{
                                            from {{ opacity: 0; transform: translateY(10px); }}
                                            to {{ opacity: 1; transform: translateY(0); }}
                                        }}
                                        .success-card {{
                                            animation: slideIn 0.5s ease-out;
                                            background-color: #F0FDF4;
                                            border: 1px solid #16A34A;
                                            border-left: 5px solid #16A34A;
                                            border-radius: 8px;
                                            padding: 1.5rem;
                                            display: flex;
                                            align-items: center;
                                            justify-content: center;
                                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                                            margin-top: 20px;
                                        }}
                                        .success-icon {{
                                            font-size: 2rem;
                                            margin-right: 1rem;
                                        }}
                                    </style>
                                    <div class="success-card">
                                        <div class="success-icon">✅</div>
                                        <div>
                                            <h3 style="color: #15803d; margin:0; font-size: 1.25rem;">Job Published Successfully</h3>
                                            <p style="color: #166534; font-size: 1rem; margin-top: 5px; margin-bottom: 0;">100-Question Exam Bank Generated.</p>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                                # st.success("Job Published")
                                time.sleep(3)
                                st.rerun()

                st.markdown("### Active Listings")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    
                    del_co = st.selectbox("Delete Listing", df["Company"].unique(), index=None)
                    if del_co:
                        if st.button("Confirm Delete"):
                            df = df[df["Company"] != del_co]
                            save_data(df)
                            st.success("Deleted")
                            st.rerun()

            with tab_apps:
                st.subheader("Incoming Applications")
                if not apps_df.empty:
                    # Interactive List
                    for i, row in apps_df.sort_values(by="Score", ascending=False).iterrows():
                        with st.container():
                            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                            with c1:
                                st.markdown(f"**{row['Name']}**")
                                st.caption(f"{row['Role']} @ {row['Company']}")
                            with c2:
                                color = "green" if row['Score'] >= 70 else "orange" if row['Score'] >= 50 else "red"
                                st.markdown(f"<span style='color:{color}; font-weight:bold; font-size:1.1em;'>{row['Score']}/100</span>", unsafe_allow_html=True)
                            with c3:
                                st.markdown(f"Status: **{row['Status']}**")
                            with c4:
                                if row['Status'] != "Shortlisted":
                                    if st.button(f"Invite & Shortlist", key=f"sl_{i}"):
                                        token = send_email(row['Email'], row['Score'], row['Company'], row['Role'], "success")
                                        if token:
                                            # Update DB
                                            # We need to find the index in the original dataframe, not the sorted one
                                            idx = apps_df[apps_df['Email'] == row['Email']].index[0]
                                            apps_df.at[idx, 'Status'] = 'Shortlisted'
                                            apps_df.at[idx, 'TestPassword'] = token
                                            apps_df.at[idx, 'TokenTime'] = pd.Timestamp.now().isoformat()
                                            save_apps(apps_df)
                                            st.success(f"Invited {row['Name']}")
                                            time.sleep(1)
                                            st.rerun()
                                else:
                                    st.success("Invited ✅")
                            st.divider()
                else:
                    st.info("No applications received yet.")

if __name__ == "__main__":
    main()
