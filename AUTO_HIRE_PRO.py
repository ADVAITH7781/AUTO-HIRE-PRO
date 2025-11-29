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
        df = pd.read_csv(CSV_FILE)
        # Ensure 'Role' column exists for backward compatibility
        if "Role" not in df.columns:
            df["Role"] = "Open Role"
        return df
    else:
        # Create initial dataframe if file doesn't exist
        df = pd.DataFrame(columns=["Company", "Role", "JD", "ResumeThreshold", "AptitudeThreshold"])
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

    # Custom CSS for Professional Theme
    st.markdown("""
        <style>
        /* Main Background */
        .stApp {
            background-color: #f8f9fa;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #2c3e50;
        }
        [data-testid="stSidebar"] * {
            color: #ecf0f1 !important;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #2c3e50;
            font-family: 'Helvetica Neue', sans-serif;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #2980b9;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #3498db;
            color: white;
        }
        
        /* Cards/Containers */
        .css-1r6slb0 {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Success Messages */
        .stSuccess {
            background-color: #d4edda;
            color: #155724;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Go to", ["Candidate View", "Admin Panel"])
    st.sidebar.markdown("---")
    st.sidebar.info("Auto Hire Pro v1.0")

    # Load current data
    df = load_data()

    # ---------------- CANDIDATE VIEW ----------------
    if app_mode == "Candidate View":
        st.title("🚀 Career Opportunities")
        st.markdown("### Find your dream job and apply today!")
        st.markdown("---")

        if df.empty:
            st.info("No job openings available at the moment. Please check back later.")
        else:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Select a Role")
                # Job Selection
                company_list = df["Company"].unique().tolist()
                selected_company = st.selectbox("Choose a Company", company_list)
                
                if selected_company:
                    st.info(f"You are viewing details for **{selected_company}**")

            with col2:
                if selected_company:
                    # Get details for selected company
                    company_data = df[df["Company"] == selected_company].iloc[0]
                    
                    with st.container():
                        st.subheader(f"📄 {company_data['Role']} at {selected_company}")
                        
                        # Download JD Button
                        st.download_button(
                            label="📥 Download Job Description",
                            data=company_data["JD"],
                            file_name=f"{selected_company}_JD.txt",
                            mime="text/plain"
                        )
                        
                        st.markdown("---")
                        st.markdown(company_data["JD"])
                        st.markdown("---")

                        # Application Form
                        st.subheader("📝 Submit Your Application")
                        with st.form("application_form"):
                            c1, c2 = st.columns(2)
                            with c1:
                                candidate_email = st.text_input("Email Address", placeholder="you@example.com")
                            with c2:
                                uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx"])
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            apply_btn = st.form_submit_button("🚀 Submit Application")

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
        # Initialize session state for login
        if 'admin_logged_in' not in st.session_state:
            st.session_state.admin_logged_in = False

        if not st.session_state.admin_logged_in:
            # Login Screen
            st.title("🔐 Admin Login")
            st.markdown("Please sign in to access the dashboard.")
            
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                login_btn = st.form_submit_button("Login")
                
                if login_btn:
                    if username == "admin" and password == "admin123":
                        st.session_state.admin_logged_in = True
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid Username or Password")
        else:
            # Logout Button
            if st.sidebar.button("Logout"):
                st.session_state.admin_logged_in = False
                st.rerun()

            st.title("📊 Admin Dashboard")
            st.markdown("### Manage Job Postings & Recruitments")
            st.markdown("---")

            with st.container():
                st.subheader("➕ Add / Update Company")
                with st.form("add_company"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        company = st.text_input("Company Name")
                        role = st.text_input("Job Role", placeholder="e.g. Software Engineer")
                        resume_threshold = st.slider("Resume Score Threshold", 0, 100, 60)
                    
                    with col2:
                        uploaded_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
                        aptitude_threshold = st.slider("Aptitude Score Threshold", 0, 40, 25)
                    
                    jd_text = ""
                    if uploaded_file is not None:
                        if uploaded_file.name.endswith(".pdf"):
                            jd_text = extract_text_from_pdf(uploaded_file)
                        elif uploaded_file.name.endswith(".docx"):
                            jd_text = extract_text_from_docx(uploaded_file)
                        st.success("✅ Text extracted from file!")

                    jd = st.text_area("Job Description Content", value=jd_text, height=200)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    submit = st.form_submit_button("💾 Save Configuration")

            if submit:
                if not company:
                    st.error("⚠️ Company Name is required!")
                else:
                    new_data = {
                        "Company": company,
                        "Role": role if role else "Open Role",
                        "JD": jd,
                        "ResumeThreshold": resume_threshold,
                        "AptitudeThreshold": aptitude_threshold
                    }
                    
                    if company in df["Company"].values:
                        # Update existing row
                        df.loc[df["Company"] == company, ["Role", "JD", "ResumeThreshold", "AptitudeThreshold"]] = [role if role else "Open Role", jd, resume_threshold, aptitude_threshold]
                        st.success(f"✅ Updated details for {company}")
                    else:
                        # Add new row
                        new_row = pd.DataFrame([new_data])
                        df = pd.concat([df, new_row], ignore_index=True)
                        st.success(f"✅ Added new company: {company}")
                    
                    # Save to CSV
                    save_data(df)

            # Display
            st.markdown("---")
            st.subheader("📌 Active Job Listings")
            # Reload data to show updates
            df = load_data()

            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No companies added yet.")

if __name__ == "__main__":
    main()
