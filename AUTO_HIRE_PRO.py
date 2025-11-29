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
        # Ensure columns exist for backward compatibility
        if "Role" not in df.columns:
            df["Role"] = "Open Role"
        if "JD_File_Path" not in df.columns:
            df["JD_File_Path"] = None
        return df
    else:
        # Create initial dataframe if file doesn't exist
        df = pd.DataFrame(columns=["Company", "Role", "JD", "JD_File_Path", "ResumeThreshold", "AptitudeThreshold"])
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
                        
                        # Download JD Button Logic
                        jd_file_path = company_data.get("JD_File_Path")
                        if jd_file_path and os.path.exists(jd_file_path):
                            with open(jd_file_path, "rb") as f:
                                st.download_button(
                                    label="📥 Download Job Description (PDF/DOCX)",
                                    data=f,
                                    file_name=os.path.basename(jd_file_path),
                                    mime="application/octet-stream"
                                )
                        else:
                            st.warning("⚠️ Job Description file not available for download.")
                        
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
            
            # Action Selection
            action = st.radio("Choose Action", ["Add New Company", "Edit Existing Company", "Delete Company"], horizontal=True)
            st.markdown("---")

            # ---------------- ADD NEW COMPANY ----------------
            if action == "Add New Company":
                st.subheader("➕ Add New Company")
                with st.form("add_company"):
                    col1, col2 = st.columns(2)
                    with col1:
                        company = st.text_input("Company Name")
                        role = st.text_input("Job Role", placeholder="e.g. Software Engineer")
                        resume_threshold = st.slider("Resume Score Threshold", 0, 100, 60)
                    with col2:
                        uploaded_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
                        aptitude_threshold = st.slider("Aptitude Score Threshold", 0, 40, 25)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    submit = st.form_submit_button("💾 Save New Company")

                if submit:
                    if not company:
                        st.error("⚠️ Company Name is required!")
                    elif company in df["Company"].values:
                        st.error("⚠️ Company already exists! Use 'Edit Existing Company' to update.")
                    elif not uploaded_file:
                        st.error("⚠️ Please upload a Job Description file.")
                    else:
                        # Handle File Upload
                        if not os.path.exists("job_descriptions"):
                            os.makedirs("job_descriptions")
                        
                        jd_file_path = os.path.join("job_descriptions", uploaded_file.name)
                        with open(jd_file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extract text (silent)
                        jd_text = ""
                        if uploaded_file.name.endswith(".pdf"):
                            jd_text = extract_text_from_pdf(uploaded_file)
                        elif uploaded_file.name.endswith(".docx"):
                            jd_text = extract_text_from_docx(uploaded_file)

                        new_data = {
                            "Company": company,
                            "Role": role if role else "Open Role",
                            "JD": jd_text,
                            "JD_File_Path": jd_file_path,
                            "ResumeThreshold": resume_threshold,
                            "AptitudeThreshold": aptitude_threshold
                        }
                        
                        new_row = pd.DataFrame([new_data])
                        df = pd.concat([df, new_row], ignore_index=True)
                        save_data(df)
                        st.success(f"✅ Added new company: {company}")
                        st.rerun()

            # ---------------- EDIT EXISTING COMPANY ----------------
            elif action == "Edit Existing Company":
                st.subheader("✏️ Edit Company Details")
                if df.empty:
                    st.info("No companies to edit.")
                else:
                    company_to_edit = st.selectbox("Select Company to Edit", df["Company"].unique())
                    
                    # Get current data
                    current_data = df[df["Company"] == company_to_edit].iloc[0]
                    
                    with st.form("edit_company"):
                        col1, col2 = st.columns(2)
                        with col1:
                            # Company name is read-only or just displayed
                            st.text_input("Company Name", value=company_to_edit, disabled=True)
                            new_role = st.text_input("Job Role", value=current_data["Role"])
                            new_resume_thresh = st.slider("Resume Score Threshold", 0, 100, int(current_data["ResumeThreshold"]))
                        with col2:
                            st.info(f"Current JD File: {os.path.basename(current_data['JD_File_Path']) if current_data['JD_File_Path'] else 'None'}")
                            new_uploaded_file = st.file_uploader("Upload New JD (Optional - Overwrites old)", type=["pdf", "docx"])
                            new_apt_thresh = st.slider("Aptitude Score Threshold", 0, 40, int(current_data["AptitudeThreshold"]))
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        submit_edit = st.form_submit_button("💾 Update Company")

                    if submit_edit:
                        # Handle File Update
                        jd_text = current_data["JD"]
                        jd_file_path = current_data["JD_File_Path"]
                        
                        if new_uploaded_file:
                            if not os.path.exists("job_descriptions"):
                                os.makedirs("job_descriptions")
                            
                            jd_file_path = os.path.join("job_descriptions", new_uploaded_file.name)
                            with open(jd_file_path, "wb") as f:
                                f.write(new_uploaded_file.getbuffer())
                            
                            if new_uploaded_file.name.endswith(".pdf"):
                                jd_text = extract_text_from_pdf(new_uploaded_file)
                            elif new_uploaded_file.name.endswith(".docx"):
                                jd_text = extract_text_from_docx(new_uploaded_file)
                        
                        # Update DataFrame
                        df.loc[df["Company"] == company_to_edit, ["Role", "JD", "JD_File_Path", "ResumeThreshold", "AptitudeThreshold"]] = [
                            new_role, jd_text, jd_file_path, new_resume_thresh, new_apt_thresh
                        ]
                        save_data(df)
                        st.success(f"✅ Updated details for {company_to_edit}")
                        st.rerun()

            # ---------------- DELETE COMPANY ----------------
            elif action == "Delete Company":
                st.subheader("🗑️ Delete Company")
                if df.empty:
                    st.info("No companies to delete.")
                else:
                    company_to_delete = st.selectbox("Select Company to Delete", df["Company"].unique())
                    st.warning(f"⚠️ Are you sure you want to delete **{company_to_delete}**? This action cannot be undone.")
                    
                    if st.button("❌ Yes, Delete Company"):
                        # Remove from DataFrame
                        df = df[df["Company"] != company_to_delete]
                        save_data(df)
                        st.success(f"✅ Deleted company: {company_to_delete}")
                        st.rerun()

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
