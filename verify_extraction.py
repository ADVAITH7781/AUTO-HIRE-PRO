
from AUTO_HIRE_PRO import extract_text_from_docx
import os

def verify_extraction():
    file_path = "test_jd.docx"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Simulate a file-like object since python-docx Document() accepts file paths or file-like objects
    # But extract_text_from_docx expects a file-like object or path.
    # In Streamlit, uploaded_file is a BytesIO-like object.
    # Let's pass the path directly as python-docx handles it.
    
    try:
        text = extract_text_from_docx(file_path)
        print(f"--- Extracted Text from {file_path} ---")
        print(text)
        print("---------------------------------------")
        
        expected_text = "This is a test Job Description from a DOCX file."
        if expected_text in text:
            print("SUCCESS: Extracted text matches expected content.")
        else:
            print(f"FAILURE: Expected '{expected_text}' but got '{text}'")
            
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    verify_extraction()
