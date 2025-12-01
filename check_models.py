import google.generativeai as genai
import os

API_KEY = "AIzaSyAvmsKTpLumaLDRCrFfGip6m077hi3cXdg"
genai.configure(api_key=API_KEY)

with open("models_output.txt", "w") as f:
    f.write("--- START MODEL LIST ---\n")
    try:
        for m in genai.list_models():
            f.write(f"Model: {m.name}\n")
            f.write(f"Methods: {m.supported_generation_methods}\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
    f.write("--- END MODEL LIST ---\n")
