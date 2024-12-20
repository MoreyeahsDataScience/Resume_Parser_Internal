######      streamlit app for resume parser(spport pdf, doc, image)     ##########

import json
import re
import subprocess
import tempfile
from io import BytesIO
import pytesseract as pt
from PIL import Image
import pandas as pd
import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate


# Set up pytesseract path for image OCR
pt.pytesseract.tesseract_cmd = 'tesseract.exe'

# Model and API key
model_name = "gemini-1.5-flash-001"
api_key = "AIzaSyCwccuIEZTRYt0AyD1EdNOo41soO8oY6CE"

# Define function to extract text from various file types

def extract_text_from_image(contents: bytes) -> str:
    try:
        with Image.open(BytesIO(contents)) as img:
            text = pt.pytesseract.image_to_string(img)
            text = re.sub(r"[#=]", "", text)
            text = re.sub(r"[^\x00-\x7F]+", "", text)
            text = re.sub(r" +", " ", text).strip()  # Remove leading and trailing spaces
            if not text.replace(" ", ""):
                raise ValueError("No words found in the extracted text")
            return text
    except Exception as e:
        raise ValueError(f"Error extracting text from image: {str(e)}")

# Process resume and extract details
def process_resume(resume_text: str) -> dict:
    llm = GoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0, timeout=120)
    prompt = f"""
    <resume text>
    {resume_text}
    </resume text>  
    Extract the following details:
    - Name
    - email id
    - phone number (without country code)

    Provide the extracted details in JSON format. If nothing is extracted, return empty "".

    <example>
    "name": 
    "email": 
    "phone_number": 
    </example>

    Provide the output in JSON format only:
    ```json
    .....
    ```
    """

    prompt = PromptTemplate.from_template(prompt)
    AIanswer = (prompt | llm)
    answer = AIanswer.invoke({"resume_text": resume_text})
    answer = re.sub(r"[\`*]", "", answer)
    answer = re.sub(r"json", "", answer)
    return json.loads(answer)

# Streamlit App
def main():
    st.title("Resume Parser and Extractor")

    # File upload
    uploaded_files = st.file_uploader("Upload Resume Files", accept_multiple_files=True, type=["pdf", "docx", "doc", "jpg", "jpeg", "png"])
    
    if uploaded_files:
        responses = []
        failed_files = []

        # Start processing the files
        for uploaded_file in uploaded_files:
            try:
                contents = uploaded_file.read()

                # Check file extension and extract text accordingly
                file_extension = uploaded_file.name.split(".")[-1].lower()
                
                if file_extension in ["jpg", "jpeg", "png"]:
                    resume_text = extract_text_from_image(contents)
                else:
                    raise ValueError(f"Unsupported file format: {uploaded_file.name}. Only image files are supported.")

                # Process resume and extract details
                if not resume_text:
                    raise ValueError(f"Can't convert file {uploaded_file.name} to text.")
                response_data = process_resume(resume_text)
                responses.append({"filename": uploaded_file.name, "data": response_data})

            except Exception as e:
                failed_files.append({"filename": uploaded_file.name, "error": str(e)})
                continue
        
        # Display results
        if responses:
            # Convert extracted data into a DataFrame
            extracted_data = [resp["data"] for resp in responses if "data" in resp]
            df = pd.DataFrame(extracted_data)

            # Excel file path
            excel_file_name = "extracted_resumes_streamlit.xlsx"
            excel_file_path = os.path.join(os.getcwd(), excel_file_name)

            # Check if the Excel file exists
            if os.path.exists(excel_file_path):
                # If the file exists, read the existing data and append the new data
                existing_df = pd.read_excel(excel_file_path)
                df = pd.concat([existing_df, df], ignore_index=True)

            # Save to the Excel file (either append or new file)
            df.to_excel(excel_file_path, index=False)

            st.success(f"Processing completed. The results are saved in {excel_file_name}.")
            st.write(df)

        if failed_files:
            st.error("Some files failed to process:")
            for failed_file in failed_files:
                st.write(f"File: {failed_file['filename']}, Error: {failed_file['error']}")
        else:
            st.info("All files were processed successfully.")

if __name__ == "__main__":
    main()
