import streamlit as st
import os
import uuid
import json
import pandas as pd
from preprocess import preprocess_and_crop_image
from ocr import run_ocr

# ==== Config ====
INPUT_DIR = "input"
PREPROCESSED_DIR = "preprocessed"
OUTPUT_DIR = "output"
WEIGHTS_PATH = "runs/detect/aadhar-fields7/weights/best.pt"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== Custom CSS ====
st.set_page_config(page_title="🪪 Nepali OCR AI", layout="wide")
st.markdown(
    """
    <style>
        .main { background-color: #f8f9fa; }
        .block-container {
            padding: 2rem 4rem 2rem 4rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            margin-top: 10px;
        }
        .stFileUploader>div>div {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 1rem;
            background-color: #ffffff;
        }
        img {
            max-width: 600px;
            width: 100%;
            height: auto;
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 10px;
            box-shadow: 0px 2px 12px rgba(0,0,0,0.1);
        }
        .element-container:has(.dataframe) {
            max-width: 800px;
            margin: 0 auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==== App UI ====
st.title(" Nepali ID OCR AI")
st.caption("Built with  using YOLOv8 + EasyOCR")

st.markdown("---")
st.subheader(" Upload Aadhaar / PAN Image")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    unique_id = str(uuid.uuid4())
    input_path = os.path.join(INPUT_DIR, f"{unique_id}.jpg")
    cropped_path = os.path.join(PREPROCESSED_DIR, f"{unique_id}_cropped.jpg")
    enhanced_path = os.path.join(PREPROCESSED_DIR, f"{unique_id}_enhanced.jpg")
    annotated_path = os.path.join(OUTPUT_DIR, f"{unique_id}_annotated.jpg")
    json_path = os.path.join(OUTPUT_DIR, f"{unique_id}_ocr.json")

    # Save uploaded image
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.markdown(" Uploaded Image")
    st.image(input_path)

    with st.spinner("🔧 Preprocessing Image..."):
        success = preprocess_and_crop_image(input_path, cropped_path, enhanced_path)

    if success:
        st.success("Image Preprocessed")
        st.markdown(" Enhanced Image")
        st.image(enhanced_path)

        with st.spinner(" Running OCR Detection..."):
            run_ocr(enhanced_path, WEIGHTS_PATH, annotated_path, json_path)

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                ocr_data = json.load(f)
            results = ocr_data.get("results", [])

            if results:
                st.markdown("---")
                st.subheader(" Annotated Result")
                st.image(annotated_path)

                st.subheader(" Extracted Information")
                df = pd.DataFrame(results)
                df_display = df[["label", "text"]].rename(
                    columns={"label": " Field", "text": " Extracted Text"}
                )
                st.table(df_display)

                with st.expander("View Raw JSON Output"):
                    st.json(ocr_data)

                st.success(" OCR Complete")
            else:
                st.warning(" No text detected in image.")
        else:
            st.error(" OCR failed.")
    else:
        st.error("Preprocessing failed. Please try again with a clearer image.")
