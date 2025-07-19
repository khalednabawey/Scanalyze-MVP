import streamlit as st
import requests

st.set_page_config(page_title="Scanalyze Medical Analysis",
                   page_icon="🩺", layout="centered")
st.title("🩺 Scanalyze Medical Analysis")

API_URL = "http://localhost:8088"

st.header("Chest X-ray Tuberculosis Prediction")
chest_file = st.file_uploader(
    "Upload Chest X-ray Image", type=["jpg", "jpeg", "png"], key="chest")

if chest_file is not None:
    if st.button("Analyze Chest X-ray"):
        with st.spinner("Analyzing..."):
            files = {"file": (chest_file.name, chest_file, chest_file.type)}
            try:
                response = requests.post(
                    f"{API_URL}/chest-predict", files=files)
                result = response.json()
                if result.get("success"):
                    st.success(f"Prediction: {result['prediction']}")
                    st.info(f"Report: {result['report']}")
                else:
                    st.error(result.get("error", "Unknown error"))
            except Exception as e:
                st.error(f"Request failed: {e}")

st.header("Kidney CT Classification")
kidney_file = st.file_uploader("Upload Kidney CT Image", type=[
                               "jpg", "jpeg", "png"], key="kidney")

if kidney_file is not None:
    if st.button("Analyze Kidney CT"):
        with st.spinner("Analyzing..."):
            files = {"file": (kidney_file.name, kidney_file, kidney_file.type)}
            try:
                response = requests.post(
                    f"{API_URL}/kidney-predict", files=files)
                result = response.json()
                if result.get("success"):
                    st.success(f"Prediction: {result['prediction']}")
                    st.info(f"Report: {result['report']}")
                else:
                    st.error(result.get("error", "Unknown error"))
            except Exception as e:
                st.error(f"Request failed: {e}")
