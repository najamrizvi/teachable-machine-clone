import streamlit as st
import requests
from PIL import Image
import io

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Teachable Machine Clone",
    layout="wide"
)

API_URL = "http://backend:8000"

# ==============================
# SIDEBAR NAVIGATION
# ==============================
st.sidebar.title("🧠 ML Dashboard")
page = st.sidebar.radio("Navigate", ["Home", "Upload Data", "Train Model", "Predict"])

# ==============================
# HOME PAGE
# ==============================
if page == "Home":

    st.title("📊 Teachable Machine Clone")

    st.markdown(
        """
        Welcome to your **End-to-End ML App** 🚀

        Features:
        - 📂 Upload Image Dataset
        - 🧠 Train ML Model (MobileNet + Logistic Regression)
        - 🔮 Make Predictions
        """
    )

    st.success("System is running successfully")

# ==============================
# UPLOAD PAGE
# ==============================
elif page == "Upload Data":

    st.title("📂 Upload Dataset")

    class_name = st.text_input("Enter Class Name (e.g. cats, dogs)")

    images = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("Upload"):

        if not class_name or not images:
            st.error("Please provide class name and images")
        else:
            files = [
                ("images", (img.name, img.getvalue(), img.type))
                for img in images
            ]

            response = requests.post(
                f"{API_URL}/upload-sample",
                data={"class_name": class_name},
                files=files
            )

            try:
                data = response.json()
                st.success("Upload successful!")
                st.json(data)
            except:
                st.error("Backend error")
                st.text(response.text)

# ==============================
# TRAIN PAGE
# ==============================
elif page == "Train Model":

    st.title("🧠 Train Model")

    st.info("Make sure you have uploaded at least 2 classes before training.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model Type", "MobileNet + Logistic Regression")

    with col2:
        st.metric("Status", "Ready")

    if st.button("🚀 Start Training", use_container_width=True):

        with st.spinner("Training model... please wait ⏳"):

            response = requests.post(f"{API_URL}/train")

        try:
            data = response.json()
        except:
            st.error("Backend did not return JSON")
            st.text(response.text)
            st.stop()

        if "error" in data:
            st.error(data["error"])
        else:
            st.success("🎉 Training Completed!")

            st.markdown("### 📊 Training Summary")

            st.write(f"**Samples:** {data.get('samples', 0)}")
            st.write(f"**Classes:** {data.get('classes', [])}")

            st.balloons()

# ==============================
# PREDICT PAGE
# ==============================
elif page == "Predict":

    st.title("🔮 Make Prediction")

    image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if image is not None:

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):

            files = {"image": (image.name, image.getvalue(), image.type)}

            response = requests.post(f"{API_URL}/predict", files=files)

            try:
                data = response.json()
            except:
                st.error("Backend did not return JSON")
                st.text(response.text)
                st.stop()

            if "error" in data:
                st.error(data["error"])
            else:
                st.success("Prediction Complete 🎯")

                st.markdown(f"### 🏷️ Class: `{data['class']}`")
                st.markdown(f"### 📊 Confidence: `{data['confidence']}%`")