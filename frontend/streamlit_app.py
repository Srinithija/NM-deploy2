import streamlit as st
import requests
import pyttsx3
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import io
import threading

# Setup TTS function to avoid run loop error
def speak_text(text):
    def run_speech():
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[1].id)  # Female voice
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=run_speech)
    thread.start()

# Page settings
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title(" Handwritten Digit Recognizer")
st.markdown("Upload or draw a digit, and let the AI predict it!")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Upload option
uploaded_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
image = None

# Get the image either from upload or canvas
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)
elif canvas_result.image_data is not None:
    image_data = canvas_result.image_data
    img = Image.fromarray((255 - image_data[:, :, 0]).astype(np.uint8))
    image = img
    st.image(image, caption="Drawn Image", use_column_width=True)

# Prediction button
if image and st.button(" Predict Digit"):
    with st.spinner("Predicting..."):
        try:
            # Convert image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # Send request to backend
            res = requests.post("https://nmbackend.onrender.com/predict", files={"file": img_bytes})
            result = res.json()

            if "error" in result:
                st.error(result["error"])
                speak_text("No digits found.")
            else:
                predicted = result["predicted"]
                confidences = result["confidences"]

                st.success(" Prediction Complete!")
                st.markdown("### Predicted Digit(s):")
                st.write(" ".join(map(str, predicted)))

                # Speak prediction
                speak_text("The predicted digits are " + ', '.join(map(str, predicted)))

                # Confidence Chart
                st.markdown("###  Confidence Chart")
                fig, ax = plt.subplots()
                ax.bar(range(len(predicted)), confidences, tick_label=[str(d) for d in predicted], color="#4CAF50")
                ax.set_xlabel("Digit")
                ax.set_ylabel("Confidence (%)")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Connection error: {e}")
            speak_text("Could not connect to server")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'></p>", unsafe_allow_html=True)
