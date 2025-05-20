import streamlit as st
import requests
import pyttsx3
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import io
import threading

# Function to handle text-to-speech in a separate thread
def speak_text(text):
    def run_speech():
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        voices = engine.getProperty("voices")
        if len(voices) > 1:
            engine.setProperty("voice", voices[1].id)  # Female voice
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_speech).start()

# Function to preprocess image
def preprocess_image(img: Image.Image) -> Image.Image:
    # Resize to 28x28 (MNIST format)
    img = ImageOps.invert(img)  # Invert so background is black, digit is white
    img = img.resize((28, 28))
    return img

# Page setup
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✍️ Handwritten Digit Recognizer")
st.markdown("Draw or upload a digit image and let AI predict what it is!")

# Canvas for drawing digits
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

# Upload image
uploaded_file = st.file_uploader("Or upload a digit image", type=["png", "jpg", "jpeg"])
image = None

# Extract image from uploaded file
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Or use drawn image from canvas
elif canvas_result.image_data is not None:
    image_data = canvas_result.image_data
    if image_data is not None:
        # Convert canvas RGBA to grayscale image
        img = Image.fromarray((255 - image_data[:, :, 0]).astype(np.uint8))  # Use red channel as grayscale
        image = img
        st.image(image, caption="Drawn Image", use_column_width=True)

# Predict button
if image is not None and st.button("Predict Digit"):
    with st.spinner("Predicting..."):
        try:
            # Preprocess image
            processed_image = preprocess_image(image)

            # Save to bytes buffer
            img_bytes = io.BytesIO()
            processed_image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            # Send POST request to backend
            response = requests.post("https://nmbackend.onrender.com/predict", files={"file": img_bytes})
            result = response.json()

            if "error" in result:
                st.error(result["error"])
                speak_text("No digits found.")
            else:
                predicted = result.get("predicted", [])
                confidences = result.get("confidences", [])

                if not predicted:
                    st.warning("No digits could be identified.")
                    speak_text("No digit detected.")
                else:
                    st.success(" Prediction Complete!")
                    st.markdown("###  Predicted Digit(s):")
                    st.write(" ".join(map(str, predicted)))

                    speak_text("The predicted digits are " + ', '.join(map(str, predicted)))

                    # Plot confidence chart
                    st.markdown("###  Confidence Chart")
                    fig, ax = plt.subplots()
                    ax.bar(range(len(predicted)), confidences, tick_label=[str(d) for d in predicted], color="#4CAF50")
                    ax.set_xlabel("Digit")
                    ax.set_ylabel("Confidence (%)")
                    ax.set_ylim(0, 100)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Connection error: {e}")
            speak_text("Could not connect to the server.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2025 Digit Recognizer App</p>", unsafe_allow_html=True)
