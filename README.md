# ✍️ Handwritten Digit Recognizer

A full-stack web application that recognizes handwritten digits using a Convolutional Neural Network (CNN). The project features a Flask backend for deep learning inference and a Streamlit frontend with an interactive drawing canvas, image upload support, confidence visualization, and voice feedback.

---

## 🚀 Features

- 🎨 **Interactive Drawing Canvas** – Draw handwritten digits directly in the browser.
- 📤 **Image Upload** – Upload handwritten digit images (`.png`, `.jpg`, `.jpeg`) for prediction.
- 🧠 **Real-Time Prediction** – A CNN model trained on the MNIST dataset predicts the handwritten digit.
- 📊 **Confidence Visualization** – Displays the top 3 predictions with confidence scores using a horizontal bar chart.
- 🔊 **Voice Feedback** – Announces the predicted digit using Text-to-Speech (TTS).

---

## 🛠️ Tech Stack

### Backend
- **Framework:** Flask
- **Deep Learning:** TensorFlow / Keras (CNN)
- **Image Processing:** OpenCV

### Frontend
- **Framework:** Streamlit
- **Drawing Canvas:** Streamlit Drawable Canvas
- **Visualization:** Matplotlib
- **Text-to-Speech:** Pyttsx3
- **API Communication:** Requests

---

## 📂 Project Structure

```text
NM-deploy2/
│
├── backend/
│   ├── app.py                      # Flask API
│   ├── digit_recognition_model.h5  # Trained CNN model
│   └── requirements.txt            # Backend dependencies
│
├── frontend/
│   ├── streamlit_app.py            # Streamlit application
│   └── requirements.txt            # Frontend dependencies
│
└── README.md                       # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd NM-deploy2
```

---

### 2. Run the Backend

Create a virtual environment, install the required dependencies, and start the Flask server.

```bash
cd backend

python -m venv venv

# Activate the virtual environment

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

python app.py
```

The Flask backend will be available at:

```
http://localhost:5000
```

---

### 3. Run the Frontend

Open a new terminal and execute the following commands:

```bash
cd frontend

python -m venv venv

# Activate the virtual environment

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt

streamlit run streamlit_app.py
```

The Streamlit application will open at:

```
http://localhost:8501
```

---

## 📷 Supported Input

- Draw a handwritten digit using the interactive canvas.
- Upload an image in one of the following formats:
  - `.png`
  - `.jpg`
  - `.jpeg`

---

## 🧠 Model

- **Model:** Convolutional Neural Network (CNN)
- **Dataset:** MNIST Handwritten Digits Dataset
- **Framework:** TensorFlow / Keras

---

## 📊 Output

The application displays:

- ✅ Predicted digit
- 📈 Top 3 prediction confidence scores
- 🔊 Voice announcement of the predicted digit

---

## 👩‍💻 Author

**Srinithija Sivakumar**
