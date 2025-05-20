from flask import Flask, request, jsonify
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained digit model
model = load_model("digit_recognition_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    try:
        # Convert and resize image
        image = Image.open(file).convert("L")
        image = ImageOps.invert(image)  # Invert to match MNIST style (white digit on black)
        image = image.resize((28, 28))
        image_array = np.array(image).astype("float32") / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        predictions = model.predict(image_array)
        top_indices = predictions[0].argsort()[-3:][::-1]
        top_confidences = predictions[0][top_indices] * 100

        return jsonify({
            "predicted": top_indices.tolist(),
            "confidences": [round(float(c), 2) for c in top_confidences]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
