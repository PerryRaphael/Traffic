import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Define image dimensions and categories
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
UPLOAD_FOLDER = "static/uploads"

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an image was uploaded
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        if file:
            # Save the uploaded file
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

        
            img = cv2.imread(file_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)
            category = np.argmax(prediction)

            return render_template("index.html", image=file.filename, category=category)

    return render_template("index.html", image=None, category=None)

if __name__ == "__main__":
    app.run(debug=True)