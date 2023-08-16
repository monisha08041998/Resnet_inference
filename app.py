"""
Image classification web application using Flask and TensorFlow.
This app allows users to upload an image and get predictions using a trained model.
"""

import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flasgger import Swagger
from flasgger.utils import swag_from
from dotenv import load_dotenv
app = Flask(__name__)
swagger = Swagger(app)
load_dotenv()

model_path = os.getenv("MODEL_PATH")
static_path = os.getenv("STATIC_PATH")
print(model_path)
model = tf.keras.models.load_model(model_path)
class_labels = ["Class 0", "Class 1"]

def preprocess_image(image_path):
    """
    Preprocesses an image for ResNet input.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array.
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Endpoint to upload an image and get predictions.
    ---
    responses:
      200:
        description: Upload page with image uploader
      400:
        description: No image provided
      500:
        description: Internal server error
    """
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        image = request.files["image"]
        if image.filename == "":
            return jsonify({"error": "No image selected"}), 400

        image_path = os.path.join(static_path, image.filename)
        image.save(image_path)

        preprocessed_image = preprocess_image(image_path)
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        prediction = {"label": predicted_label, "score": float(predictions[0][predicted_class])}

        return jsonify({"prediction": prediction}), 200

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
