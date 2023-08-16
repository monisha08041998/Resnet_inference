ResNet Image Classifier Inference Flask App
This folder contains a Flask web application for image classification using a trained ResNet model.

Installation

Install dependencies: pip install -r requirements.txt
Set up environment variables in .env (See configuration below).
Usage
Run the app: python app.py

Access the app in your browser at http://localhost:5000/

Configuration
Configure the app by setting the following in the .env file:

STATIC_PATH: Path to the static folder for uploaded images.
CSV_PATH: Path to the CSV file with image labels.

Endpoints
/: Upload images for predictions.


Dependencies
Flask
TensorFlow
NumPy
Pandas
python-dotenv