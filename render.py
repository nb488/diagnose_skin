from flask import Flask, render_template, request, redirect, url_for, jsonify
from joblib import load
from PIL import Image
import os
import tensorflow as tf

# Load the model
model = load("ml_model/model.keras")

def requestResults(image):
    # TODO: preprocess image


    prediction = model.predict(image)
    return prediction  # Placeholder for the actual prediction

app = Flask(__name__)

# Secret key for session (you can use os.urandom(24) to generate a random key)
app.secret_key = os.urandom(24)

# Define route for root URL
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("diagnose.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    img = Image.open(file)

    # Get prediction result
    result = requestResults(img)  # Get the result TODO
    print(result)
    # Redirect to the results page with the result as a query parameter
    return redirect(url_for(".results", result=result))

@app.route('/picversion')
def results(result):
    # Get the result from the URL query parameter
    # result = request.args.get('result')

    # Render the results page with the result variable
    return render_template('picversion.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)

