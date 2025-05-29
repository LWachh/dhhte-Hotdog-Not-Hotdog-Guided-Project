# ----------------- Write your code below this line. -------------------- #

import os
from io import BytesIO
from flask import Flask, render_template, request
from config import config
from hotdogclassifier import HotDogClassifier

app = Flask(__name__)
model = HotDogClassifier()
model.load_model(config["model_weight"])

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", flag=False, project_description=config["project_description"], project_name=config["project_name"])

@app.route("/", methods=["POST"])
def classify():
    uploaded_file = request.files["files"]
    data = BytesIO(uploaded_file.read())
    if uploaded_file.filename != "":
        img, predicted = model.predict(data)
    else:
        predicted, img = "", ""
    return render_template("index.html", predicted=predicted, img=img, flag=True, project_description=config["project_description"], project_name=config["project_name"])

config = {
    "model_weight": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hotdog-not-hotdog/data"
                    "/model_weight.pt",
    "project_name": "Hotdog or Not Hotdog Detector",
    "project_description": "Have you ever wondered whether or not an image had a hotdog in it? Wonder no further because the Hotdog Not Hotdog Detector can do just that!"
}

# ----------------- You do NOT need to understand what the code below does. -------------------- #

if __name__ == '__main__':
    PORT = os.environ.get('PORT') or 8080
    DEBUG = os.environ.get('DEBUG') != 'TRUE'
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
