import os
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.applications.mobilenet_v2 import MobileNetV2
from flask import Flask, request, jsonify
from flask import Flask, render_template, request

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

model = MobileNetV2(weights='imagenet')

def model_predict(img):
    img = cv2.resize(img, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x, mode='tf')
    preds = model.predict(x)
    return preds

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    top_n = request.form.get("top_n", "1")
    top_n = int(top_n)

    if "image" not in request.files:
        return jsonify({"error": "No image provided."}), 400

    image = request.files["image"]
    image = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)

    preds = model_predict(image)
    pred_proba = "{:.3f}".format(np.amax(preds))
    pred_class = decode_predictions(preds, top=top_n)

    if top_n == 1:
        result = str(pred_class[0][0][1])
        result = result.replace('_', ' ').capitalize()
    else:
        result = [x[1].replace('_', ' ').capitalize() for x in pred_class[0]]

    pred_class = [list(x) for x in pred_class[0]]
    pred_class = [[x[0], x[1], str(x[2])] for x in pred_class]

    response = {
        "result": result,
        "top_probability": pred_proba,
        "classes": pred_class
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
