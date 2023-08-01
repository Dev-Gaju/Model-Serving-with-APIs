"""
    This is an sample Deep learning Image classification model serving app using FastApi.
"""

import os
import cv2
import time
import uvicorn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Request
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from fastapi.logger import logger
from typing import Optional, List, Tuple, Union
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



# paths and configs classes to keep the variables together and making the code more readable

class AppConfig(BaseSettings):
    app_name: str = "Model Serving Demo"
    version: str = "0.1"


class ImageConfigs(BaseSettings):
    img_shape: Tuple = (224, 224)


# Reasponse models. these are pydantic response data model for validation

class Index_Response(BaseModel):
    response: str = "Model Serving Demo"


class Classifier_Response(BaseModel):
    result: Union[str, List]
    top_probability: str
    classes: List


class Version_Response(BaseModel):
    app_name: str
    version: str


appconfig = AppConfig()
imageconfig = ImageConfigs()

app = FastAPI(title=appconfig.app_name,
              version=appconfig.version,
              description="A demo backend app for serving image classification model with FastApi.")

# CORS implementation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# custom middleware for checking response time
@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> str:
    """ This is a custom middleware function to return the api response time in response header.

    Args:
        request (Request): Request object
        call_next ([type]): FASTAPI call function

    Returns:
        str : response
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


model = MobileNetV2(weights='imagenet')
logger.info("model loaded!!!")


def model_predict(img: np.ndarray):
    """This is the input preprocessing and model prediction function bunched together.

    Args:
        img (np.ndarray): input image converted to np array.

    Returns:
        np.ndarray: model prediction output
    """
    img = cv2.resize(img, imageconfig.img_shape)
    x = np.expand_dims(img, axis=0)
    # Be mindful of the input preprocessing otherwise model will not perform as expected
    x = preprocess_input(x, mode='tf')
    preds = model.predict(x)
    return preds


@app.get("/")
async def index():
    """This is the root endpoint. We can redirect this to docs or the version endpoint as well.

    Returns:
        string: Returns the app name
    """
    return appconfig.app_name


@app.get("/version", response_model=Version_Response)
def version():
    """
    Version endpoint.

    Returns:
        dict: App Information.
    """
    return {
        "app_name": appconfig.app_name,
        "version": appconfig.version
    }


@app.post("/predict", response_model=Classifier_Response)
async def predict(image: UploadFile = File(...), top_n: str = '1'):
    """ Imagenet prediction endpoint. It takes an image and top n prediction number and provides response accordingly.
    If the 'top_n' is 1 the result will be a single string of top class name.
    If it's more than one, then the classes will be returned as a list of strings.

    Args:
        image (UploadFile, optional): Input image file. Defaults to File(...).
        top_n (int, optional): top number of classes. Defaults to 1.

    Returns:
        dict : response object. look at the classifier response model.
    """
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    top_n = int(top_n)
    # Make prediction
    preds = model_predict(img)

    # Process result
    pred_proba = "{:.3f}".format(np.amax(preds))  # Max probability
    pred_class = decode_predictions(preds, top=top_n)  # ImageNet Decode

    # processing result value
    if top_n == 1:
        result = str(pred_class[0][0][1])
        result = result.replace('_', ' ').capitalize()
    else:
        result = [x[1].replace('_', ' ').capitalize() for x in pred_class[0]]

    # processing prediction classes
    pred_class = [list(x) for x in pred_class[0]]
    pred_class = [[x[0], x[1], str(x[2])] for x in pred_class]

    return {
        "result": result,
        "top_probability": pred_proba,
        "classes": pred_class

    }


if __name__ == "__main__":
    # uvicorn is a asynchronus web server to serve the backend application.
    # the first 'app' in the following line denote the file name
    # second one denotes the fastapi app name
    # this convention is applicable for flask dev server as well.
    uvicorn.run("app:app", host="127.0.0.1", port=5000) #debug=False