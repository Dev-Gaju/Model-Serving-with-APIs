import os
import cv2
import time
import uvicorn
import numpy as np
import onnxruntime
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Request
from pydantic_settings import BaseSettings
from pydantic import BaseModel
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


# model = MobileNetV2(weights='imagenet')
# logger.info("model loaded!!!")

onnx_path = "model/effcientv2b2.onnx"
ort_session = onnxruntime.InferenceSession(onnx_path)


def model_predict(img: np.ndarray):
    img = cv2.resize(img, imageconfig.img_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    x = img.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = np.transpose(x, (0, 3, 1, 2))  # ONNX expects NCHW format

    # ONNX Runtime expects input tensor with a specific name
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # Perform inference using ONNX Runtime
    preds = ort_session.run([output_name], {input_name: x})
    return preds[0]


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
async def predict(image: UploadFile = File(...)):
    """ ONNX model prediction endpoint for binary classification (dog or cat).

    Args:
        image (UploadFile): Input image file.

    Returns:
        dict : response object. Look at the classifier response model.
    """
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Make prediction using ONNX
    preds = model_predict(img)

    # Assuming class_names represents ["cat", "dog"]
    class_names = ["cat", "dog"]

    # Determine the predicted class
    predicted_class_index = np.argmax(preds)
    predicted_class_name = class_names[predicted_class_index].capitalize()

    # Get the probability of the predicted class
    pred_proba = "{:.3f}".format(preds[0][predicted_class_index])

    # Prepare response
    result = predicted_class_name
    pred_class = [[class_name, class_name == predicted_class_name, str(preds[0][i])]
                  for i, class_name in enumerate(class_names)]

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
    uvicorn.run("app_onnx:app", host="127.0.0.1", port=5000)
