import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, Form, Request
import threading

app = Flask(__name__)
fastapi_app = FastAPI()

def get_version_info():
    version_info = {
        "app_name": "Model Serving Demo",
        "version": "0.1"
    }
    return version_info

# Your existing code
with open('Lgbm_model.pkl', 'rb') as file:
    top_classifier = pickle.load(file)
with open('ranforest_model.pkl', 'rb') as f:
    second_classifier=pickle.load(f)
with open('blending_model.pkl', 'rb') as ff:
    ensemble=pickle.load(ff)
with open('xgb_model.pkl', 'rb') as xg:
    xgb = pickle.load(xg)
with open('model_mlp.pkl', 'rb') as ml:
    mlp =pickle.load(ml)

df=pd.read_csv('C:/Users\gazur/Desktop/Polyfins_Intern-2023/dataset/forest-cover-type-prediction/train.csv')
df=df.drop(["Id", "Cover_Type"], axis = 1)
Features=[col for col in df.columns]
sc=StandardScaler()

def data_preprocess(df):
    df = df.drop(['Soil_Type15', 'Soil_Type7'], axis=1)
    Features = [col for col in df.columns]
    df['EHiElv'] = df['Horizontal_Distance_To_Roadways'] * df['Elevation']
    df['EViElv'] = df['Vertical_Distance_To_Hydrology'] * df['Elevation']
    # the compass direction that a terrain faces
    df["Aspect"][df["Aspect"] < 0] += 360
    df["Aspect"][df["Aspect"] > 359] -= 360
    # Hillshade 3D representation of a surface,
    # It's a shade of grey so all the values must lie in the range (0, 255)
    df.loc[df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
    df.loc[df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
    df.loc[df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
    df.loc[df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
    df.loc[df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
    df.loc[df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
    df['Highwater'] = (df.Vertical_Distance_To_Hydrology < 0).astype(int)
    df['EVDtH'] = df.Elevation - df.Vertical_Distance_To_Hydrology
    df['EHDtH'] = df.Elevation - df.Horizontal_Distance_To_Hydrology * 0.2
    df['Euclidean_Distance_to_Hydrolody'] = (df['Horizontal_Distance_To_Hydrology'] ** 2 + df[
        'Vertical_Distance_To_Hydrology'] ** 2) ** 0.5
    df['Manhattan_Distance_to_Hydrolody'] = df['Horizontal_Distance_To_Hydrology'] + df[
        'Vertical_Distance_To_Hydrology']
    df['Hydro_Fire_1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['Hydro_Fire_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Road_1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
    df['Hydro_Road_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])
    df['Hillshade_3pm_is_zero'] = (df.Hillshade_3pm == 0).astype(int)
    df['min'] = df[Features].min(axis=1)
    df['max'] = df[Features].max(axis=1)
    df['mean'] = df[Features].mean(axis=1)
    df['std'] = df[Features].std(axis=1)
    return df



@fastapi_app.post('/fastapi/predict')
async def fastapi_predict_result(data: str = Form(...)):
    values = [int(num) for num in data.split(',')]
    data = dict(zip(Features, values))
    a = pd.DataFrame([data])
    fe = data_preprocess(a)

    # Ensemble predictions (same as before)
    pred_mlp = mlp.predict(sc.fit_transform(fe))
    pred_xgb = xgb.predict(fe)
    pred_light = top_classifier.predict(fe)
    pred_ra = second_classifier.predict(fe)
    meta_fe = np.column_stack((pred_mlp, pred_xgb, pred_light, pred_ra))
    blending = ensemble.predict(meta_fe)  # Predict using the ensemble model

    # Convert the NumPy array to a Python list before returning as JSON
    prediction = pred_light.tolist()
    prediction1 = pred_ra.tolist()
    prediction2 = blending.tolist()

    # Return prediction as JSON response
    response_data = {
        'prediction': prediction[0],
        'prediction1': prediction1[0],
        'prediction2': prediction2[0]
    }
    return response_data

@app.route('/')
def index():
    version_info = get_version_info()
    return render_template("index.html", version_info=version_info)


# Run both Flask and FastAPI in separate threads

def run_flask():
    app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    flask_thread.join()

