from flask import Flask
import ghhops_server as hs
import rhino3dm
import keras as K
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
hops = hs.Hops(app)

# Load the saved scaler
#scaler = joblib.load('X:/My Drive/CEA/PhD/Code/ACADIA24/scaler.save')
scaler_features = joblib.load('/model/scaler_features.pkl')
scaler_targets = joblib.load('/model/scaler_targets.pkl')
scaler_yeo = joblib.load('/model/scaler_yeo.pkl')
@hops.component(
    "/concrete_reuse",
    name="Predict Reuse",
    description="Predict reuse objectives from a pre-trained surrogate model",
    inputs=[
        hs.HopsString("Model Path", "MP", "Input location of the pre-trained surrogate model", optional=True),
        hs.HopsNumber("Inventory Types", "IT", "Categories of inventory dimensions: 0 = L, 1 = L + M, 2 = L + M + S, 3 = M + S"),
        hs.HopsNumber("Wall Length", "WL", "Wall length in cm from 400, 600 to 800"),
        hs.HopsNumber("Wall Height", "WH", "Wall height in cm from 230, 260 to 290"),
        hs.HopsNumber("Door Location", "DL", "Parametrized door location from 0.00, 0.25, 0.50 to 0.75"),
        hs.HopsNumber("Curve Frequency", "CF", "Parametrized curve frequency of the wall outline from 1,2,4 to 6"),
        hs.HopsNumber("Curve Amplitude", "CA", "Parametrized curve amplitude of the wall outline from 2,4 to 6")
    ],
    outputs=[
        hs.HopsNumber("Area of Leftover Inventory", "LI", "The total area of leftover inventory in cm2"),
        hs.HopsNumber("Displacement Average", "DA", "The average displacement in cm"),
        hs.HopsNumber("Uncovered Wall Area", "UWA", "The total area of wall uncovered in cm2"),
        hs.HopsNumber("Average Mapping Tolerance", "AMT", "The average tolerance of mapping")
    ]
)

def concrete_reuse(mPath, Inventory, wLength, wHeight, door, cFreq, cAmpl):
    # Pre-processing data
    x = np.array([Inventory, wLength, wHeight, door, cFreq, cAmpl])

    x_scaled = scaler_features.transform([x])  # Scale data using the preloaded scaler
    print(f"[INFO]....scaled inputs: {x_scaled}")

    # Load model and predict
    if not mPath:
        mPath = "/model/model.keras"
    print("[INFO]....loading default surrogate model")
    model = K.models.load_model(mPath)
    scaled_pred = model.predict(x_scaled)

    # Inverse transform predictions
    nn_pred = scaler_targets.inverse_transform(scaled_pred)
    predictions = scaler_yeo.inverse_transform(nn_pred)
    print(f"[INFO]....predictions: {predictions}")

    # Convert predictions to tuple to pass as separate outputs
    return tuple(predictions.flatten().tolist())

if __name__ == "__main__":
    app.run(debug=True)
