import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="WMS Asset Prediction API", version="1.0")

MODEL_PATH = r"c:\Users\PC\Documents\Next educacion\TFM\rf_model.pkl"
DATA_PATH = r"c:\Users\PC\Documents\Next educacion\TFM\wms_dataset_transformed.csv"

model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

@app.get("/kpis")
def get_kpis():
    try:
        df = pd.read_csv(DATA_PATH)
        # Preparar features para el modelo
        features = df.drop(columns=["asset_tag_", "serial_", "description", "location", "condition_encoded", "target_needs_repair"], errors="ignore")
        
        if model is not None:
            probs = model.predict_proba(features)[:, 1] * 100
        else:
            probs = [0] * len(df)
            
        df["failure_prob"] = probs
        
        # Logica del semaforo
        df["status_color"] = "Verde"
        mask_rojo = (df["failure_prob"] > 70) | (df["depreciation_pct"] > 100)
        mask_amarillo = (~mask_rojo) & ((df["failure_prob"] >= 40) | (df["stagnation_time_days"] > 180))
        
        df.loc[mask_rojo, "status_color"] = "Rojo"
        df.loc[mask_amarillo, "status_color"] = "Amarillo"
        
        riesgo_financiero = df[df["status_color"] == "Rojo"]["price"].sum()
        capital_zombi = df[df["stagnation_time_days"] > 180]["price"].sum()
        depreciacion_promedio = df["depreciation_pct"].mean()
        
        conteo_colores = df["status_color"].value_counts().to_dict()
        
        return {
            "riesgo_financiero_total": float(riesgo_financiero),
            "capital_zombi_total": float(capital_zombi),
            "depreciacion_promedio": float(depreciacion_promedio),
            "semaforo_distribucion": conteo_colores
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
def predict_asset(data: dict):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Convierte el dict a dataframe de 1 fila
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1] * 100
    pred_class = int(model.predict(df)[0])
    
    return {
        "probability_of_failure": round(prob, 2),
        "predicted_class": pred_class
    }
