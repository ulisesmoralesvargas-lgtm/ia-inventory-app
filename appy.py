import os
import pandas as pd
import joblib
from fastapi import FastAPI

app = FastAPI(title="WMS Asset Prediction API", version="1.0")

# Rutas dinámicas
MODEL_PATH = os.getenv("MODEL_PATH", "./data/rf_model.pkl")
DATA_PATH = os.getenv("DATA_PATH", "./data/wms_dataset_transformed.csv")

model = None
df_global = None

@app.on_event("startup")
def startup_event():
    global model, df_global
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(DATA_PATH):
        df_global = pd.read_csv(DATA_PATH)

@app.get("/kpis")
def get_kpis():
    try:
        if df_global is None:
            return {"error": "Dataset no cargado en el servidor"}
        
        df = df_global.copy()
        features = df.drop(columns=["asset_tag_", "serial_", "description", "location", "condition_encoded", "target_needs_repair"], errors="ignore")
        
        if model is not None:
            probs = model.predict_proba(features)[:, 1] * 100
        else:
            probs = [0] * len(df)
            
        df["failure_prob"] = probs
        
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
    
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1] * 100
    pred_class = int(model.predict(df)[0])
    
    return {
        "probability_of_failure": round(prob, 2),
        "predicted_class": pred_class
    }
