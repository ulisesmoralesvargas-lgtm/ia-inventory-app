import os
import streamlit as st
import pandas as pd
import requests

# Si estás en GCP, pasarás la URL de Cloud Run de la API aquí
API_URL = os.getenv("API_URL", "http://localhost:8000")
DATA_PATH = os.getenv("DATA_PATH", "./data/wms_dataset_transformed.csv")

st.set_page_config(page_title="WMS Asset Dashboard", layout="wide")
st.title("📊 Dashboard Financiero y Predictivo de Activos Fijos")

# El resto de tu código de dashboard se mantiene igual...
# (Solo asegúrate de cambiar la lectura del CSV local a DATA_PATH)
