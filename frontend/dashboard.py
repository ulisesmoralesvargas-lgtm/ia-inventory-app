import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

import streamlit as st
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000"
DATA_PATH = r"c:\Users\PC\Documents\Next educacion\TFM\wms_dataset_transformed.csv"

# Configuración de diseño
st.set_page_config(page_title="WMS Asset Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Dashboard Financiero y Predictivo de Activos Fijos")
st.markdown("Plataforma interactiva para evaluar el riesgo operativo e impacto financiero del inventario (WMS).")

# --- 1. KPIs FINANCIEROS (Consumiendo la API) ---
st.header("Métricas de Negocio Globales")
try:
    with st.spinner("Conectando al Motor de Inteligencia Artificial (Backend API)..."):
        res = requests.get(f"{API_URL}/kpis")
    
    if res.status_code == 200:
        data = res.json()
        
        # Tarjetas de KPI
        col1, col2, col3 = st.columns(3)
        col1.metric("Valor Total en Riesgo (Activos Críticos ROJO)", f"${data['riesgo_financiero_total']:,.2f}")
        col2.metric("Capital Inmovilizado (Activos Zombis AMARILLO)", f"${data['capital_zombi_total']:,.2f}")
        col3.metric("Depreciación Promedio del Inventario", f"{data['depreciacion_promedio']:.2f}%")
        
        # Resumen del semáforo
        semaforo = data.get("semaforo_distribucion", {})
        st.write("**Distribución del Semáforo en el Inventario Total:**")
        
        c1, c2, c3 = st.columns(3)
        c1.success(f"🟢 Verde (Saludable): {semaforo.get('Verde', 0)} equipos")
        c2.warning(f"🟡 Amarillo (Alerta): {semaforo.get('Amarillo', 0)} equipos")
        c3.error(f"🔴 Rojo (Crítico): {semaforo.get('Rojo', 0)} equipos")
    else:
        st.error("⚠️ Error conectando con la API (FastAPI no está corriendo).")
except Exception as e:
    st.error("⚠️ No se pudo conectar a la API. Asegúrate de que `api.py` esté corriendo en el puerto 8000.")

st.markdown("---")

# --- 2. SIMULADOR DE SEMÁFORO PREDICTIVO ---
st.header("Semáforo de Riesgo por Activo")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()
activos = df["asset_tag_"].dropna().unique().tolist()

selected_tag = st.selectbox("Seleccione el Asset Tag de un equipo para evaluar su riesgo en tiempo real:", activos)

if selected_tag:
    asset_data = df[df["asset_tag_"] == selected_tag].iloc[0]
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.write(f"**Descripción:** {asset_data['description']}")
    with col_info2:
        st.write(f"**Costo Original:** ${asset_data['price']:.2f}")
    
    # Preparar el payload (quitar identificadores y targets)
    features_to_drop = ["asset_tag_", "serial_", "description", "location", "condition_encoded", "target_needs_repair"]
    payload = asset_data.drop(labels=features_to_drop, errors="ignore").fillna(0).to_dict()
    
    st.write("") # Espacio
    
    # Boton para evaluar
    if st.button("🔮 Evaluar Riesgo Operativo", type="primary"):
        with st.spinner("Calculando predicción..."):
            try:
                res = requests.post(f"{API_URL}/predict", json=payload)
                if res.status_code == 200:
                    result = res.json()
                    prob = result["probability_of_failure"]
                    
                    # Variables adicionales para la lógica de negocio
                    dep_pct = asset_data["depreciation_pct"]
                    stagnation = asset_data["stagnation_time_days"]
                    
                    # Lógica de Negocio del Semáforo
                    color = "Verde"
                    hex_color = "#28a745"
                    mensaje = "Saludable. El equipo opera dentro de los márgenes normales de vida útil."
                    icono = "✅"
                    
                    if prob > 70 or dep_pct > 100:
                        color = "Rojo"
                        hex_color = "#dc3545"
                        mensaje = "CRÍTICO. Reemplazo inmediato. Presupuestar su sustitución."
                        icono = "🚨"
                    elif prob >= 40 or stagnation > 180:
                        color = "Amarillo"
                        hex_color = "#ffc107"
                        mensaje = "ALERTA. Activo infrautilizado (Zombi) o con riesgo de fallo a corto plazo. Se sugiere reasignación o revisión."
                        icono = "⚠️"
                    
                    # Renderizar el Semáforo
                    st.markdown(f"""
                    <div style="background-color: {hex_color}; padding: 30px; border-radius: 10px; color: white; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                        <h1 style="color: white; margin-bottom: 0;">{icono} Semáforo: {color.upper()}</h1>
                        <hr style="border-top: 1px solid white;">
                        <div style="display: flex; justify-content: space-around; text-align: center;">
                            <div>
                                <p style="margin-bottom: 0; font-size: 1.2em;">Probabilidad de Fallo (IA)</p>
                                <h2 style="color: white; margin-top: 0;">{prob}%</h2>
                            </div>
                            <div>
                                <p style="margin-bottom: 0; font-size: 1.2em;">Depreciación Consumida</p>
                                <h2 style="color: white; margin-top: 0;">{dep_pct:.1f}%</h2>
                            </div>
                            <div>
                                <p style="margin-bottom: 0; font-size: 1.2em;">Días Estancado</p>
                                <h2 style="color: white; margin-top: 0;">{stagnation} días</h2>
                            </div>
                        </div>
                        <h3 style="background-color: rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; margin-top: 20px;">
                            <strong>Recomendación Estratégica:</strong> {mensaje}
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error(f"Error en la predicción de la API: {res.text}")
            except Exception as e:
                st.error(f"⚠️ Error conectando con la API: {e}")
