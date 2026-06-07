# -*- coding: utf-8 -*-
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Configurar estilo visual
sns.set_theme(style="whitegrid")

# Rutas
INPUT_FILE = r"c:\Users\PC\Documents\Next educacion\TFM\wms_dataset_transformed.csv"
CM_PLOT_PATH = r"c:\Users\PC\Documents\Next educacion\TFM\confusion_matrix.png"
FI_PLOT_PATH = r"c:\Users\PC\Documents\Next educacion\TFM\feature_importance.png"
KMEANS_PLOT_PATH = r"c:\Users\PC\Documents\Next educacion\TFM\kmeans_elbow_silhouette.png"

def main():
    print("="*80)
    print("  WMS ASSETS - MACHINE LEARNING PIPELINE")
    print("="*80)

    # 1. Cargar Datos
    print("\n[1] Cargando datos limpios...")
    df = pd.read_csv(INPUT_FILE)
    
    # 2. Separar Features y Target, evitar Data Leakage
    cols_to_drop = [
        "asset_tag_", "serial_", "description", "location", # Identificadores y texto libre
        "condition_encoded", # EVITAMOS FUGAS DE DATOS (DATA LEAKAGE)
        "target_needs_repair" # Variable a predecir
    ]
    
    # Asegurarnos de que las columnas existan antes de eliminarlas
    drop_actual = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=drop_actual)
    y = df["target_needs_repair"]
    
    print(f"Dimensiones de entrenamiento: X={X.shape}, y={y.shape}")
    print(f"Distribución del target:\n{y.value_counts(normalize=True)*100}\n")

    # =========================================================================
    # MODELO 1: RANDOM FOREST CLASSIFIER (Supervisado)
    # =========================================================================
    print("="*80)
    print("  [A] RANDOM FOREST CLASSIFIER (Validación Cruzada 5-Fold)")
    print("="*80)
    
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    cv_results = cross_validate(rf, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    
    print("\nResultados Promedio de Cross-Validation (5-Fold):")
    print(f"  - Accuracy General : {cv_results['test_accuracy'].mean():.4f}")
    print(f"  - Precision (Clase 1): {cv_results['test_precision'].mean():.4f}")
    print(f"  - Recall    (Clase 1): {cv_results['test_recall'].mean():.4f}")
    print(f"  - F1-Score  (Clase 1): {cv_results['test_f1'].mean():.4f}")
    
    # Entrenar el modelo final con todo el dataset para gráficos y exportación
    rf.fit(X, y)
    y_pred = rf.predict(X)
    
    # Exportar el modelo
    model_path = r"c:\Users\PC\Documents\Next educacion\TFM\rf_model.pkl"
    joblib.dump(rf, model_path)
    print(f"\n[+] Modelo Random Forest guardado en: {model_path}")
    
    print("\nReporte de Clasificación del Modelo Final (Ojo: Evaluado en Training):")
    print(classification_report(y, y_pred))
    
    # Graficar Matriz de Confusión
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Operativo/Almacenado (0)', 'Necesita Reparación (1)'],
                yticklabels=['Operativo/Almacenado (0)', 'Necesita Reparación (1)'])
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción del Modelo')
    plt.title('Matriz de Confusión - Random Forest')
    plt.tight_layout()
    plt.savefig(CM_PLOT_PATH)
    plt.close()
    print(f"\n[+] Matriz de confusión guardada en: {CM_PLOT_PATH}")
    
    # Graficar Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 10
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances[indices][:top_n], y=X.columns[indices][:top_n], palette="viridis")
    plt.title('Top 10 Variables más importantes financieramente (Random Forest)')
    plt.xlabel('Importancia Relativa')
    plt.tight_layout()
    plt.savefig(FI_PLOT_PATH)
    plt.close()
    print(f"[+] Gráfico de importancia de variables guardado en: {FI_PLOT_PATH}")

    # =========================================================================
    # MODELO 2: K-MEANS CLUSTERING (No Supervisado)
    # =========================================================================
    print("\n" + "="*80)
    print("  [B] K-MEANS CLUSTERING (Agrupación de Activos)")
    print("="*80)
    
    # Seleccionar variables continuas/numéricas importantes para clustering
    cluster_cols = [
        "price", "lifespan_days", "estimated_useful_life_days", 
        "asset_current_age_days", "linear_depreciation_rate", 
        "stagnation_time_days", "depreciation_pct"
    ]
    # Imputar lifespan_days nulos a 0 o media sólo para el clustering
    X_clust = df[cluster_cols].copy()
    X_clust["lifespan_days"] = X_clust["lifespan_days"].fillna(0)
    
    # Escalar variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clust)
    
    # Encontrar K óptimo
    print("\nEjecutando iteraciones para encontrar K óptimo (Codo y Silueta)...")
    k_values = range(2, 11)
    inertias = []
    silhouettes = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
        
    # Graficar Codo y Silueta
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:red'
    ax1.set_xlabel('Número de Clusters (k)')
    ax1.set_ylabel('Inertia (Método del Codo)', color=color)
    ax1.plot(k_values, inertias, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(k_values, silhouettes, marker='s', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Evaluación del número óptimo de clusters')
    fig.tight_layout()
    plt.savefig(KMEANS_PLOT_PATH)
    plt.close()
    print(f"[+] Gráfico del Codo y Silueta guardado en: {KMEANS_PLOT_PATH}")
    
    # Entrenar K-Means Final con K=3 (como sugirió el usuario para segmentar)
    print("\nEntrenando K-Means final con k=3...")
    kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans_final.fit_predict(X_scaled)
    
    # Mostrar el perfil de los grupos (Promedios)
    print("\nPerfiles Promedio de cada Grupo (Cluster):")
    profile = df.groupby("cluster")[cluster_cols].mean().round(2)
    print(profile.to_string())

    print("\n" + "="*80)
    print("  PIPELINE COMPLETADO CON ÉXITO")
    print("="*80)

if __name__ == "__main__":
    main()
