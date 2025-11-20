# Predicción de Incidentes de Ciberseguridad mediante Modelos de Clasificación Supervisada

## Descripción General

El propósito de este proyecto es desarrollar, entrenar y comparar distintos modelos de clasificación supervisada aplicados a un conjunto de datos técnicos relacionados con incidentes de ciberseguridad.  
El objetivo principal fue identificar el modelo con mejor capacidad predictiva y evaluar su rendimiento utilizando métricas estándar de clasificación: Accuracy, Precision, Recall y F1-Score.

---

## 1. Conjunto de Datos

- Filas: 100,000  
- Columnas: 16  
- Origen: Dataset simulado con registros de ataques cibernéticos.  

### Variables principales
- Categóricas: `attack_type`, `target_system`, `result`, `security_tools_used`, `user_role`, `location`, `industry`, `mitigation_method`
- Numéricas: `data_compromised_GB`, `attack_duration_min`, `attack_severity`, `response_time_min`
- Variable objetivo: `mes` (mes del incidente, valores 1–12)

### Observaciones clave del EDA
- Sin valores nulos ni atípicos significativos.  
- Distribuciones uniformes, tanto en variables numéricas como categóricas.  
- Bajas correlaciones entre variables numéricas (independencia entre atributos).  
- Distribución balanceada entre los 12 meses, sin sesgos temporales marcados.  
- Los datos parecen sintéticos y equilibrados artificialmente, lo que reduce la presencia de patrones fuertes.

---

## 2. Metodología

### 2.1. Preprocesamiento
- Codificación de variables categóricas: `OneHotEncoder`  
- Escalado de variables numéricas: `MinMaxScaler`  
- Eliminación de columnas irrelevantes: IPs y marcas de tiempo  
- Uso de un pipeline de Scikit-learn que integra preprocesamiento y modelo en un flujo reproducible.  

### 2.2. Modelos Implementados
1. **Árbol de Decisión (`DecisionTreeClassifier`)**  
   - Parámetros óptimos: `max_depth=20`, `min_samples_split=5`  
   - F1-score promedio: 0.507  

2. **SVM Lineal (`LinearSVC`)**  
   - Parámetros óptimos: `kernel='linear'`, `C=0.1`, `gamma='scale'`  
   - F1-score promedio: 0.513 (mejor resultado)  

3. **Random Forest (`RandomForestClassifier`)**  
   - Parámetros óptimos: `n_estimators=50`, `max_depth=10`, `min_samples_split=2`  
   - F1-score promedio: 0.507  

### 2.3. Evaluación
Se utilizó validación cruzada (k=3) y las métricas:  
- Accuracy: proporción de aciertos totales.  
- Precision: confiabilidad de predicciones positivas.  
- Recall: capacidad para detectar casos positivos reales.  
- F1-Score: equilibrio entre precisión y recall.  

---

## 3. Resultados Comparativos

| Modelo             | Accuracy | Precision | Recall | F1-Score |
|--------------------|-----------|------------|---------|-----------|
| Árbol de Decisión  | 0.501     | 0.502      | 0.502   | 0.499     |
| SVM (Lineal)       | 0.497     | 0.497      | 0.497   | 0.495     |
| Random Forest      | 0.498     | 0.498      | 0.498   | 0.496     |

**Conclusiones de desempeño:**
- Los tres modelos obtuvieron métricas similares (~0.50).  
- El SVM lineal fue ligeramente superior en F1-score (0.513 en validación cruzada).  
- Los resultados sugieren que los datos no presentan patrones fuertes que permitan una separación efectiva entre clases.

---

## 4. Interpretación y Análisis

- Los datos equilibrados y sin correlaciones significativas limitan la capacidad de discriminación entre clases.  
- Rendimientos cercanos al azar (~50%) indican que el conjunto de datos carece de estructura predictiva clara.  
- Los modelos son robustos pero limitados por la naturaleza del dataset.  
- Árbol de Decisión: modelo más interpretable y equilibrado.  
- SVM Lineal: mejor rendimiento general, aunque con mayor costo computacional.  
- Random Forest: estable, pero sin mejoras sustanciales frente al árbol individual.

---

## 5. Conclusiones Finales

- Los tres modelos mostraron rendimiento similar al azar, evidenciando la limitada capacidad predictiva del dataset.  
- La metodología (EDA, pipeline, validación cruzada, optimización de hiperparámetros) es correcta y replicable para futuros conjuntos de datos.  
- Se recomienda:  
  - Aplicar ingeniería de características o seleccionar nuevas variables.  
  - Probar modelos basados en boosting (XGBoost, LightGBM).  
  - Considerar aumento de datos o simulaciones más realistas.

---

## 6. Reflexión Técnica

Este proyecto permitió:
- Comprender la importancia del preprocesamiento y la correcta transformación de datos.  
- Evidenciar que la calidad y relevancia de las variables influye más que la complejidad del modelo.  
- Destacar el uso de pipelines como herramienta esencial para garantizar reproducibilidad y eficiencia.  

**Modelos comparados:**  
- Árbol de Decisión: interpretabilidad.  
- SVM: capacidad de generalización.  
- Random Forest: estabilidad y robustez.  

En conjunto, este trabajo demuestra que el proceso de modelado supervisado debe acompañarse de una revisión crítica del valor informativo de las variables para alcanzar un rendimiento predictivo superior.

---

## 7. Autor
Proyecto desarrollado con fines académicos por 
**AGUIRRE PENAFIEL FREDY MESIAS  
BLANDON TRUJILLO ANGIE TATIANA
PÉREZ BARRERA ENRIQUE ALBERTO 
RODRIGUEZ GALAN JIMMY VICENTE 
 (2025)**.  

## Estructura
```
.
/UEES-ML-Grupo9_Tarea2Sem3/
│
├── data/
│   └── cybersecurity synthesized data.csv
│
├── notebooks/
│   ├── cyber_ml_semana2.ipynb
│   ├── cyber_mns_semana3.ipynb.ipynb
│   └── cyber_ml_semana4_xai_y_sesgos.ipynb (nuevo)
│
├── reports/
│   ├── cyber_ml_semana4_xai_y_sesgos/ (nuevo)
│   │   ├── figures/
│   │   │   ├── 1_DistribucionVariableObjetivo.png
│   │   │   ├── 2_MatrizConfusionRandomForest.png
│   │   │   ├── 3_TopCaracteristicasRF.png
│   │   │   ├── 4_SHAP summary plot.png
│   └── └── └── 5_SHAP dependence plot.png
│   ├── cyber_mns_semana3 (modelos no supervisados)/
│   │   ├── figures/
│   │   │   ├── kmeans_clusters.png
│   │   │   ├── dbscan_resultados.png
│   │   │   ├── pca_clusters.png
│   └── └── └── tsne_clusters.png
│   ├── cyber_ml_semana2 (modelos supervisados)/
│   │   ├── figures/
│   │   │   ├── 1_RelacionesEntreVariablesNumericasYTipoAtaque.png
│   │   │   ├── 2_DistribucionesNumericas.png
│   │   │   ├── 3_BoxplotsVariablesNumericas.png
│   │   │   ├── 4_CorrelacionesNumericas.png
│   │   │   ├── 5_FrecuenciaTiposAtaque.png
│   │   │   ├── 6_resultadosAtaques.png
│   │   │   ├── 7_Top10UbicacionesAfectadas.png
│   │   │   ├── 8_Top10IndustriasAfectadas.png
│   │   │   ├── 9_DuracionAtaquePorTipo.png
│   │   │   ├── 10_SeveridadAtaque.png
│   │   │   ├── 11_CantidadAtaquesPorMes.png
│   │   │   ├── 12_MatrizConfusion.png
│   │   │   └── 13_ComparacionModelos.png
│   │   └── tables/
│
├── old/
│   ├── README.md
│   └── README.semana3.md
│
└── README.md (modificar)

Implementado en **Python** utilizando librerías de **Scikit-learn**, **Pandas**, **NumPy** y **Matplotlib**.
