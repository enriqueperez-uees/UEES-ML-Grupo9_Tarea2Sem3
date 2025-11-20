# Proyecto de Aprendizaje Automático – Incidentes de Ciberseguridad (Grupo 9)

Repositorio colaborativo del **Grupo 9** para la materia de *Aprendizaje Automático* (Maestría en IA – UEES).  
El objetivo general es aplicar modelos de *Machine Learning* y técnicas de *Explainable AI (XAI)* sobre un dataset de incidentes de ciberseguridad, siguiendo un flujo completo de proyecto:

- Preparación y análisis exploratorio de datos.
- Modelos supervisados de clasificación.
- Métodos no supervisados de agrupamiento.
- Aplicación de técnicas de explicabilidad.
- Reflexión ética sobre el uso de modelos automatizados.

---

## 1. Dataset

Se utiliza el dataset sintético:

- **Nombre:** `cybersecurity synthesized data.csv`  
- **Descripción general:** cada fila representa un incidente de ciberseguridad con información sobre:
  - Tipo de ataque (`attack_type`)
  - Sistema objetivo (`target_system`)
  - Resultado del incidente (`outcome`)
  - Severidad (`attack_severity`)
  - Volumen de datos comprometidos (`data_compromised_GB`)
  - Duración del ataque (`attack_duration_min`)
  - Tiempos de respuesta (`response_time_min`)
  - Herramientas de seguridad utilizadas (`security_tools_used`)
  - Rol de usuario (`user_role`)
  - Industria y ubicación (`industry`, `location`)
  - Información temporal (`timestamp`), de la cual se derivan:
    - `hour`, `dayofweek`, `is_weekend`

Este dataset se ha utilizado de forma progresiva en las semanas 2, 3 y 4 para distintos enfoques de aprendizaje automático.

---

## 2. Estructura del repositorio

> Los nombres de archivos pueden ajustarse ligeramente según la versión final, pero la lógica general de organización es la siguiente:

```text
/UEES-ML-Grupo9_Tarea2Sem3/
├─ data/
│  └─ cybersecurity synthesized data.csv
├─ notebooks/
│  ├─ cyber_ml_semana2_modelos_supervisados.ipynb
│  ├─ cyber_mns_semana3_clustering.ipynb
│  └─ cyber_ml_semana4_xai_y_sesgos.ipynb
├─ reports/
│  ├─ semana2_modelos_supervisados/
│  │  ├─ figures/
│  │  └─ tables/
│  ├─ semana3_clustering/
│  │  ├─ figures/
│  │  └─ tables/
│  └─ semana4_xai_sesgos/
│     ├─ figures/
│     └─ tables/
└─ README.md  ← (este archivo)
```

**Carpetas principales:**

- `data/`  
  Contiene el archivo CSV base utilizado en todos los experimentos.

- `notebooks/`  
  Reúne los cuadernos de trabajo en Google Colab/Jupyter para cada semana:
  - **Semana 2:** Modelos supervisados de clasificación.
  - **Semana 3:** Modelos no supervisados de agrupamiento (clustering).
  - **Semana 4:** Modelo supervisado con técnicas de explicabilidad (SHAP, LIME) y reflexión ética.

- `reports/`  
  Carpeta para almacenar figuras, tablas y material de apoyo que se utilice en los informes y presentaciones:
  - `figures/` contiene los gráficos exportados desde los notebooks.
  - `tables/` contiene tablas o resúmenes en formato CSV/Excel, si aplica.

---

## 3. Flujo metodológico por semana

### 3.1 Semana 2 – Modelos supervisados de clasificación

**Notebook principal:**  
`notebooks/cyber_ml_semana2_modelos_supervisados.ipynb`

**Objetivo:**  
Entrenar y comparar distintos modelos supervisados para predecir el resultado del incidente (`outcome`), utilizando variables técnicas y contextuales del dataset.

**Pasos principales:**

1. **Análisis exploratorio de datos (EDA):**
   - Revisión de tipos de datos.
   - Estadísticos descriptivos básicos.
   - Distribución de la variable objetivo `outcome` y de variables clave.

2. **Preparación de datos:**
   - Tratamiento de valores faltantes y duplicados, si aparecen.
   - Codificación de variables categóricas.
   - División de datos en entrenamiento y prueba con estratificación.

3. **Modelos supervisados:**
   - Entrenamiento de al menos dos clasificadores (por ejemplo:
     - Regresión logística,
     - Árbol de decisión,
     - Random Forest,
     - SVM).
   - Ajuste básico de hiperparámetros donde corresponde.

4. **Evaluación de modelos:**
   - Métricas utilizadas:
     - Accuracy,
     - Precision,
     - Recall,
     - F1-score,
     - Matriz de confusión.
   - Discusión sobre cuál modelo ofrece mejor compromiso entre desempeño y complejidad.

5. **Resultados clave:**
   - Identificación del modelo con mejor desempeño para usarlo como referencia en semanas posteriores.
   - Observaciones sobre posibles desbalances de clase y su impacto en las métricas.

---

### 3.2 Semana 3 – Clustering y análisis de perfiles

**Notebook principal:**  
`notebooks/cyber_mns_semana3_clustering.ipynb`

**Objetivo:**  
Aplicar técnicas no supervisadas para identificar posibles **perfiles de incidentes** o patrones ocultos en el dataset.

**Pasos principales:**

1. **Selección de variables para clustering:**
   - Normalización/estandarización de variables numéricas cuando es necesario.
   - Codificación de variables categóricas o selección de un subconjunto representativo.

2. **Modelos aplicados:**
   - K-means (con análisis del número óptimo de clusters).
   - DBSCAN (explorando parámetros `eps` y `min_samples`).
   - Uso de técnicas de reducción de dimensión (PCA, t-SNE) para visualización.

3. **Interpretación de clusters:**
   - Análisis de promedio de variables por cluster.
   - Caracterización de grupos: tipo de ataque predominante, severidad, sector, etc.
   - Discusión sobre qué tipos de perfiles se identifican.

4. **Limitaciones:**
   - Sensibilidad a la selección de variables y parámetros.
   - Posibles mezclas de grupos que no son directamente interpretables.

---

### 3.3 Semana 4 – XAI, sesgos y reflexión ética

**Notebook principal:**  
`notebooks/cyber_ml_semana4_xai_y_sesgos.ipynb`

**Objetivo:**  
Partiendo de un modelo supervisado (Random Forest), aplicar técnicas de **explicabilidad** para entender cómo toma decisiones el modelo y reflexionar sobre su impacto ético en un contexto de ciberseguridad.

**Pasos principales:**

1. **Modelo base:**
   - Elección de **RandomForestClassifier** como modelo principal de clasificación.
   - Ingeniería de características:
     - Variables temporales derivadas (`hour`, `dayofweek`, `is_weekend`).
     - Eliminación de identificadores directos (`attacker_ip`, `target_ip`).
   - Validación cruzada y evaluación con métricas estándar.

2. **Importancia de variables:**
   - Revisión de la importancia de características reportada por el Random Forest.
   - Identificación de variables con mayor influencia global.

3. **Técnicas XAI aplicadas:**
   - **SHAP:**
     - Gráfico summary plot (importancia global de características).
     - Dependence plot para la característica más relevante.
   - **LIME:**
     - Explicaciones locales para casos individuales del conjunto de prueba.
     - Comparación entre predicción del modelo y etiqueta real.

4. **Comparación SHAP vs LIME:**
   - Identificación de variables que aparecen como relevantes en ambas técnicas.
   - Discusión sobre cómo SHAP aporta una visión global y LIME una visión local.
   - Relación con el contexto de incidentes de ciberseguridad.

5. **Reflexiones éticas:**
   - Riesgos de falsos positivos y falsos negativos en la clasificación de incidentes.
   - Posibles sesgos según industria, ubicación geográfica o tipo de usuario.
   - Recomendaciones para el uso responsable del modelo (human-in-the-loop, revisión periódica, trazabilidad de versiones).

---

## 4. Cómo ejecutar los notebooks

### 4.1 Requisitos

- Python 3.x
- Librerías principales:
  - `numpy`, `pandas`, `matplotlib`, `scikit-learn`
  - `shap`
  - `lime`
  - (Opcional) librerías adicionales usadas en las semanas 2 y 3 para visualización.

En Google Colab, se pueden instalar las librerías adicionales con:

```python
!pip install shap lime
```

### 4.2 Pasos sugeridos

1. Clonar el repositorio o subirlo a Google Drive.
2. Abrir cada notebook en Colab:
   - `cyber_ml_semana2_modelos_supervisados.ipynb`
   - `cyber_mns_semana3_clustering.ipynb`
   - `cyber_ml_semana4_xai_y_sesgos.ipynb`
3. Verificar que la variable `DATA_PATH` en los notebooks apunte correctamente a:
   ```python
   DATA_PATH = "../data/cybersecurity synthesized data.csv"
   ```
   o a la ruta correspondiente en tu entorno.
4. Ejecutar las celdas en orden, revisando:
   - Gráficos,
   - Métricas,
   - Tablas y comentarios intermedios.

---

## 5. Resultados y aportes principales

- Se implementaron y compararon distintos modelos supervisados de clasificación aplicados al problema de detección de patrones en incidentes de ciberseguridad.
- Se exploraron técnicas de clustering para identificar grupos de incidentes con características similares, apoyando el análisis de perfiles de riesgo.
- Se aplicaron técnicas de explicabilidad (SHAP y LIME) para entender mejor cómo el modelo toma decisiones, identificando:
  - Variables de mayor impacto,
  - Diferencias entre explicaciones globales y locales,
  - Posibles señales de sesgo.
- Se desarrollaron reflexiones sobre:
  - Transparencia y auditabilidad de modelos,
  - Riesgos éticos y sociales de su implementación,
  - Recomendaciones para un uso responsable en entornos reales.

---

## 6. Integrantes del Grupo 9

- **Angie Blandón**
- **Enrique Pérez**
- **Jimmy Rodríguez**
- **Fredy Aguirre**

Cada integrante participó en el desarrollo de los notebooks, análisis de resultados y elaboración de las reflexiones y presentaciones asociadas a las semanas 2, 3 y 4.

---

## 7. Notas finales

- Este repositorio sirve como portafolio técnico del trabajo desarrollado en la asignatura de *Aprendizaje Automático*.
- Los notebooks están pensados para ser leídos de forma secuencial (Semana 2 → Semana 3 → Semana 4), de manera que se aprecie la evolución del análisis desde los modelos supervisados hasta la explicabilidad y la reflexión ética.
