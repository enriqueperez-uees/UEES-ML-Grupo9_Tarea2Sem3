# Taller Colaborativo – Semana 3  
## Segmentación de incidentes de ciberseguridad con aprendizaje no supervisado

Este repositorio corresponde al Taller Colaborativo de la **Semana 3** de la asignatura de *Aprendizaje Automático* de la Maestría en Inteligencia Artificial.  

El objetivo es aplicar técnicas de **aprendizaje no supervisado** para segmentar incidentes de ciberseguridad y obtener **perfiles de riesgo** que apoyen la toma de decisiones en un entorno tecnológico.

---

## 1. Contexto y objetivo

Se trabaja con el dataset sintético **Cybersecurity Incident Dataset** publicado en Kaggle (Habeeb, s. f.), que describe incidentes de ciberseguridad con variables como:

- Tipo de ataque: `attack_type` (p. ej., Phishing, Malware, DDoS, Ransomware).  
- Sistema objetivo: `target_system` (p. ej., Cloud, On-premise, IoT, Mobile).  
- Resultado del incidente: `outcome` (Success/Failure).  
- Datos comprometidos: `data_compromised_GB`.  
- Duración del ataque: `attack_duration_min`.  
- Severidad: `attack_severity`.  
- Tiempo de respuesta: `response_time_min`.  
- Variables de contexto: industria, ubicación, herramientas de seguridad, método de mitigación, entre otras.

**Objetivo del taller**

- Implementar y analizar los modelos:
  - **K-means** y **DBSCAN** para clustering.
  - **PCA** y **t-SNE** para reducción y visualización en 2D.
- Identificar **perfiles de incidentes** en función de impacto y gestión.
- Comparar los métodos y comunicar conclusiones de forma técnica y visual.

Este trabajo se alinea con la rúbrica del curso en los componentes de:

- Análisis y limpieza del dataset.  
- Implementación de K-means y DBSCAN.  
- Aplicación de PCA/t-SNE.  
- Visualización y segmentación de perfiles.  
- Análisis crítico y propuestas de uso.

---

## 2. Estructura del repositorio

> 

```text
.
/UEES-ML-Grupo9_Tarea2Sem3/
│
├── data/
│   └── cybersecurity synthesized data.csv
│
├── notebooks/
│   ├── cyber_ml_semana2.ipynb
│   └── cyber_mns_semana3.ipynb (nuevo)
│
├── reports/
│   ├── cyber_ml_semana2 (modelos supervisados)/ (mover)
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
│   ├── cyber_mns_semana3 (modelos no supervisados)/ (nuevo)
│   │   ├── figures/
│   │   │   ├── kmeans_clusters.png
│   │   │   ├── dbscan_resultados.png
│   │   │   ├── pca_clusters.png
│   └── └── └── tsne_clusters.png
│
├── old/ (nuevo)
│   ├── semana2/
│   └── └── README.md (modificado)
│
├── README.md (modificar)

```

- `data/`: contiene el archivo CSV con los incidentes sintetizados.
- `notebooks/`: incluye el notebook principal con todo el flujo (EDA, modelos, visualización).
- `figures/`: almacena las imágenes exportadas utilizadas en la presentación.
- `README.md`: documentación técnica, justificación y síntesis de resultados.

---

## 3. Dataset utilizado

- **Fuente**: Kaggle – *Cybersecurity Incident Dataset* (Habeeb, s. f.).  
- **Tipo**: datos sintéticos que simulan incidentes reales de ciberseguridad.
- **Granularidad**: cada fila representa un incidente individual.
- **Variables clave para clustering (numéricas)**:
  - `data_compromised_GB`: volumen de datos comprometidos (impacto en confidencialidad).
  - `attack_duration_min`: duración del ataque en minutos (persistencia).
  - `attack_severity`: nivel cuantitativo de severidad del incidente.
  - `response_time_min`: tiempo de respuesta del equipo de seguridad (eficiencia operativa).

Estas variables permiten describir los incidentes en dos dimensiones fundamentales:

1. **Impacto del ataque** (cantidad de datos y severidad).  
2. **Gestión del incidente** (duración y tiempo de respuesta).

Las variables categóricas (`attack_type`, `target_system`, `industry`, etc.) se utilizan como **contexto** para interpretar los clusters obtenidos.

---

## 4. Metodología

La metodología sigue el flujo recomendado en el curso y en la documentación de *scikit-learn* (scikit-learn developers, 2024):

### 4.1. Análisis exploratorio y limpieza (EDA)

En el notebook se realizan las siguientes etapas:

1. **Exploración inicial**
   - `df.shape`, `df.info()`, `df.describe()` para conocer tamaño, tipos de datos y estadísticas básicas.
   - Identificación de columnas numéricas y categóricas.

2. **Valores faltantes**
   - Cálculo de `df.isna().sum()` por columna.
   - Decisiones explícitas sobre tratamiento de nulos:  
     - Eliminación de filas/columnas irrelevantes o con alta proporción de nulos.  
     - Mantenimiento de columnas cuando el impacto es mínimo.  
   - Justificación de las decisiones en el notebook.

3. **Distribución de variables numéricas**
   - Histogramas para `data_compromised_GB`, `attack_duration_min`, `attack_severity`, `response_time_min`.
   - Comentario sobre sesgos, presencia de outliers y su impacto potencial en algoritmos basados en distancia.

4. **Exploración de variables categóricas**
   - Gráficos de barras para:
     - Tipos de ataque más frecuentes (`attack_type`).  
     - Sistemas objetivo (`target_system`).  
   - Descripción de patrones relevantes (p. ej., predominio de cierto tipo de ataque o sistema objetivo).

5. **Selección de variables para clustering**
   - Se priorizan las variables numéricas relacionadas con:
     - Impacto del incidente.
     - Duración y tiempo de respuesta.
   - Otras variables (IPs, identificadores, timestamp crudo) se consideran poco informativas para la segmentación y se excluyen del vector de características.

### 4.2. Preparación de datos

- **Escalado de variables**  
  Se aplica `StandardScaler` a las variables numéricas seleccionadas para evitar que diferencias de escala dominen el cálculo de distancias:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Este paso es crítico para el correcto funcionamiento de **K-means** y **DBSCAN**.

---

## 5. Modelos de clustering

### 5.1. K-means

1. **Selección del número de clusters (`k`)**
   - Se calcula la inercia para valores de `k` en un rango (por ejemplo, 2 a 10) y se gráfica el **método del codo**.
   - Se calcula el **silhouette score** sobre una muestra del dataset para los mismos valores de `k`.
   - Con base en ambos criterios se elige un valor:

> **TODO – ACTUALIZAR:**  
> “Se seleccionó `k = X` porque presenta un buen compromiso entre baja inercia y un silhouette score aceptable, evitando tanto la sobresegmentación como la agrupación excesiva.”

2. **Entrenamiento del modelo final**
   - Entrenamiento de `KMeans(n_clusters=k_optimo, random_state=42, n_init=10)`.
   - Creación de una columna `KMeans_Cluster` en el dataframe original.

3. **Perfilamiento de clusters**
   - Cálculo de medias por cluster para las variables numéricas:

     ```python
     df.groupby('KMeans_Cluster')[features].mean()
     ```

   - Identificación del tipo de ataque más frecuente por cluster:

     ```python
     df.groupby('KMeans_Cluster')['attack_type'].agg(lambda x: x.value_counts().index[0])
     ```

4. **Perfiles de incidentes (ejemplo de estructura)**

> **TODO – ACTUALIZAR CON DATOS REALES**  
> (Los nombres y descripciones deben adaptarse a los resultados reales; esto es un ejemplo de plantilla.)

- **Cluster 0 – Incidentes críticos de alto impacto**  
  - Severidad: alta.  
  - Datos comprometidos: altos.  
  - Duración del ataque: prolongada.  
  - Tiempo de respuesta: lento.  
  - Tipo de ataque predominante: [p. ej., Ransomware].  
  - Interpretación: incidentes de alto riesgo que demandan máxima prioridad.

- **Cluster 1 – Incidentes moderados bien gestionados**  
  - Severidad: media.  
  - Datos comprometidos: moderados.  
  - Duración: intermedia.  
  - Tiempo de respuesta: relativamente rápido.  
  - Tipo de ataque predominante: [p. ej., Malware genérico].  
  - Interpretación: incidentes frecuentes, pero razonablemente controlados.

- **Cluster 2 – Incidentes de bajo impacto / ruido operativo**  
  - Severidad: baja.  
  - Datos comprometidos: casi nulos.  
  - Duración: corta.  
  - Tiempo de respuesta: rápido.  
  - Tipo de ataque predominante: [p. ej., escaneos o intentos fallidos].  
  - Interpretación: eventos de bajo riesgo, útiles para medir el “ruido” del entorno.

- **Cluster 3 – [Nombre según resultados]**  
  - [Completar con patrones observados].

Esta descripción responde directamente a la pregunta:  
**¿Qué tipo de perfiles se pueden identificar?**

### 5.2. DBSCAN

1. **Estimación de `eps` mediante gráfico k-distancia**
   - Sobre una muestra de tamaño razonable (p. ej., 5 000 registros).
   - Se calcula la distancia al 5.º vecino y se grafica la curva ordenada.
   - El “codo” de la curva orienta el valor inicial de `eps`.

2. **Prueba de combinaciones de hiperparámetros**
   - Se prueban varias combinaciones de `eps` y `min_samples`.
   - Para cada combinación se reportan:
     - Número de clusters encontrados.
     - Cantidad de puntos marcados como ruido (`-1`).

3. **Selección de parámetros finales**

> **TODO – ACTUALIZAR:**  
> “Se seleccionó `eps = X` y `min_samples = Y` porque produce Z clusters interpretables y una proporción de ruido de aproximadamente W %, lo que permite identificar incidentes atípicos sin perder la estructura principal de los datos.”

4. **Resultados con DBSCAN**
   - Se crea la columna `DBSCAN_Cluster`.
   - Se calculan perfiles de cluster (excluyendo `-1`) de forma similar a K-means.
   - Se calcula la proporción de ruido:

     ```python
     noise_ratio = (df['DBSCAN_Cluster'] == -1).mean()
     ```

5. **Interpretación**
   - Los clusters de DBSCAN suelen resaltar:
     - Grupos densos de incidentes de características muy homogéneas.
     - Casos en los que K-means pudo mezclar subgrupos.
   - Los puntos marcados como ruido se interpretan como:
     - Incidentes **atípicos o extremos** (posibles outliers relevantes).
     - Registros que no se ajustan a ningún patrón denso claro.

---

## 6. Reducción y visualización: PCA y t-SNE

### 6.1. PCA (Principal Component Analysis)

- Se aplica PCA con 2 componentes principales sobre `X_scaled`.
- Se reporta el porcentaje de varianza explicada:

> **TODO – ACTUALIZAR:**  
> “Las dos primeras componentes principales explican aproximadamente **XX %** de la varianza total.”

- Se generan gráficos 2D de PCA coloreados por:
  - `KMeans_Cluster`.
  - `DBSCAN_Cluster`.

Estos gráficos permiten:

- Evaluar visualmente si los clusters están razonablemente separados.
- Identificar solapamientos o estructuras lineales.

### 6.2. t-SNE

- Se aplica `TSNE(n_components=2)` sobre una **muestra** del dataset (p. ej., 5 000 registros) por temas de costo computacional.
- Se grafica el resultado 2D coloreado por `KMeans_Cluster`.

t-SNE ayuda a:

- Explorar estructuras **no lineales**.
- Detectar subgrupos dentro de un mismo cluster de K-means.
- Validar si los clusters capturan patrones locales de forma coherente.

---

## 7. Comparación de modelos y uso de visualizaciones

### 7.1. Diferencias clave entre K-means y DBSCAN

En este caso:

- **K-means**:
  - Requiere especificar el número de clusters `k`.
  - Produce clusters de tamaño relativamente similar y forma aproximadamente esférica.
  - Es adecuado para obtener una **segmentación estable** de niveles de riesgo/impacto.

- **DBSCAN**:
  - No requiere fijar `k`, pero sí elegir `eps` y `min_samples`.
  - Identifica clusters de distinta densidad y marca **ruido**.
  - Es útil para detectar incidentes **atípicos o muy específicos** que K-means podría diluir en clusters grandes.

> **TODO – ACTUALIZAR:**  
> Incluir 1–2 ejemplos concretos de cómo difieren los agrupamientos para este dataset según los resultados reales del equipo.

### 7.2. Rol de las visualizaciones

Las visualizaciones (histogramas, gráficos de barras, PCA 2D, t-SNE 2D, curvas de codo y silhouette) se utilizan para:

- **Justificar decisiones técnicas** (elección de `k`, selección de hiperparámetros en DBSCAN).
- **Comunicar hallazgos** a una audiencia no técnica:
  - Perfiles de incidentes.
  - Diferencias de severidad, pérdida de datos y tiempos de respuesta.
  - Presencia de incidentes atípicos.

Estas visualizaciones se integran en la **presentación oral / en video** de 5–10 minutos exigida en el taller.

---

## 8. Limitaciones y trabajo futuro

Entre las principales limitaciones identificadas:

- El dataset es **sintético**, por lo que puede no reflejar todas las complejidades de un entorno real de ciberseguridad.
- El clustering se realizó principalmente con **variables numéricas**; las variables categóricas se usaron solo para interpretación.
- Los modelos son sensibles a:
  - Escala de las variables (resuelto parcialmente con estandarización).
  - Selección de hiperparámetros (`k`, `eps`, `min_samples`).
- Técnicas como t-SNE y el cálculo del silhouette score pueden ser **costosos** en grandes volúmenes de datos.

Posibles mejoras futuras:

- Integrar variables categóricas mediante **codificación one-hot** y evaluar su impacto en la calidad de los clusters.
- Probar otros algoritmos de clustering (p. ej., Gaussian Mixture Models, HDBSCAN).
- Incorporar métricas de negocio (pérdida económica, impacto en SLA) para redefinir los perfiles en términos de riesgo operativo.
- Validar los resultados con **expertos en ciberseguridad** y con datos reales de una organización.

---

## 9. Referencias

Habeeb, M. (s. f.). *Cybersecurity Incident Dataset* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/mustafahabeeb90/cybersecurity-incident-dataset  

scikit-learn developers. (2024). *User guide: Clustering*. Scikit-learn. https://scikit-learn.org/stable/modules/clustering.html  
