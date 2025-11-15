# Taller Colaborativo – Semana 3  
## Segmentación de incidentes de ciberseguridad con aprendizaje no supervisado

Este repositorio corresponde al Taller Colaborativo de la **Semana 3** de la asignatura de *Machine Learning* de la Maestría en Inteligencia Artificial.  

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

```text
.
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
│   │   │   ├── Distribuciones de variables numéricas.png
│   │   │   ├── Gráfico k-distancia para estimar eps (DBSCAN).png
│   │   │   ├── Método del codo – K-means.png
│   │   │   ├── PCA (2D) coloreado por clusters DBSCAN.png
│   │   │   ├── PCA (2D) coloreado por clusters K-means.png
│   │   │   ├── Silhouette para diferentes k (muestra).png
│   │   │   ├── Sistemas objetivo.png
│   │   │   ├── Top tipos de ataque.png
│   └── └── └── tsne_clusters.pngt-SNE (2D) – muestra coloreada por clusters K-means.png
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
  - `attack_severity`: nivel cuantitativo de severidad del incidente (1 a 10).
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
   - Se calcula la inercia para valores de `k` en un rango de 2 a 10 y se gráfica el **método del codo**.
   - Se calcula el **silhouette score** sobre una muestra del dataset para los mismos valores de `k`.
   - El mejor compromiso se obtiene con **`k = 8`**, donde:
     - La inercia sigue disminuyendo, pero la curva del codo entra en una zona de rendimientos decrecientes.
     - El silhouette score alcanza su valor máximo (~0.226), superior al de valores vecinos de `k`.

   En otras palabras, `k = 8` ofrece un buen equilibrio entre separación de grupos y complejidad del modelo, evitando tanto la sobreagrupación como la fragmentación excesiva.

2. **Entrenamiento del modelo final**
   - Entrenamiento de `KMeans(n_clusters=8, random_state=42, n_init=10)`.
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

4. **Perfiles de incidentes (macroperfiles)**

Aunque las medias numéricas son relativamente similares (≈50 GB, ≈150 minutos de duración, ≈90 minutos de respuesta), K-means permite distinguir **macroperfiles** con combinaciones diferentes de severidad e impacto. A un nivel de negocio, estos clusters se pueden reagrupar en:

- **Incidentes de baja severidad y bajo impacto**, que se asemejan a ruido operativo o intentos contenidos.  
- **Incidentes de severidad media**, con impacto moderado y tiempos de respuesta razonables, que representan la operación “normal” del SOC.  
- **Incidentes de alta severidad**, que deben ser priorizados por el equipo de seguridad por su nivel de riesgo, incluso si el volumen de datos y los tiempos promedio son similares.

Esta descripción responde a la pregunta:  
**¿Qué tipo de perfiles se pueden identificar?** (desde la perspectiva de K-means).

### 5.2. DBSCAN

1. **Estimación de `eps` mediante gráfico k-distancia**
   - Sobre una muestra de tamaño razonable (5 000 registros).
   - Se calcula la distancia al 5.º vecino y se grafica la curva ordenada.
   - El “codo” de la curva orienta el valor inicial de `eps` alrededor de 0.3.

2. **Prueba de combinaciones de hiperparámetros**
   - Se probaron varias combinaciones de `eps` y `min_samples`.
   - Con `eps` altos (0.5 y 0.7) DBSCAN produjo un solo cluster y 0 ruido → modelo sin capacidad de distinguir estructura.
   - Con `eps = 0.3` y `min_samples = 20` se obtuvieron 48 clusters y casi 5 000 puntos de ruido → sobrefragmentación.

3. **Selección de parámetros finales**

La configuración elegida fue:

```python
eps_final = 0.3
min_samples_final = 10
```

Con estos valores:

- Se obtuvieron **10 clusters** y solo un **0.05 % de puntos marcados como ruido**, un equilibrio razonable entre número de grupos y cantidad de outliers.
- Los clusters son suficientemente densos y no se pierde demasiada información en forma de ruido.

4. **Resultados con DBSCAN**

Al analizar la tabla de medias, se observa que:

- `data_compromised_GB` se mantiene alrededor de 49–51 GB en todos los clusters.
- `attack_duration_min` oscila muy ligeramente entre ~149 y 152 minutos.
- `response_time_min` también es muy estable (~89–92 minutos).
- La variable que realmente diferencia los clusters es **`attack_severity`**, cuyos valores van de 1 a 10, y prácticamente definen cada grupo.

En la práctica, DBSCAN está segmentando los incidentes por **niveles de severidad**, mientras que las otras variables permanecen casi constantes.

5. **Interpretación**

Los clusters de DBSCAN pueden reagruparse en:

- **Perfiles de baja severidad** (severidad 1–3).  
- **Perfiles de severidad media** (severidad 4–7).  
- **Perfiles de alta severidad** (severidad 8–10).

Esto refuerza la severidad como indicador dominante del riesgo en este dataset, pero no descubre patrones nuevos en cuanto a duración o tiempos de respuesta.

---

## 6. Reducción y visualización: PCA y t-SNE

### 6.1. PCA (Principal Component Analysis)

- Se aplica PCA con 2 componentes principales sobre `X_scaled`.
- Las dos primeras componentes capturan una proporción relevante de la varianza, aunque no la totalidad.
- En el gráfico de **PCA (2D) coloreado por clusters K-means**:

  - Los incidentes forman una nube continua en forma de elipse.
  - Algunos clusters tienden a ocupar zonas predominantes del plano, pero existe **solapamiento** entre colores, lo que indica que la estructura no es puramente lineal.
  - Aun así, PCA ofrece una **visión global** del espacio de incidentes y permite verificar que la asignación de clusters no es completamente aleatoria.

### 6.2. t-SNE

- Se aplica `TSNE(n_components=2)` sobre una muestra de 5 000 registros por temas de costo computacional.
- En el gráfico de **t-SNE (2D) coloreado por clusters K-means**:

  - Los 8 clusters aparecen como **“islas” bien definidas**, con muy poco solapamiento.
  - Incidentes del mismo cluster se agrupan en regiones compactas, mientras que incidentes de clusters diferentes quedan claramente separados.

- t-SNE, por lo tanto, proporciona una **validación visual fuerte** de que los 8 clusters de K-means representan grupos coherentes en términos de vecindad local.

### 6.3. Rol conjunto de PCA y t-SNE

- **PCA** ayuda a entender cómo se distribuyen los incidentes en el espacio de mayor varianza y a comprobar que, aunque la separación no es perfecta, los clusters de K-means tienen cierta estructura.  
- **t-SNE** enfatiza la separación local entre grupos y muestra que los clusters encontrados por K-means forman grupos consistentes.

En conjunto, ambas técnicas respaldan la elección de `k = 8` y la validez cualitativa de los perfiles de incidentes definidos.

---

## 7. Comparación de modelos y uso de visualizaciones

### 7.1. Diferencias clave entre K-means y DBSCAN en este dataset

- **K-means**:
  - Requiere especificar el número de clusters `k`.
  - Produce 8 clusters de tamaño relativamente similar y forma aproximadamente esférica.
  - Construye **macroperfiles** que combinan varios niveles de severidad, datos comprometidos, duración y tiempos de respuesta.
  - Es útil para obtener una **segmentación operativa compacta**, fácil de comunicar y emplear en la priorización de incidentes.

- **DBSCAN**:
  - No requiere fijar `k`, pero es muy sensible a `eps` y `min_samples`.
  - Con la configuración seleccionada (`eps = 0.3`, `min_samples = 10`) genera 10 clusters y casi nada de ruido.
  - En este dataset termina agrupando principalmente por **severidad**, ya que las demás variables son muy homogéneas entre grupos.
  - Resulta útil para **estratificar** el conjunto de incidentes por niveles de severidad, pero aporta poco en términos de descubrimiento de nuevas estructuras de impacto/duración.

En síntesis:

- K-means ofrece una visión más **agregada y práctica** de los incidentes para la gestión diaria.
- DBSCAN refuerza la importancia de la **severidad** como variable dominante, pero no introduce perfiles cualitativamente nuevos.

### 7.2. Rol de las visualizaciones

Las visualizaciones (distribuciones, curvas de codo y silhouette, PCA 2D, t-SNE 2D, tablas de medias por cluster) se utilizaron para:

- **Justificar decisiones técnicas**:
  - Elección de `k = 8` en K-means.
  - Selección de `eps = 0.3` y `min_samples = 10` en DBSCAN.
- **Comunicar hallazgos** de forma clara:
  - Perfilamiento de incidentes por niveles de severidad y riesgo.
  - Confirmación visual de la coherencia de los clusters.

Estas visualizaciones se integran en la **presentación técnica de 8 minutos** requerida en el taller.

---

## 8. Limitaciones y trabajo futuro

Entre las principales limitaciones identificadas:

1. **Dataset sintético y homogéneo**

   - Las medias de `data_compromised_GB`, `attack_duration_min` y `response_time_min` son muy similares entre clusters.
   - Esto limita la capacidad de los algoritmos para encontrar grupos muy diferenciados y hace que la segmentación esté dominada por `attack_severity`.

   **Posible solución:**  
   Trabajar con un dataset real o con datos sintéticos que incorporen mayor variabilidad en impacto, duración y tiempos de respuesta.

2. **Uso exclusivo de variables numéricas**

   - Las variables categóricas (`attack_type`, `target_system`, `industry`, etc.) se usaron solo para interpretación.
   - Esto puede ocultar patrones relevantes, por ejemplo, ataques específicos a ciertos sistemas.

   **Posible solución:**  
   Incluir estas variables en el vector de características mediante **codificación one-hot** y reentrenar los modelos para evaluar si mejoran la interpretabilidad de los clusters.

3. **Sensibilidad a hiperparámetros y métricas moderadas**

   - El silhouette score de K-means es moderado (~0.22), lo que indica una separación correcta, pero no excelente.
   - DBSCAN es muy sensible a pequeñas variaciones de `eps` y `min_samples`.

   **Posible solución:**  
   Explorar sistemáticamente rejillas de hiperparámetros y considerar otros algoritmos de clustering (Gaussian Mixture Models, clustering jerárquico, HDBSCAN).

4. **Falta de validación con expertos de dominio**

   - La interpretación de perfiles se basa solo en estadísticas y visualizaciones.

   **Posible solución:**  
   Presentar los resultados a analistas de seguridad (SOC, CSIRT), recoger retroalimentación y ajustar la definición de perfiles y variables a partir de su experiencia.

---

## 9. Conclusiones principales

- Es posible segmentar los incidentes de ciberseguridad en **perfiles de riesgo** utilizando técnicas de clustering, incluso sobre un dataset sintético.
- **K-means con 8 clusters** proporciona una segmentación operativa en macroperfiles que combinan severidad, datos comprometidos, duración y tiempos de respuesta.  
- **DBSCAN** refuerza principalmente la **estratificación por severidad**, mostrando que esta variable es la que realmente domina la estructura del dataset.  
- **PCA** y **t-SNE** son herramientas complementarias para validar y comunicar visualmente la calidad de los clusters.

Aun con sus limitaciones, este enfoque de aprendizaje no supervisado puede servir como base para sistemas de priorización de incidentes de ciberseguridad y para el diseño de estrategias de respuesta diferenciadas.

---

## 10. Referencias

Habeeb, M. (s. f.). *Cybersecurity Incident Dataset* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/mustafahabeeb90/cybersecurity-incident-dataset  

scikit-learn developers. (2024). *User guide: Clustering*. Scikit-learn. https://scikit-learn.org/stable/modules/clustering.html  
