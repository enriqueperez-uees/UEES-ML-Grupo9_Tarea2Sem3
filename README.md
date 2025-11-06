# UEES-ML-Grupo9_Tarea2Sem3
Objetivo
El propósito del análisis es desarrollar, entrenar y comparar distintos modelos de clasificación supervisada (Árbol de Decisión, SVM y Random Forest) aplicados a un conjunto de datos técnicos relacionados con incidentes de ciberseguridad.
El objetivo final era identificar el modelo con mejor capacidad predictiva y evaluar su rendimiento mediante métricas estándar: accuracy, precision, recall y F1-score.
 
1.	Análisis Exploratorio de Datos (EDA)
Dimensiones del dataset:
●	Filas: 100,000

●	Columnas: 16

Descripción general del conjunto de datos:
El dataset simula registros de ciberataques, incluyendo tipo de ataque, sistema afectado, resultado, severidad, tiempo de respuesta, métodos de mitigación, y ubicación geográfica.
Columnas principales:
Variable	Tipo	Descripción
attack_type	Categórica	Tipo de ataque (Phishing, DDoS, Ransomware, etc.)
target_system	Categórica	Sistema objetivo (Cloud Service, Database, API, etc.)
outcome	Categórica	Resultado del ataque (Success/Failure)
data_compromised_GB	Numérica	Datos comprometidos (GB)
attack_duration_min	Numérica	Duración del ataque (minutos)
attack_severity	Numérica	Severidad (escala 1-10)
response_time_min	Numérica	Tiempo de respuesta (minutos)
mitigation_method	Categórica	Método de mitigación utilizado
month	Numérica	Mes del incidente
Valores nulos:

 No se encontraron valores nulos en ninguna columna.
Variables categóricas:
●	attack_type: 8 categorías
●	target_system: 8 categorías
●	outcome: 2 categorías
●	security_tools_used: 8 categorías
●	user_role: 4 categorías
●	location: 10 categorías
●	industry: 8 categorías
●	mitigation_method: 5 categorías

Distribución de la variable objetivo (month):

Los ataques se distribuyen de forma relativamente uniforme a lo largo de los meses, con ligera mayor frecuencia en marzo, agosto y diciembre.
Estadísticas descriptivas numéricas:
Variable	Media	Mínimo	Máximo	Desv. Est.
data_compromised_GB	50.06	0	100	28.82
attack_duration_min	151.07	1	300	86.73
attack_severity	5.49	1	10	2.86
response_time_min	90.45	1	180	51.90

1.1 Visualizaciones EDA
Distribuciones numéricas:
 Se observa una distribución uniforme en la mayoría de las variables continuas, sin presencia significativa de outliers.
 
Boxplots:
 Las variables numéricas presentan rangos amplios, especialmente attack_duration_min y response_time_min, lo que sugiere variabilidad alta en los tiempos de ataque y respuesta.
 
Mapa de correlación:
 Las correlaciones entre variables numéricas son prácticamente nulas, lo que indica independencia entre ellas y refuerza la necesidad de usar modelos de clasificación no lineales. 
Gráficos de frecuencia:
●	Tipos de ataque: Brute Force, DDoS y Zero-Day Exploit son los más comunes 
●	Resultados del ataque: Proporción casi equilibrada entre Success (50.03%) y Failure (49.97%).
 
●	Top 10 ubicaciones más afectadas: Brasil, USA y Canadá lideran los incidentes.
 
●	Top industrias afectadas: Gobierno, manufactura y Educación son las más impactadas.
 
●	Duración por tipo de ataque: DDoS y Ransomware tienden a tener duraciones más prolongadas.
 
●	Severidad según resultado: Los ataques exitosos muestran mayor severidad promedio.
 
●	Cantidad de ataques por mes: Distribución estable con ligeros picos estacionales.
 
Variable objetivo (target)
La variable que se desea predecir es month, que representa el mes en el que ocurrió el ataque cibernético.
 Este campo es de tipo categórico (valores de 1 a 12), y se utiliza para identificar patrones temporales en los ataques.
Variables predictoras
El conjunto de variables independientes se compone de atributos tanto categóricos como numéricos, que describen las características del ataque.
 Variables categóricas
●	attack_type: tipo de ataque (Phishing, DDoS, Ransomware, etc.)

●	target_system: sistema objetivo (API, Database, Cloud Service, etc.)

●	outcome: resultado del ataque (Success o Failure)

●	security_tools_used: herramientas de seguridad empleadas

●	user_role: rol del usuario afectado (Admin, Developer, User, etc.)

●	location: país o región donde ocurrió el incidente

●	industry: sector de la organización afectada

●	mitigation_method: método usado para mitigar el ataque

Estas variables aportan contexto sobre cómo, dónde y a quién afectó el ataque. Sin embargo, en este dataset muchas presentan una distribución balanceada y sin patrones dominantes, lo que explica por qué los gráficos de frecuencia muestran proporciones similares entre categorías.
   Variables numéricas
●	data_compromised_GB: cantidad de datos comprometidos (en GB)

●	attack_duration_min: duración total del ataque (en minutos)

●	attack_severity: severidad del ataque (escala 1–10)

●	response_time_min: tiempo de respuesta (en minutos)

Estas variables cuantitativas reflejan la intensidad y respuesta ante los incidentes.
 Su distribución es casi uniforme, sin concentraciones extremas, lo cual indica que el dataset fue simulado o generado con valores aleatorios balanceados por eso en los histogramas y boxplots no se observan diferencias significativas entre categorías o meses.
Transformaciones realizadas
Para preparar los datos antes del entrenamiento de los modelos, se aplicaron los siguientes pasos:
1.	Codificación de variables categóricas
 Se aplicó One-Hot Encoding, que transforma cada categoría en una columna binaria (0/1).
 Esto permite que los algoritmos numéricos (como SVM y Decision Tree) procesen información cualitativa sin perder significado.

2.	Normalización de variables numéricas
 Se utilizó MinMaxScaler, que reescala todos los valores al rango [0, 1].
 Esto evita que variables con rangos grandes (como duración o datos comprometidos) dominen el modelo frente a otras de menor magnitud.

3.	Eliminación de columnas irrelevantes
 Se eliminaron variables que no aportan valor predictivo o que podrían generar ruido:

•	attacker_ip

•	target_ip

•	timestamp

 Estas columnas funcionan como identificadores únicos, pero no contienen información útil para la predicción.

4.	Verificación de balance en la variable objetivo (month)
Se confirmó que los 12 meses tienen una cantidad similar de registros, lo que evita sesgo temporal en el entrenamiento.
Esto también explica por qué en los gráficos de distribución por mes no se aprecian picos o concentraciones marcadas.

Observaciones y análisis interpretativo
●	Sin valores faltantes:
Todos los registros contienen datos completos, lo cual facilita la limpieza y evita imputaciones artificiales.

●	Ausencia de outliers extremos:
Los boxplots muestran valores dentro de rangos razonables, sin casos atípicos significativos. Esto favorece la estabilidad del modelo.

●	Correlaciones bajas:
El mapa de calor mostró muy poca correlación entre variables numéricas.
Esto indica que cada variable aporta información distinta, pero también sugiere que no hay relaciones lineales fuertes que expliquen el mes del ataque.

●	Frecuencias similares entre categorías:
Los gráficos de barras de attack_type, industry o location muestran distribuciones bastante equilibradas.
Esto sugiere que el dataset es sintético o equilibrado artificialmente (es decir, no proviene de un escenario real con predominio de ciertos ataques o regiones).
Por tanto, los modelos pueden tener dificultades para encontrar patrones fuertes y podrían rendir con métricas cercanas al azar (como 0.49–0.50 en F1 o accuracy).
Los datos se encuentran limpios, balanceados y estandarizados, listos para el entrenamiento de modelos supervisados como SVM (Máquina de Vectores de Soporte) y Árbol de Decisión.
Sin embargo, debido a la baja variabilidad entre las categorías y la falta de correlaciones fuertes, se espera que los modelos no logren una alta capacidad predictiva, lo que se refleja en métricas de desempeño cercanas al 50%.
2. Implementación del modelo (SVM y Árbol de Decisión)
En esta etapa se implementaron tres modelos de clasificación supervisada utilizando Scikit-learn: Árbol de Decisión, SVM (Máquinas de Vectores de Soporte) y Random Forest.
 El objetivo fue comparar su rendimiento en un escenario de detección de patrones relacionados con ataques cibernéticos, utilizando un conjunto de datos con variables tanto numéricas como categóricas.
2.1. Preprocesamiento de los datos
Antes de entrenar los modelos, se definió un preprocesador con ColumnTransformer, encargado de:
●	Escalar las variables numéricas usando MinMaxScaler (para mantenerlas en el rango [0,1]).

●	Codificar las variables categóricas mediante OneHotEncoder, transformando los textos en variables binarias para su correcta interpretación por los modelos.

Las variables utilizadas fueron:
Variables categóricas:
 attack_type, target_system, security_tools_used, user_role, location, industry, mitigation_method, month
Variables numéricas:
 data_compromised_GB, attack_duration_min, attack_severity, response_time_min, hour
Para garantizar una comparación justa y reproducible, todos los modelos fueron implementados mediante un pipeline de Scikit-learn, que incluye:
1.	Preprocesamiento automático (codificación de variables categóricas y escalado de variables numéricas).

2.	Entrenamiento del modelo (clasificador SVM, Árbol de Decisión y Random Forest).

3.	Búsqueda de hiperparámetros mediante RandomizedSearchCV.
Soporte Vector Machine (SVM)
El modelo SVM lineal (LinearSVC) fue elegido por su eficiencia en datasets grandes y balanceadas.
La búsqueda aleatoria (RandomizedSearchCV) exploró combinaciones del parámetro de penalización C, del tipo de kernel y del parámetro gamma.
Los resultados óptimos fueron:
{'clf__kernel': 'linear', 'clf__gamma': 'scale', 'clf__C': 0.1}
El modelo con kernel lineal resultó ser el más estable, obteniendo un F1-Score promedio de 0.513 durante la validación cruzada. Esto sugiere que el modelo capta patrones lineales entre las variables predictoras y la variable objetivo.
Árbol de Decisión
El modelo de Árbol de Decisión (DecisionTreeClassifier) permite una interpretación más clara de las reglas de decisión, facilitando la comprensión de qué atributos del ataque contribuyen más a su clasificación.
Los hiperparámetros evaluados incluyeron la profundidad máxima (max_depth) y el tamaño mínimo de muestra por división (min_samples_split).
 Los valores óptimos fueron:
{'clf__min_samples_split': 5, 'clf__max_depth': 20}
El Árbol de Decisión obtuvo un F1-Score promedio de 0.507, lo que indica un desempeño ligeramente menor que el SVM pero con mejor interpretabilidad.

3. Implementación del Árbol de Decisión
El Árbol de Decisión se implementó con la clase DecisionTreeClassifier, usando un pipeline que incluye tanto el preprocesamiento como el modelo.
3.1. Ajuste de hiperparámetros
Se aplicó una búsqueda aleatoria (RandomizedSearchCV) con validación cruzada (k=3) para optimizar los siguientes hiperparámetros:
Hiperparámetro	Descripción	Valores probados
max_depth	Profundidad máxima del árbol	[5, 10, 20, None]
min_samples_split	Mínimo de muestras para dividir un nodo	[2, 5, 10]

El mejor conjunto de parámetros encontrado fue:
{'clf__max_depth': 20, 'clf__min_samples_split': 5}

3.2. Resultados del Árbol de Decisión
El modelo alcanzó una puntuación F1 promedio de 0.507, mostrando un rendimiento equilibrado entre precisión y exhaustividad (recall).
 Sin embargo, los valores fueron cercanos a 0.5, lo que sugiere que los datos no presentan patrones fácilmente separables o que las clases están balanceadas de forma equitativa.



4. Comparación de modelos y visualización de métricas
Modelo	Mejores hiperparámetros	F1 (macro)
Árbol de Decisión	min_samples_split=5, max_depth=20	0.5072
SVM (Lineal)	kernel='linear', C=0.1, gamma='scale'	0.5135 (Mejor)
Random Forest	n_estimators=50, min_samples_split=2, max_depth=10	0.5076
 
El mejor modelo fue el SVM lineal, porque obtuvo el mayor puntaje de F1 (0.5135), superando ligeramente a los otros dos modelos.
El F1-score mide un equilibrio entre precisión y recall, por lo que es ideal en casos con clases desbalanceadas o cuando nos interesa que el modelo no solo acierte, sino también detecte correctamente los casos de falla. 
Aunque la diferencia entre modelos es pequeña, el SVM con kernel lineal mostró mejor capacidad de generalización en los datos de validación cruzada. Además, el parámetro C=0.1 indica que el modelo fue más regularizado (menos sobreajuste), lo que suele mejorar el rendimiento en datos reales o ruidosos.
El modelo SVM (lineal) fue el que mejor equilibró la detección de incidentes (recall) y la precisión de sus predicciones, obteniendo el mejor puntaje F1 en la validación cruzada. Esto sugiere que su frontera de decisión lineal logra separar de manera más efectiva las condiciones que llevan a una falla en la infraestructura de TI.
Una vez optimizados los tres modelos, se compararon utilizando el conjunto de prueba (X_test, y_test) mediante las métricas:
●	Accuracy (Exactitud): proporción de aciertos totales.

●	Precision (Precisión): qué tan confiables son las predicciones positivas.

●	Recall (Exhaustividad): qué proporción de positivos reales fueron detectados.

●	F1-Score: balance entre precisión y recall.

4.1 Resultados comparativos
Modelo	Accuracy	Precision	Recall	F1-Score
Árbol de Decisión	0.501	0.502	0.502	0.499
SVM (Lineal)	0.497	0.497	0.497	0.495
Random Forest	0.498	0.498	0.498	0.496

●	Todos los modelos obtuvieron métricas similares, cercanas a 0.5, lo que sugiere que el conjunto de datos presenta una alta complejidad o bajo nivel de separabilidad entre clases.

●	El Árbol de Decisión fue el modelo con mejor rendimiento global (F1 = 0.499), mostrando un ligero equilibrio superior entre precisión y recall.

●	El SVM lineal tuvo un desempeño similar, pero con mayor tiempo de entrenamiento.

●	El Random Forest, aunque más robusto, no logró superar al árbol individual debido al tamaño de muestra y la posible redundancia de variables.

 
 4.2 Análisis de Resultados
Los tres modelos presentan rendimientos muy similares, con valores cercanos al 50% en todas las métricas.
 Este comportamiento sugiere que el conjunto de datos podría:
●	Ser difícil de separar linealmente, es decir, las clases no presentan patrones claros para diferenciarse.

●	Estar balanceado, con un número casi igual de ejemplos para cada clase, lo que provoca que los modelos tiendan a predecir ambas clases con igual probabilidad.

●	Requerir más variables relevantes o una mejor transformación de las características actuales.

A continuación, se detalla el rendimiento individual de cada modelo:
Árbol de Decisión
●	Precisión y recall equilibradas (0.50 en promedio).

●	El modelo probablemente no logró generar una estructura jerárquica significativa debido a la falta de patrones fuertes entre las variables predictoras y el resultado.

●	Su ventaja principal es la interpretabilidad, pero su desempeño fue limitado.

Conclusión parcial: el Árbol de Decisión es simple de interpretar, pero su capacidad predictiva no supera el azar (50%).
SVM (Máquinas de Vectores de Soporte)
●	Presenta valores similares a los del Árbol de Decisión, con una ligera variación en recall.

●	El kernel RBF se seleccionó como el mejor, lo cual indica que el modelo intentó generar una frontera no lineal.

●	Sin embargo, los resultados muestran que la separación de clases sigue siendo débil.

Conclusión parcial: el modelo SVM no logró una mejora significativa; probablemente las variables no sean adecuadas para una separación hiperdimensional efectiva.
 Random Forest
●	También mostró métricas en torno a 0.50, sin mejoras sustanciales respecto a los otros modelos.

●	A pesar de combinar múltiples árboles (lo que usualmente mejora la generalización), en este caso los árboles parecen haber aprendido patrones poco diferenciadores.

Conclusión parcial: aunque robusto y menos propenso al sobreajuste, el Random Forest no logró capturar una estructura predictiva clara en los datos.
 4.3 Interpretación General
El desempeño homogéneo de los tres modelos sugiere que:
●	Las características disponibles no son lo suficientemente informativas para predecir correctamente la variable objetivo.

●	Podría existir ruido o redundancia en las variables, afectando la capacidad de aprendizaje.

●	Sería recomendable realizar un análisis exploratorio adicional, aplicar técnicas de selección o ingeniería de características y explorar modelos basados en redes neuronales o boosting (como XGBoost o LightGBM).

Conclusión Final
Modelo más equilibrado	Árbol de Decisión
Mejor kernel SVM	RBF (γ=auto, C=10)
Recomendación futura	Mejorar variables predictoras, probar modelos de boosting
En resumen, los tres modelos lograron un rendimiento similar al azar (≈50%), lo cual evidencia que el conjunto de datos no presenta una estructura lineal o no lineal clara que permita discriminar entre las clases.
 Sin embargo, la metodología implementada (pipeline, preprocesamiento, validación cruzada, optimización y evaluación comparativa) es correcta y replicable para nuevos conjuntos de datos más informativos.
5. Reflexión técnica y conclusiones
5.1. Reflexión técnica
Este proyecto permitió comparar tres algoritmos representativos de clasificación supervisada:
●	Árboles de Decisión: ofrecen interpretabilidad y rapidez en entrenamiento.

●	SVM: son útiles para límites de decisión definidos, aunque su rendimiento depende del kernel y escalado.

●	Random Forest: mejora la estabilidad y generalización, pero requiere más recursos computacionales.

Durante la experimentación, se identificó que:
●	La correcta preparación y transformación de datos tiene tanto impacto como la selección del modelo.

●	La búsqueda de hiperparámetros mejora la eficiencia de cada modelo.

●	Métricas similares entre modelos indican que la información discriminante de las variables es limitada, sugiriendo explorar nuevas características o técnicas de selección de variables.

5.2. Conclusiones finales
●	El Árbol de Decisión fue el modelo con mejor equilibrio entre rendimiento y tiempo de cómputo.

●	Se recomienda aplicar técnicas adicionales como aumento de datos, ingeniería de características o balanceo de clases para mejorar la capacidad predictiva.

●	Este trabajo demuestra la importancia del pipeline como estructura estándar para el desarrollo reproducible de modelos de Machine Learning.

