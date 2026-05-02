# Documentación: Sistema de Recomendación de Alimentos (Modelo KNN)
**Proyecto:** CaloFit - Asistente Inteligente IA  
**Metodología:** CRISP-DM (Adaptada para Tesis)  
**Algoritmo:** K-Nearest Neighbors (Vecinos Más Cercanos)  

---

## 1. Fase: Comprensión del Negocio (Business Understanding)

### El Problema a Resolver (Objetivo de la Tesis)
Cuando a un usuario le faltan nutrientes específicos para cumplir su meta diaria (por ejemplo, le faltan 30g de proteína pero ya no puede comer carbohidratos), el Asistente IA debe recomendarle qué comer. El problema es que los Modelos de Lenguaje (como ChatGPT o Llama-3) no son buenos haciendo matemáticas exactas y suelen "alucinar" o inventar recetas que no cuadran con los números.

### La Solución y el Porqué
En lugar de dejarle las matemáticas a Llama-3, se construyó un "Motor de Recomendación" matemático. **¿Por qué?** Porque necesitamos precisión clínica. Este motor calcula el déficit exacto del usuario y busca en la base de datos oficial del Perú el alimento que encaje perfectamente como la pieza faltante de un rompecabezas. Así, aseguramos rigor científico.

---

## 2. Fase: Comprensión de los Datos (Data Understanding)

Para garantizar la viabilidad del proyecto en el contexto peruano, se utilizaron fuentes oficiales:
- **Fuente Principal:** Tabla de Composición de Alimentos del Instituto Nacional de Salud del Perú (INS/CENAN 2017).
- **Volumen Analizado:** 1,102 registros iniciales.
- **Variables Extraídas (Features):** Calorías, Proteínas, Carbohidratos y Grasas (por cada 100g de porción).

---

## 3. Fase: Preparación de los Datos (Data Preparation)

Para que la Inteligencia Artificial no cometa errores, la data pura tuvo que ser procesada:

1. **Limpieza de Inconsistencias:** Se eliminaron alimentos repetidos y aquellos con valores imposibles (como calorías negativas), resultando en un dataset final de **1,084 alimentos únicos**.
2. **Imputación de Nulos:** Los nutrientes vacíos en la base de datos se reemplazaron con `0.0g` para evitar que el código falle al sumar.
3. **Equilibrio Numérico (Feature Scaling):** **¿Por qué se hizo esto?** Las calorías suelen ser números grandes (ej. 350) y los nutrientes números pequeños (ej. 15g). Si no se "equilibran" (usando la técnica `StandardScaler`), el modelo solo le prestaría atención a las calorías e ignoraría las proteínas. Al escalarlos, obligamos al modelo a darle la misma importancia a todos los macronutrientes.

---

## 4. Fase: Modelado (Modeling)

### ¿Por qué K-Nearest Neighbors (KNN)?
Se eligió el algoritmo KNN (K-Vecinos Más Cercanos) porque su trabajo es agrupar elementos que se parecen geométricamente. Al darle al modelo un "Déficit Nutricional", este busca los "3 vecinos" (alimentos) más idénticos a ese requerimiento.

### La Clave Técnica: "Similitud Coseno"
En KNN se puede medir la distancia de varias formas, pero aquí se usó **Similitud Coseno**. **¿Por qué?** Porque la Similitud Coseno mide el *ángulo* (la proporción) en lugar de la distancia recta. Esto significa que si el usuario busca algo con "mucha proteína y cero grasa", el modelo buscará esa **proporción exacta** (como clara de huevo), sin importar si la porción es de 100g o 500g. Es la medida perfecta para nutrición.

---

## 5. Fase: Evaluación (Evaluation - Métricas y Pruebas Reales)

Para demostrar empíricamente que el modelo funciona y no "adivina", se ejecutaron 3 pruebas de estrés simulando déficits reales de usuarios. El modelo logró una **similitud matemática promedio superior al 95%** en todos los casos:

#### Prueba 1: Usuario en Definición Muscular (Busca "Pura Proteína")
* **El Déficit (Lo que se pidió):** 150 kcal | 30g Proteína | 0g Carbos | 5g Grasa
* **Resultados Obtenidos (Top 3):**
  1. **Pescado lorna, pulpa seca** *(Similitud: 99.7%)* → (150 kcal | 30.9g Pro | 0.0g Car | 2.0g Gra)
  2. **Hígado de res frito** *(Similitud: 99.5%)* → (174 kcal | 21.0g Pro | 8.0g Car | 6.4g Gra)
  3. **Pescado bonito seco** *(Similitud: 99.5%)* → (184 kcal | 32.3g Pro | 0.0g Car | 5.1g Gra)
* *Análisis:* El modelo identificó a la perfección que el usuario requería carne magra. El pescado lorna encajó al 99.7% con lo solicitado.

#### Prueba 2: Usuario Pre-Entrenamiento (Busca "Pura Energía / Carbos")
* **El Déficit (Lo que se pidió):** 350 kcal | 5g Proteína | 60g Carbos | 2g Grasa
* **Resultados Obtenidos (Top 3):**
  1. **Maíz gigante rojo** *(Similitud: 97.9%)* → (348 kcal | 4.4g Pro | 80.3g Car | 0.7g Gra)
  2. **Dulce de Leche** *(Similitud: 97.8%)* → (297 kcal | 6.0g Pro | 57.5g Car | 6.0g Gra)
  3. **Maíz chochoca** *(Similitud: 97.8%)* → (349 kcal | 5.2g Pro | 78.0g Car | 2.5g Gra)
* *Análisis:* Al pedir carbohidratos altos con nula grasa, el modelo priorizó fuentes de energía rápida (maíz) con una precisión calórica casi exacta (348 kcal vs 350 kcal pedidas).

#### Prueba 3: Comida Balanceada (Cantidades Iguales)
* **El Déficit (Lo que se pidió):** 400 kcal | 20g Proteína | 20g Carbos | 20g Grasa
* **Resultados Obtenidos (Top 1):**
  1. **Leche entera en polvo** *(Similitud: 96.1%)* → (484 kcal | 27.0g Pro | 36.1g Car | 26.1g Gra)
* *Análisis:* Un requerimiento complejo de empatar, pero el modelo logró encontrar un alimento base (leche entera) que mantiene las proporciones equilibradas en los 3 macronutrientes.

---

## 6. Fase: Despliegue e Integración (Deployment)

1. **Empaquetado (Exportación):** El modelo fue compilado en un archivo `.pkl` de tan solo **0.10 MB**.
2. **Integración Híbrida (El Gran Aporte de la Tesis):**
   * **El Cerebro Lógico (KNN):** Realiza la búsqueda matemática en milisegundos y devuelve las métricas exactas y los 3 alimentos peruanos idóneos.
   * **El Cerebro Empático (Llama-3):** Toma esos 3 alimentos y arma una respuesta conversacional natural.
   
**Conclusión:** Se demuestra que la combinación de Machine Learning Clásico (KNN) con IA Generativa (Llama-3) soluciona el problema de las "alucinaciones" de los chatbots, garantizando recomendaciones avaladas por el Ministerio de Salud del Perú.
