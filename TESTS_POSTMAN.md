# 🧪 PREGUNTAS DE TEST PARA POSTMAN - Parser de Respuestas IA

## 📋 **Endpoint:**
```
POST http://localhost:8000/api/asistente/consultar
```

## 🔑 **Headers:**
```
Authorization: Bearer TU_TOKEN_AQUI
Content-Type: application/json
```

---

## ✅ **TEST 1: Desayuno + Almuerzo + Ejercicios (Formato Completo)**

### **Body:**
```json
{
    "mensaje": "Hola CaloFit, necesito ayuda. Me gustaría que me recomiendes: 1) Un desayuno energético para empezar bien el día, 2) Un almuerzo para ganar masa muscular que sea diferente a lo que siempre como, y 3) Qué ejercicios debería hacer hoy en el gym considerando que peso 80kg. Recuerda que soy vegano y alérgico al maní.",
    "historial": []
}
```

### **Qué verificar:**
- ✅ `respuesta_estructurada.secciones` debe tener 3 elementos
- ✅ Sección 1: `tipo: "comida"`, `subtipo: "desayuno"`
- ✅ Sección 2: `tipo: "comida"`, `subtipo: "almuerzo"`
- ✅ Sección 3: `tipo: "ejercicio"`, `ejercicios: [...]` con al menos 3 ejercicios
- ✅ Cada comida debe tener `nombre`, `ingredientes` no nulos
- ✅ Si hay advertencia nutricional: `tiene_advertencia_nutricional: true`

---

## ✅ **TEST 2: Solo Cena (Formato Simple)**

### **Body:**
```json
{
    "mensaje": "Recomiéndame una cena saludable y vegana que me ayude a no excederme en calorías pero que tenga suficiente proteína.",
    "historial": []
}
```

### **Qué verificar:**
- ✅ `respuesta_estructurada.secciones` debe tener al menos 1 elemento
- ✅ Sección con `subtipo: "cena"` o tipo general
- ✅ Debe tener `ingredientes` con cantidades (250g, 100ml, etc.)
- ✅ Puede tener `preparacion` si la IA la incluye

---

## ✅ **TEST 3: Solo Ejercicios**

### **Body:**
```json
{
    "mensaje": "Hoy tengo poco tiempo pero quiero entrenar. ¿Qué rutina de ejercicios me recomiendas para hacer en 30 minutos que trabaje todo el cuerpo?",
    "historial": []
}
```

### **Qué verificar:**
- ✅ `respuesta_estructurada.secciones` debe tener 1 elemento tipo `"ejercicio"`
- ✅ `ejercicios` debe ser una lista con al menos 4 ejercicios
- ✅ Cada ejercicio debe incluir "series", "repeticiones" o "minutos"

---

## ✅ **TEST 4: Desayuno + Almuerzo + Cena (Día Completo)**

### **Body:**
```json
{
    "mensaje": "Necesito un plan de alimentación completo para hoy: desayuno, almuerzo y cena. Quiero ganar masa muscular, peso 80kg y soy vegano. Los platos deben ser peruanos.",
    "historial": []
}
```

### **Qué verificar:**
- ✅ `respuesta_estructurada.secciones` debe tener 3 elementos
- ✅ Cada uno con `subtipo` diferente: "desayuno", "almuerzo", "cena"
- ✅ Todos deben tener `nombre` del plato peruano
- ✅ Todos deben tener `ingredientes` con cantidades

---

## ✅ **TEST 5: Snack/Merienda**

### **Body:**
```json
{
    "mensaje": "Tengo hambre entre comidas. ¿Qué snack vegano y alto en proteínas me recomiendas que pueda llevar al trabajo?",
    "historial": []
}
```

### **Qué verificar:**
- ✅ Debe detectar `subtipo: "snack"` o `"merienda"`
- ✅ Debe tener `ingredientes` simples
- ✅ Idealmente con `preparacion` o indicaciones

---

## ✅ **TEST 6: Mensaje Motivacional (Sin Comida ni Ejercicio)**

### **Body:**
```json
{
    "mensaje": "Hoy me siento desmotivado. Siento que no estoy progresando con mi dieta. ¿Qué me dices?",
    "historial": []
}
```

### **Qué verificar:**
- ✅ `respuesta_estructurada.secciones` debe tener 1 elemento
- ✅ Tipo: `"general"`
- ✅ Debe tener `contenido` con mensaje motivacional

---

## ✅ **TEST 7: Con Historial (Memoria)**

### **Body:**
```json
{
    "mensaje": "¿Y qué opciones tengo para la cena?",
    "historial": [
        {
            "role": "user",
            "content": "Recomiéndame un desayuno vegano"
        },
        {
            "role": "assistant",
            "content": "Te recomiendo un Tacu Tacu vegano..."
        }
    ]
}
```

### **Qué verificar:**
- ✅ La IA debe recordar el contexto (desayuno ya dado)
- ✅ Debe recomendar solo cena
- ✅ Parser debe detectar `subtipo: "cena"`

---

## ✅ **TEST 8: Advertencia Nutricional (Validador Activo)**

### **Body:**
```json
{
    "mensaje": "Dame un desayuno con 200g de carne de soya y 200g de habas",
    "historial": []
}
```

### **Qué verificar:**
- ✅ `tiene_advertencia_nutricional: true`
- ✅ `advertencia_nutricional` debe contener "⚠️ **Nota Nutricional"
- ✅ Debe mencionar el cálculo real de proteínas (probablemente >100g)

---

## 📊 **Resultados Esperados Generales:**

### **Estructura Mínima:**
```json
{
  "respuesta_estructurada": {
    "respuesta_completa": "...",
    "secciones": [
      {
        "tipo": "comida" | "ejercicio" | "general",
        "subtipo": "desayuno" | "almuerzo" | "cena" | "snack",
        "nombre": "Nombre del plato",
        "ingredientes": "Lista de ingredientes...",
        "preparacion": "Instrucciones...",
        "justificacion": "Por qué es bueno para ti..."
      }
    ],
    "tiene_advertencia_nutricional": true | false,
    "advertencia_nutricional": "..." | null
  }
}
```

---

## 🚨 **Errores Comunes a Revisar:**

1. ❌ `ingredientes: null` → Debería extraer ingredientes alternativos
2. ❌ `ejercicios: null` → Verificar que detecta líneas con `*` 
3. ❌ `nombre: "Desayuno energético"` → Debería extraer nombre real del plato
4. ❌ Secciones vacías → Verificar que detecta al menos 1 sección
5. ❌ `preparacion: null` → Verificar si la IA incluye "Preparación" sin `**`

---

## 💡 **Tips para Testing:**

1. **Verifica los logs de Docker** para ver el debugging:
   ```bash
   docker logs calofit_backend -f
   ```
   Deberías ver:
   ```
   📊 DEBUG Sección 1 (desayuno): Contenido de 450 caracteres
     - Nombre: Tacu Tacu vegano
     - Ingredientes: ✅
     - Preparación: ✅
   ```

2. **Copia el JSON de respuesta** y formátalo en Postman con "Beautify"

3. **Busca `respuesta_estructurada.secciones[0].ingredientes`** para verificar que no sea `null`

4. **Prueba con diferentes formatos** para asegurar robustez

---

¡Buena suerte con el testing, Leonardo! 🇵🇪🚀💪
