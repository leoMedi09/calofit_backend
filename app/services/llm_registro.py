"""
llm_registro.py — Registro directo vía LLM (sin lookup de BD de alimentos).

Arquitectura nueva:
  mensaje → LLM extrae nombre + macros → INSERT directo → respuesta limpia

Reemplaza la lógica de 5 capas de asistente_registro_comida.py y el
asistente_registro_ejercicio.py para el path del chat conversacional.
Los modelos ML (KNN/RF) siguen intactos — progreso_calorias se sigue llenando.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────────

_IDENTIDAD = """Eres el Asistente CaloFit del gimnasio World Light Lambayeque — un profesional con doble especialización:
• Nutricionista Clínico certificado con dominio completo de la Tabla Peruana de Composición de Alimentos (INS/CENAN 2017), OpenFoodFacts y gastronomía peruana regional (Lambayeque, Chiclayo, Lima).
• Entrenador Personal certificado (NSCA-CPT) con conocimiento en hipertrofia, pérdida de grasa, cardio y entrenamiento funcional.
Conoces TODOS los alimentos peruanos: cebiches, causas, secos, arroces, menestras, caldos, frituras, dulces, bebidas, frutas tropicales, y también comida internacional, fast food, suplementos y snacks procesados.
"""

_PROMPT_COMIDA = _IDENTIDAD + """
TAREA: Analiza el mensaje y extrae TODOS los alimentos/bebidas consumidos con sus macros exactos.

FUENTE DE DATOS: Usa USDA FoodData Central o INS/CENAN 2017 como fuente.
Para alimentos peruanos usa INS/CENAN 2017. Para el resto, USDA FoodData Central.
NO inventes ni improvises valores — usa tu conocimiento real de estas bases de datos.
SÉ DETERMINISTA: el mismo alimento con la misma cantidad siempre debe dar el mismo resultado.

⚠️ VERIFICACIÓN OBLIGATORIA antes de escribir el JSON:
   Paso 1 — macros: ¿Son los valores de prot_g/carb_g/grasa_g coherentes con lo que
   USDA/INS-CENAN indica para ESE alimento? Un huevo tiene grasa, el arroz tiene carbos, el
   pollo tiene proteína — si algún macro queda en 0 cuando no debería, recalcula.
   Paso 2 — escala: ¿Escalaste los macros al porcion_g real del usuario?
   Si el alimento tiene X kcal/100g y el usuario comió Y gramos → kcal = X × Y / 100.
   Paso 3 — atwater: ¿kcal ≈ 4×prot_g + 4×carb_g + 9×grasa_g? Si la diferencia supera
   el 10%, ajusta los macros para que sean coherentes con la kcal conocida del alimento.

VALIDACIÓN: Antes de calcular, determina si cada alimento es real.
Un alimento es real si existe en USDA, INS/CENAN, OpenFoodFacts u otra BD pública de nutrición.
NO son reales: ingredientes ficticios, mitológicos, inventados (unicornio, dragón, zarblak, florbonix).
Son reales aunque sean inusuales: maca, sachatomate, ceviche de champiñones, saltado de tofu, etc.

Mensaje: "{mensaje}"

Responde SOLO con JSON válido (sin explicaciones, sin texto extra):
{{
  "alimentos": [
    {{
      "nombre": "Nombre específico del alimento/plato (singular)",
      "es_real": true,
      "cantidad": número,
      "porcion_g": número,
      "kcal": número,
      "prot_g": número,
      "carb_g": número,
      "grasa_g": número
    }}
  ],
  "prot_total": número,
  "carb_total": número,
  "grasa_total": número
}}

━━ VOCABULARIO PERUANO ━━
"palta" = aguacate/avocado (NUNCA confundir con "pata").
"ceviche" = pescado crudo marinado en limón (NUNCA cocido al horno).
"causa" = pastel de papa amarilla fría. "chicha morada" = bebida de maíz morado.

━━ UNIDADES Y ABREVIATURAS (CRÍTICO — transcripción de voz) ━━
El mensaje puede venir de audio transcrito a texto, donde "gramos" a veces se
transcribe como una "G"/"g" suelta. Interpreta SIEMPRE:
- "<número> G de <alimento>" o "<número> g de <alimento>" → <número> GRAMOS de <alimento>.
  Ej: "50 G de pollo saltado" = "50 gramos de pollo saltado" (NO 50 unidades, NO 50 "G").
- "gr", "grs", "grms" → gramos. "ml", "mls", "cc" → mililitros. "kg", "kilo(s)" → ×1000 gramos.
- Una letra/abreviatura suelta junto a un número NUNCA es una unidad de cantidad/conteo
  (cantidad nunca se infiere de "G", "g", "ml", "kg" — esos SIEMPRE son peso/volumen → porcion_g).

━━ REGLAS OBLIGATORIAS ━━
1. Si es_real = false → ese item NO se incluye en el JSON final (omitirlo).
2. Si TODOS son ficticios → {{"alimentos":[], "prot_total":0, "carb_total":0, "grasa_total":0}}
3. ⚠️ SIEMPRE incluye TODOS los alimentos/bebidas reales mencionados en el mensaje, SIN EXCEPCIÓN
   — incluso si el mensaje menciona 2, 3 o más alimentos distintos en la misma frase
   (separados por "y", "con", "más", "," o "además de"). NUNCA omitas un alimento
   mencionado explícitamente solo porque aparece junto a otro. Antes de responder,
   verifica que cada alimento/plato nombrado por el usuario tenga su propio ítem
   (o forme parte de un combo según la regla 9) — si falta alguno, agrégalo.
   ⚠️ CUENTA LAS MENCIONES, NO LOS NOMBRES ÚNICOS: si el mensaje describe varias comidas
   (Desayuno/Almuerzo/Cena/Merienda) y el MISMO alimento aparece mencionado en MÁS DE UNA
   sección, debe haber UN ÍTEM POR CADA MENCIÓN (el array "alimentos" puede tener el mismo
   "nombre" repetido más de una vez) — NUNCA fusiones menciones de secciones distintas en
   un solo ítem. Ejemplo concreto:
     Mensaje: "Desayuno: plátano sancochado con queso de dieta y un bizcocho.
               Almuerzo: ceviche con torta de choclo, camote.
               Cena: 3 bizcochos con queso de dieta"
     → "alimentos" debe tener 8 ítems: Plátano sancochado, Queso de dieta (desayuno),
        Bizcocho ×1 (desayuno), Ceviche, Torta de choclo, Camote, Bizcocho ×3 (cena),
        Queso de dieta (cena) — "Queso de dieta" aparece DOS VECES porque se menciona
        en DOS comidas distintas, cada una con su propia porción.
4. MÉTODO DE COCCIÓN cambia kcal: FRITO (absorbe aceite) ≠ COCIDO ≠ CRUDO.
5. prot_total = Σ prot_g. carb_total = Σ carb_g. grasa_total = Σ grasa_g.
6. CANTIDADES: "dos panes con pollo" → UN solo ítem {{nombre:"Pan con Pollo", cantidad:2, kcal: total×2}}. NUNCA separes en Pan ×2 + Pollo por separado — el "con" indica un combo, no ingredientes sueltos. kcal/macros son TOTALES ya multiplicados. nombre siempre en singular.
7. kcal debe ser consistente con P/C/G: verifica que ≈ 4×P + 4×C + 9×G. ⚠️ prot_g, carb_g, grasa_g y kcal son SIEMPRE para el "porcion_g" TOTAL de ese ítem, NUNCA valores de referencia por 100g sin escalar. Si "porcion_g" es menor a 100, los macros DEBEN ser proporcionalmente menores que los valores típicos por 100g de ese alimento (ej: si 100g de maní tienen ~26g de proteína, 28g de maní deben tener ~7g de proteína, NO 26g).
8. Si no se menciona cantidad explícita → cantidad:1.
9. COMBOS "X con Y" — UN solo ítem SOLO si "X con Y" es el NOMBRE de un plato/preparación
   reconocido como UNA unidad. Ejemplos:
   · CUALQUIER tipo de pan (pan francés, pan de molde, pan integral, ciabatta, baguette, etc.)
     con un topping/relleno (queso, palta/aguacate, mantequilla, mermelada, pollo, jamón, huevo, etc.)
     → SIEMPRE UN solo ítem "Pan [tipo] con [topping]". NUNCA separes el pan de su topping.
   · "tostada con mermelada/mantequilla/queso", "arroz con leche",
     "arroz con pollo/pato/pavo/res/chancho/mariscos", "papa con...", "tallarines con...",
     "puré con...", "menestra con..."
   → genera UN ítem único con todos sus componentes incluidos en sus macros.
   ⚠️ Si "X" y "Y" son DOS PLATOS/ALIMENTOS COMPLETOS E INDEPENDIENTES que simplemente se
   comieron juntos (ej: "pollo saltado con plátano sancochado", "arroz con pollo con una gaseosa",
   "lomo saltado con una ensalada"), trátalos como DOS ítems SEPARADOS, cada uno con sus
   propias macros — NO los fusiones en uno solo y NO descartes ninguno.
   Lo mismo aplica si están unidos por "y", "más" o ",": cada alimento/plato completo
   mencionado es su propio ítem, salvo que coincidan con un combo reconocido de esta regla.
10. ⚠️ "cantidad" es SOLO el número de PORCIONES/UNIDADES discretas (ej: "dos panes"→2, "tres galletas"→3). NUNCA pongas un valor en gramos/mililitros en "cantidad". Si el mensaje dice "150g de arroz", "200 gramos de pollo", "50 G de pollo" (= 50 gramos, ver sección de unidades), "300ml de jugo", "2 kg de pollo", "1.5 kilos de papa" → eso va en "porcion_g" (convierte kg a gramos: 1 kg = 1000g) y "cantidad" sigue siendo 1. kcal/macros deben corresponder al total de "porcion_g" (ej: 2 kg de pollo a la plancha = 2000g ≈ 3300 kcal, NO uses una porción estándar de 100-300g cuando el usuario especificó kilos). "cantidad" jamás debe ser mayor a 10.
11. PORCIONES POR DEFECTO (SOLO si el usuario NO especifica ninguna cantidad, unidad ni gramaje — ver regla 12 si sí especifica):
    - PLATO DE FONDO / almuerzo completo (arroz con algo, lomo saltado, seco, ají de gallina, tallarines, guisos, frituras con guarnición, causas rellenas): porción 350-450g → 600-1000 kcal. Proteínas magras (pollo, pescado, pavo) ≈600-750 kcal; proteínas grasas (pato, cerdo, res, chicharrón) ≈800-1000 kcal. NUNCA estimes un plato de fondo en menos de 600 kcal.
      ⚠️ PLATOS PERUANOS RECONOCIDOS — tratarlos SIEMPRE como UN solo ítem de plato de fondo
      (aplica regla 9 automáticamente): "arroz con pollo", "arroz con pato", "arroz con mariscos",
      "lomo saltado", "seco de pollo", "seco de res", "ají de gallina", "carapulcra", "causa rellena",
      "sudado de pescado", "chaufa de pollo", "estofado de pollo", "tallarines verdes con pollo".
      NUNCA descompongas estos platos en ingredientes separados — son un plato completo con 600-900 kcal.
    - BEBIDAS (jugo, limonada, gaseosa, chicha): 200-300 ml.
    - PAN/SÁNDWICH individual: 1 unidad ≈ 150-250 kcal base + relleno.
    - ENSALADA/ENTRADA (sin sopa ni caldo — ver regla SOPA abajo): 80-350 kcal.
    - DESAYUNO hogareño: usa porciones hogareñas normales (NO restaurante).
      Referencia por ítem: 1 huevo frito/revuelto (con aceite) ≈ 90 kcal · 1 rebanada pan de molde ≈ 75 kcal · vaso leche entera 200ml ≈ 130 kcal · taza avena cocida 200ml ≈ 150 kcal · queso fresco 30g ≈ 75 kcal.
      2-3 ítems de desayuno combinados suman 350-600 kcal. NUNCA reportes menos de 320 kcal si el usuario menciona 2 o más ítems de desayuno (huevos + pan, huevos + leche, etc.).
    - SOPA/CALDO/CREMA (sopa de pollo, caldo de gallina, crema de verduras, sopa de fideos):
      plato líquido — proteína típica 8-15g (NUNCA 30g+ en una sopa sola). Sin guarnición
      sólida mencionada aparte (arroz, papa extra, pan adicional), estima 120-250 kcal.
      LÍMITE ABSOLUTO: 300 kcal. Una sopa hogareña NUNCA supera 300 kcal por sí sola.
12. UNIDADES COTIDIANAS: si el usuario usa medidas caseras (rebanada/tajada/lonja/rodaja, trozo/pedazo, cucharada/cucharadita, taza, vaso, puñado, plato/porción), convierte a "porcion_g" REAL según ESE alimento específico y la cantidad mencionada — usa tu conocimiento nutricional para estimar el peso típico de esa medida para ese alimento (ej: una rebanada/rodaja de un tubérculo o pan es delgada, ~15-40g; una cucharada de una salsa/crema es ~15-20g; un vaso/taza de líquido es ~200-250ml; un puñado es ~25-40g). La unidad/cantidad EXPLÍCITA del usuario SIEMPRE tiene prioridad sobre las porciones por defecto de la regla 11 — NUNCA asumas un "plato completo" si el usuario especificó una porción menor (ej: "dos rebanadas de papa sancochada" es una porción pequeña de papa, NO un plato entero de papa a la huancaina).
13. MODIFICADORES DE TAMAÑO: "medio/media" → ~50% de la porción base (de la regla 11 o de una porción estándar de ese alimento); "un cuarto de" → ~25%; "porción/plato pequeño(a)" → ~60-70%; "porción/plato grande" → ~130-160%; "porción/plato mediano(a)" → 100% (base normal). Aplica ese porcentaje TANTO a "porcion_g" COMO a kcal/prot_g/carb_g/grasa_g de forma proporcional (ej: "medio vaso de leche" → ~120ml y la mitad de las kcal/macros de un vaso completo; "porción pequeña de causa de pollo" → ~60-70% del porcion_g y kcal de una causa de pollo normal, NO la porción completa).
14. CONSISTENCIA: para un mismo alimento y la misma porción, usa SIEMPRE los valores nutricionales
    estándar (USDA/INS-CENAN) de ese alimento — NO improvises valores nuevos cada vez. Si tienes
    duda entre varias preparaciones, usa la versión más común/estándar en Perú.
"""

_PROMPT_EJERCICIO = _IDENTIDAD + """
TAREA: Analiza el mensaje y extrae TODOS los ejercicios o actividades físicas mencionados.
Si hay varios ejercicios en el mismo mensaje, extráelos TODOS como ítems separados.

Mensaje: "{mensaje}"
Peso corporal del usuario: {peso_kg} kg

Responde SOLO con JSON array válido (aunque sea un solo ejercicio, siempre usa array):
[
  {{
    "encontrado": true,
    "ejercicio": "Nombre oficial del ejercicio",
    "grupo_muscular": "Pecho / Espalda / Piernas / Hombros / Brazos / Core / Cardio / Full Body",
    "series": número_o_null,
    "reps": número_o_null,
    "peso_kg": número_o_null,
    "duracion_min": número,
    "kcal_quemadas": número,
    "met": número_decimal,
    "intensidad": "Alta" | "Media" | "Baja"
  }}
]

Si NO hay ejercicio real: [{{"encontrado": false, "ejercicio": null, "kcal_quemadas": 0, "duracion_min": 0, "met": 0, "intensidad": "Baja"}}]

━━ TABLA MET PROFESIONAL ━━
CARDIO:
  Caminata lenta (4km/h)=3.0  Caminata rápida (6km/h)=4.5  Trote suave (8km/h)=8.3
  Correr moderado (10km/h)=10  Correr rápido (12km/h)=11.5  Ciclismo moderado=8.0
  Natación recreativa=6.0  Natación intensa=10.0  Bicicleta estática=7.0
  Saltar cuerda=12.0  Elíptica moderada=5.0  Remo máquina=7.0

FUERZA (gym):
  Press banca=5.0  Press militar=5.0  Press inclinado=5.0
  Sentadilla libre=6.0  Prensa de piernas=5.0  Peso muerto=6.0
  Dominadas/Pull-ups=8.0  Jalón al pecho=5.0  Remo con barra=6.0
  Curl de bíceps=3.5  Extensión tríceps=3.5  Elevaciones laterales=3.0
  Hip thrust=5.0  Zancadas/Lunges=5.5  Extensión cuádriceps=3.5
  Flexiones/Push-ups=8.0  Fondos en paralelas=8.0

FUNCIONAL / HIIT:
  Burpees=10.0  Box jumps=10.0  Kettlebell swings=12.0
  Battle ropes=10.0  HIIT circuito=9.0  CrossFit WOD=12.0
  TRX suspension=7.0  Plancha isométrica=4.0  Mountain climbers=8.0

DEPORTES:
  Fútbol=7.0  Básquet=8.0  Vóley=4.0  Tenis=7.5  Boxeo sparring=9.0

━━ REGLAS PROFESIONALES ━━
1. kcal = MET × {peso_kg} × 3.5 / 200 × duracion_min  (fórmula MET estándar)
2. Si NO hay ejercicio real: {{"encontrado": false, "ejercicio": null, "kcal_quemadas": 0, "duracion_min": 0, "met": 0, "intensidad": "Baja"}}
3. duracion_min: extrae del mensaje; si no se dice, estima según volumen:
   — 1 ejercicio 3×10: ~15 min  — rutina completa gym: ~45-60 min
   — trote sin duración: ~30 min  — cardio máquina sin tiempo: ~30 min
4. intensidad: Alta (MET≥8), Media (MET 5-7.9), Baja (MET<5)
5. Reconoce jerga peruana: "tiré" = realicé, "jalé" = hice fuerza, "metí" = hice
"""

_PROMPT_RECOMENDACION_COMIDA = """Eres un clasificador de platos. Responde SOLO con una lista. Nada más.

Dieta: {dieta}. Calorías disponibles: {restante} kcal. Restricciones: {condiciones}.

Escribe EXACTAMENTE 3 líneas en este formato (sin introducción, sin conclusión):
- NombrePlato1 (~XXX kcal)
- NombrePlato2 (~YYY kcal)
- NombrePlato3 (~ZZZ kcal)

Platos veganos peruanos válidos: causa de palmito, causa de champiñones, seco de lentejas, guiso de garbanzos, locro de zapallo, sopa de quinua, chaufa de tofu, ceviche de palmito, pepián de quinua, hummus con verduras.

PROHIBIDO: recetas, ingredientes, pasos, párrafos, texto antes o después de las 3 líneas.

Respuesta:"""
_PROMPT_RECOMENDACION_EJERCICIO = _IDENTIDAD + """
TAREA: El usuario pide sugerencias de EJERCICIO. Responde SOLO como entrenador personal.

Perfil:
- Nombre: {nombre}  |  Objetivo: {objetivo}  |  Condiciones médicas: {condiciones}

Mensaje del usuario: "{mensaje}"

TABLA DE EJERCICIOS POR GRUPO MUSCULAR:
- pecho/pectoral: Press Banca, Press Inclinado, Aperturas con Mancuernas, Fondos en Paralelas
- espalda/dorsal: Jalón al Pecho, Remo con Barra, Dominadas, Remo en Polea
- piernas: Sentadilla Libre, Prensa de Piernas, Peso Muerto Rumano, Extensión de Cuádriceps
- hombros/deltoides: Press Militar, Elevaciones Laterales, Face Pull, Pájaros
- bíceps/brazos: Curl con Barra, Curl con Mancuernas, Curl Martillo, Curl Concentrado
- tríceps: Extensión en Polea, Press Francés, Fondos, Patada de Tríceps
- abdomen/core: Plancha, Crunch, Elevación de Piernas, Russian Twist
- cardio: Trote, Bicicleta Estática, Elíptica, Saltar Cuerda, Burpees
- full body/general: Sentadilla, Peso Muerto, Dominadas, Burpees, Press Banca

RESPONDE en texto natural, 2-3 oraciones. Sin listas, sin tags, sin markdown.
Identifica el grupo muscular y sugiere 2-3 ejercicios CORRECTOS con series/reps.
Ejemplo: "Para trabajar el pecho te recomiendo Press Banca 3×10, Aperturas 3×12 y Fondos 3×8."
No empieces con "Hola" si el usuario no saludó. Tono motivador y directo.
PROHIBIDO: recetas de comida, mencionar kcal de alimentos.
PROHIBIDO terminar con pregunta.
"""

# Alias para compatibilidad (selecciona prompt según tipo de recomendación)
_PROMPT_RECOMENDACION = _PROMPT_RECOMENDACION_COMIDA

_PROMPT_CHAT = _IDENTIDAD + """
TAREA: Responde al mensaje del usuario de forma conversacional.

Perfil del usuario:
- Nombre: {nombre}
- {consumido}/{meta} kcal consumidas ({pct}%)  |  {quemado} kcal quemadas hoy
- Dieta: {dieta}  |  Condiciones: {condiciones}  |  Objetivo: {objetivo}

Conversación reciente:
{historial}

Mensaje actual: "{mensaje}"

REGLAS DE RESPUESTA:
⛔ REGLAS ABSOLUTAS (se aplican SIEMPRE, sin excepción):
  1. PROHIBIDO cualquier markdown: **negrita**, *cursiva*, # títulos. Solo texto plano.
  2. PROHIBIDO empezar con frases de relleno: "Leonardo, me alegra...", "Qué buena pregunta...", "Es un placer...". Empieza directo al tema.
  3. PROHIBIDO terminar con pregunta: "¿Quieres saber más?", "¿Te gustaría...?". Termina con punto.

⛔ ADAPTACIÓN DE DIETA (CRÍTICO):
  Si Dieta = "Vegano" o "Vegetariano" → PROHIBIDO ingredientes animales en recetas.
    Para platos con carne/pescado: adapta AUTOMÁTICAMENTE al sustituto vegetal SIN que el usuario lo pida.
    Ceviche vegano → usa palmito o champiñones. Lomo saltado vegano → usa tofu o setas.
    Siempre MENCIONA que es la versión vegana: "Versión vegana: en lugar de pescado, usa palmito..."
  Si Dieta = Normal, Mediterránea, Cetogénica, Diabético u OTRA → PROHIBIDO mencionar
    versiones veganas, sustitutos vegetales ni alternativas veganas en la receta.
    Usa los ingredientes originales del plato sin ofrecer variantes no solicitadas.
  Si Condiciones incluye Diabetes → evita azúcar, miel, carbos refinados en la receta.

RECETAS ('cómo se hace X', 'receta de X'):
  Empieza DIRECTAMENTE: "Ingredientes: ..."
  Formato: Ingredientes (4-6 items) → Preparación (4-5 pasos numerados).
  NO intro, NO cierre, NO markdown.

TÉCNICA DE EJERCICIO ('cómo hacer X', 'técnica de X'):
  Empieza con "1." directamente. 3-4 pasos: posición → movimiento → consejo.

PREGUNTAS SIMPLES: máximo 2-3 oraciones directas.
- CONGRUENCIA DE TONO: No empieces con "¡Claro que sí!" si la respuesta es negativa. Sé directo.

- PREGUNTAS DE EJERCICIO FÍSICO ('puedo trotar', 'puedo nadar', 'puedo hacer ejercicio'):
  ⚠️ IGNORA el balance calórico. NO menciones kcal consumidas ni la meta diaria.
  Responde SOLO sobre el ejercicio: si se puede, cómo y un consejo práctico.
  ✓ "Claro, el trote en casa es excelente cardio. Hazlo 20-30 min a ritmo cómodo."
  ✗ PROHIBIDO: "Con 1607 kcal ya consumiste..." ← no tiene relación con la pregunta.

- PREGUNTAS SOBRE SI UN ALIMENTO ES BUENO/SANO ('es bueno X', 'puedo comer X todos los días'):
  ⚠️ Responde la pregunta nutricional directamente. NO hables del balance del día.
  ✓ "es bueno comer palta todos los días" → "Sí, la palta es muy saludable — grasas monoinsaturadas, vitamina E y fibra. 1 unidad diaria es ideal."
  ✗ PROHIBIDO: "Con X kcal ya superaste tu meta..." ← responde la pregunta, no el balance.

- Consulta de calorías ('cuántas kcal tiene X', 'cuánto engorda X'):
  Porciones estándar: palta=240kcal/unidad · plátano=107kcal · huevo=85kcal · arroz=260kcal/plato.
  "palta" = aguacate/avocado. NUNCA confundir con "pata".
  Vegano pregunta por animal → responde NO directamente.

- Recetas peruanas: Causa=PAPA AMARILLA. Ceviche=PESCADO CRUDO. Lomo saltado=RES.
  Vegano: adapta con tofu/palmito manteniendo la base.
- Usa el historial para dar continuidad a la conversación.
- PROHIBIDO terminar con pregunta a menos que el usuario pida consejo explícito.

⛔ PERSONA GRAMATICAL (CRÍTICO):
  Las comidas/ejercicios del historial son del USUARIO, no tuyos. Refiérete a ellas SIEMPRE
  en SEGUNDA PERSONA ("almorzaste", "cenaste", "registraste", "comiste").
  PROHIBIDO usar primera persona para acciones del usuario ("Almorcé", "Cené", "Hice").
  ✓ "Almorzaste causa ferreñafana y cenaste un cebiche de caballa."
  ✗ "Almorcé causa ferreñafana y cenaste un cebiche de caballa." ← mezcla de personas, incorrecto.

⛔ BALANCE VS META (CRÍTICO):
  Si consumido > meta → el usuario YA EXCEDIÓ su meta. Dilo de forma directa y sin contradicciones
  ("ya superaste tu meta por X kcal, ten cuidado"). NUNCA digas que "está cumpliendo su objetivo"
  si el consumo es mayor a la meta — son afirmaciones contradictorias.
  Si consumido <= meta → puedes decir cuánto le queda disponible.
"""


# ── Funciones principales ─────────────────────────────────────────────────────

async def registrar_comida_llm(
    mensaje: str,
    perfil,
    plan_hoy: dict,
    db: Session,
    ia_engine,
    historial: list = None,
) -> dict:
    """Registra comida con macros estimados por LLM. Sin lookup de BD."""
    # ── Capa 0: buscar en caché de macros (platos recomendados previamente) ──
    # Si el usuario está registrando un plato que el asistente recomendó en esta
    # sesión, se usan los macros exactos cacheados → consistencia perfecta.
    cached = _buscar_en_cache(mensaje)
    if cached:
        kcal  = round(float(cached.get("kcal", 0)), 1)
        prot  = round(float(cached.get("prot_g", 0)), 1)
        carb  = round(float(cached.get("carb_g", 0)), 1)
        grasa = round(float(cached.get("grasa_g", 0)), 1)
        nombre_cached = cached.get("nombre", mensaje)
        if kcal > 0 or prot > 0 or carb > 0 or grasa > 0:
            logger.info("[Registro] Usando macros cacheados para '%s': %s kcal", nombre_cached, kcal)
            # Simular el dict que retornaría el LLM
            datos = {
                "alimentos": [{"nombre": nombre_cached, "es_real": True,
                                "kcal": kcal, "prot_g": prot, "carb_g": carb, "grasa_g": grasa}],
                "prot_total": prot, "carb_total": carb, "grasa_total": grasa,
                "kcal_total": kcal,  # preservar kcal del caché para evitar recálculo
            }
            # Saltar al bloque de guardado directamente
            goto_save = True
        else:
            datos = None
            goto_save = False
    else:
        datos = None
        goto_save = False

    # ── Capa 1: estimación LLM (si no hay caché) ──────────────────────────────
    if not goto_save:
        prompt = _PROMPT_COMIDA.format(mensaje=mensaje)
        # Siempre usar 900 tokens — mensajes con 3 comidas y cantidades específicas
        # necesitan espacio para el JSON completo (5-9 items = ~700-800 tokens)
        raw = await ia_engine._llamar_groq(prompt, max_tokens=900, temp=0.0)
        datos = _parse_json(raw)

    # Guard temprano: si datos es None (JSON truncado o inválido) → pedir aclaración
    if not datos:
        return {
            "success": False,
            "tipo_detectado": "no_identificado",
            "mensaje": f"No pude procesar todos los alimentos, {perfil.first_name}. ¿Puedes repetirlo dividido por comida? Ej: 'en el desayuno comí X'",
        }

    # Filtrar alimentos no reales (es_real: false) antes de validar
    if datos.get("alimentos"):
        datos["alimentos"] = [
            a for a in datos["alimentos"]
            if a.get("es_real", True) is not False
        ]

    # Validar que hay alimentos con macros
    _items = datos.get("alimentos", [])

    # ── Modificadores de tamaño ("medio", "porción pequeña/grande") ──────────
    # El LLM tiende a ignorar estos modificadores y devolver la porción
    # estándar. Se corrige escalando porcion_g/kcal/macros del único ítem
    # detectado (no aplica a mensajes con varios alimentos para no escalar
    # ítems que no llevan el modificador).
    _msg_low_porcion = mensaje.lower() if mensaje else ""
    _factor_porcion = None
    if re.search(r'\bmedi[oa]\b|\bmitad\b', _msg_low_porcion):
        _factor_porcion = 0.5
    elif re.search(r'\bun cuarto\b|\bcuarta parte\b|\b1/4\b', _msg_low_porcion):
        _factor_porcion = 0.25
    elif re.search(r'porci[oó]n (chica|pequeñ[ao])|plato (chico|pequeñ[oa])', _msg_low_porcion):
        _factor_porcion = 0.65
    elif re.search(r'porci[oó]n (grande|extra)|plato grande|doble porci[oó]n', _msg_low_porcion):
        _factor_porcion = 1.4

    # Aplica el modificador cuando hay 1 ítem (caso normal) o exactamente 2 ítems
    # y el modificador está en la primera mitad del mensaje — indica que aplica
    # al combo completo (ej: "un cuarto de pan francés con palta" → pan+palta son 1 porción)
    _modifier_early = (
        re.search(r'\b(medi[oa]|mitad|un cuarto|cuarta parte|porci[oó]n (chica|pequeñ[ao]|grande)|plato (chico|pequeñ[ao]|grande)|doble porci[oó]n)\b',
                  _msg_low_porcion[:max(1, len(_msg_low_porcion) // 2)])
        if _msg_low_porcion else None
    )
    if _factor_porcion and (len(_items) == 1 or (len(_items) == 2 and _modifier_early)):
        for _it in _items:
            for _campo in ("porcion_g", "kcal", "prot_g", "carb_g", "grasa_g"):
                if _it.get(_campo) is not None:
                    _it[_campo] = round(float(_it[_campo]) * _factor_porcion, 1)
        for _campo_total in ("prot_total", "carb_total", "grasa_total", "kcal_total"):
            if datos.get(_campo_total) is not None:
                datos[_campo_total] = round(float(datos[_campo_total]) * _factor_porcion, 1)
    _prot_items  = sum(float(a.get("prot_g",  0) or 0) for a in _items)
    _carb_items  = sum(float(a.get("carb_g",  0) or 0) for a in _items)
    _grasa_items = sum(float(a.get("grasa_g", 0) or 0) for a in _items)
    # No exigir macros > 0: alimentos/bebidas reales con 0 kcal (café negro, agua,
    # té sin azúcar, gaseosa zero) son válidos y deben registrarse igual.
    if not _items:
        return {
            "success": False,
            "tipo_detectado": "no_identificado",
            "mensaje": f"No identifiqué ningún alimento, {perfil.first_name}. ¿Qué comiste exactamente?",
        }

    # Fuente de verdad: SUMA de los macros POR ÍTEM (no los totales que devuelve el LLM
    # aparte, que a veces no coinciden con la suma real de sus propios ítems).
    # Esto garantiza que kcal == Σ kcal de cada fila insertada en comida_registros,
    # para que el Balance (suma de comidas) coincida con el total mostrado en el chat.
    if _prot_items > 0 or _carb_items > 0 or _grasa_items > 0:
        prot  = round(_prot_items, 1)
        carb  = round(_carb_items, 1)
        grasa = round(_grasa_items, 1)
        kcal  = round(4 * prot + 4 * carb + 9 * grasa, 1)
    else:
        prot  = round(float(datos.get("prot_total", 0)), 1)
        carb  = round(float(datos.get("carb_total", 0)), 1)
        grasa = round(float(datos.get("grasa_total", 0)), 1)
        kcal_desde_macros = round(4 * prot + 4 * carb + 9 * grasa, 1)
        kcal_llm = round(float(datos.get("kcal_total", 0)), 1)
        # Preferir macros si dan algún valor positivo; fallback a kcal_llm solo si macros=0
        kcal = kcal_desde_macros if kcal_desde_macros > 0 else kcal_llm

    # Tope de sanidad: cantidades absurdas (ej. "50 kg de arroz") generan totales
    # de macros irreales. Si el total supera el tope, escalar proporcionalmente
    # a un máximo razonable y avisar al usuario.
    _KCAL_MAX_RAZONABLE = 5000
    _factor_cap = 1.0
    advertencia_cantidad = None
    _factor_momento = 1.0
    advertencia_momento = None
    if kcal > _KCAL_MAX_RAZONABLE:
        _factor_cap = _KCAL_MAX_RAZONABLE / kcal
        kcal  = round(kcal * _factor_cap, 1)
        prot  = round(prot * _factor_cap, 1)
        carb  = round(carb * _factor_cap, 1)
        grasa = round(grasa * _factor_cap, 1)
        advertencia_cantidad = (
            f"⚠️ La cantidad indicada parece excesiva — registré un máximo razonable "
            f"de {round(kcal)} kcal. Si en verdad comiste esa cantidad, regístralo en "
            f"porciones separadas a lo largo del día."
        )

    # Cap por momento del día — evita que el LLM infle porciones de desayuno/cena/merienda
    _msg_low_momento = mensaje.lower() if mensaje else ""
    _momento_registro = None
    if any(k in _msg_low_momento for k in ("desayuno", "desayuné", "desayune")):
        _momento_registro = "DESAYUNO"
    elif any(k in _msg_low_momento for k in ("merienda", "snack")):
        _momento_registro = "MERIENDA"
    elif any(k in _msg_low_momento for k in ("cena", "cené", "cene")):
        _momento_registro = "CENA"
    _KCAL_CAP_MOMENTO_REG = {"DESAYUNO": 700, "MERIENDA": 400, "CENA": 750}
    _cap_momento = _KCAL_CAP_MOMENTO_REG.get(_momento_registro)
    if _cap_momento and kcal > _cap_momento and not advertencia_cantidad:
        _factor_momento = _cap_momento / kcal
        kcal   = round(kcal   * _factor_momento, 1)
        prot   = round(prot   * _factor_momento, 1)
        carb   = round(carb   * _factor_momento, 1)
        grasa  = round(grasa  * _factor_momento, 1)
        advertencia_momento = (
            f"⚠️ Los macros parecían elevados para un {_momento_registro.lower()} hogareño "
            f"— ajustado a {round(kcal)} kcal."
        )
        logger.info("[Registro] Cap momento %s aplicado → %.0f kcal", _momento_registro, kcal)

    # Cap específico para sopas/caldos — el LLM tiende a inflar sopas a 400+ kcal
    # cuando la realidad de una sopa hogareña sin guarnición extra es 120-250 kcal
    _SOPA_KW = ("sopa ", "caldo ", "crema de ", "sopa de ", " sopa", "caldito")
    _is_sopa = any(k in _msg_low_momento for k in _SOPA_KW)
    # Si el usuario menciona guarnición sólida explícita junto a la sopa, NO aplicar cap
    _SOPA_LADOS = ("con arroz", "con papa", "con pan", "con fideo", "con yuca",
                   "con camote", "con choclo", "y arroz", "y papa", "y pan")
    _tiene_lado_solido = any(s in _msg_low_momento for s in _SOPA_LADOS)
    _KCAL_CAP_SOPA = 300
    _factor_sopa = 1.0
    if _is_sopa and not _tiene_lado_solido and kcal > _KCAL_CAP_SOPA and not advertencia_cantidad and not advertencia_momento:
        _factor_sopa = _KCAL_CAP_SOPA / kcal
        kcal   = round(kcal   * _factor_sopa, 1)
        prot   = round(prot   * _factor_sopa, 1)
        carb   = round(carb   * _factor_sopa, 1)
        grasa  = round(grasa  * _factor_sopa, 1)
        advertencia_momento = "⚠️ Sopa estimada como plato líquido hogareño — ajustado a rango normal (sin guarnición sólida extra mencionada)."
        logger.info("[Registro] Cap sopa aplicado → %.0f kcal", kcal)

    alimentos_raw = datos.get("alimentos", [])
    # Construir nombres con multiplicador ×N para mostrar en chat y balance
    def _nombre_con_cantidad(a: dict) -> str:
        n = a.get("nombre", "")
        try:
            q = int(float(a.get("cantidad", 1) or 1))
        except (TypeError, ValueError):
            q = 1
        return f"{n} ×{q}" if q > 1 else n
    nombres = [_nombre_con_cantidad(a) for a in alimentos_raw if a.get("nombre")]

    # Actualizar progreso_calorias (totales del día)
    from app.core.utils import get_peru_date
    hoy = get_peru_date()
    prog = _get_or_create_progreso(db, perfil.id, hoy, plan_hoy)
    prog.calorias_consumidas      = int((prog.calorias_consumidas or 0) + kcal)
    prog.proteinas_consumidas     = round((prog.proteinas_consumidas or 0) + prot, 1)
    prog.carbohidratos_consumidos = round((prog.carbohidratos_consumidos or 0) + carb, 1)
    prog.grasas_consumidas        = round((prog.grasas_consumidas or 0) + grasa, 1)

    # Insertar en comida_registros (fuente del Balance screen)
    # Cuando cantidad > 1, insertar N entradas individuales con macros/N cada una.
    # Así Flutter agrupa por nombre y muestra la viñeta ×N con el diálogo de borrado
    # que ya permite elegir cuántas porciones eliminar (stepper −/+).
    from app.models.comida_registro import ComidaRegistro
    n_items = max(1, len(alimentos_raw))
    for item in alimentos_raw:
        nombre_item = item.get("nombre", nombres[0] if nombres else "Alimento")
        try:
            cantidad_item = int(float(item.get("cantidad", 1) or 1))
        except (TypeError, ValueError):
            cantidad_item = 1
        # Tope de seguridad: "cantidad" es el número de porciones discretas
        # (ej. "dos panes"). Si el LLM confunde gramos con cantidad
        # (ej. "150g de arroz" → cantidad:150), nunca debe insertar más de
        # 10 filas por ítem.
        cantidad_item = max(1, min(cantidad_item, 10))
        # Macros por porción unitaria (aplicando el mismo tope de sanidad que los totales)
        _factor_total = _factor_cap * _factor_momento * _factor_sopa
        p_item = round(float(item.get("prot_g", prot / n_items)) * _factor_total / cantidad_item, 1)
        c_item = round(float(item.get("carb_g", carb / n_items)) * _factor_total / cantidad_item, 1)
        g_item = round(float(item.get("grasa_g", grasa / n_items)) * _factor_total / cantidad_item, 1)
        # kcal SIEMPRE derivado de P/C/G de este ítem (4-4-9) — nunca el "kcal" crudo
        # del LLM, que puede no ser consistente con sus propios macros. Así Σ kcal de
        # las filas de este ítem == 4*prot_item + 4*carb_item + 9*grasa_item del total.
        k_item = round(4 * p_item + 4 * c_item + 9 * g_item, 1)
        for _ in range(cantidad_item):
            registro = ComidaRegistro(
                client_id=perfil.id,
                fecha=hoy,
                nombre_alimento=nombre_item,
                kcal=k_item,
                proteina_g=p_item,
                carbohidratos_g=c_item,
                grasas_g=g_item,
                tipo_resolucion="llm_estimado",
                confianza=0.85,
                texto_original=mensaje[:490],
            )
            db.add(registro)

    db.commit()

    nombres_str = " + ".join(nombres[:3])
    if len(nombres) > 3:
        nombres_str += f" y {len(nombres)-3} más"
    # Lista completa para que Flutter pueda mostrar todos los ítems
    nombres_completos = nombres

    meta      = float(plan_hoy.get("calorias_dia", 2000))
    consumido = float(prog.calorias_consumidas)
    
    # Calorías quemadas: fuente autoritativa = workout_logs
    from sqlalchemy import text as _sql_wl
    _dialect = getattr(getattr(db, "bind", None), "dialect", None)
    _dname = getattr(_dialect, "name", "") or ""
    if _dname == "postgresql":
        quemado = float(db.execute(_sql_wl(
            "SELECT COALESCE(SUM(calorias_quemadas), 0) FROM workout_logs "
            "WHERE client_id = :cid "
            "  AND (created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Lima')::date = :hoy"
        ), {"cid": perfil.id, "hoy": hoy}).scalar() or 0)
    else:
        quemado = float(db.execute(_sql_wl(
            "SELECT COALESCE(SUM(calorias_quemadas), 0) FROM workout_logs "
            "WHERE client_id = :cid AND date(created_at) = :hoy"
        ), {"cid": perfil.id, "hoy": hoy}).scalar() or 0)

    restante  = max(0.0, meta - consumido + quemado)  # igual que la UI: suma quemadas

    # Detectar conflicto dietético y generar alerta suave
    alerta_dieta = _detectar_conflicto_dieta(nombres, getattr(perfil, "diet_type", None))

    return {
        "success": True,
        "tipo_detectado": "nutricion",
        "alimentos": nombres,
        "datos": {
            "nombre": nombres_str,
            "alimentos_lista": nombres_completos,
            "calorias": kcal,
            "proteinas_g": prot,
            "carbohidratos_g": carb,
            "grasas_g": grasa,
            "alerta_dieta": alerta_dieta,
            "advertencia_cantidad": advertencia_cantidad,
        },
        "balance_actualizado": {
            "consumido": round(consumido, 1),
            "meta":      round(meta, 1),
            "restante":  round(restante, 1),
            "quemado":   round(quemado, 1),
        },
        "mensaje": (
            f"✅ Registré: {nombres_str} — {round(kcal)} kcal. "
            f"Llevas {round(consumido)} de {round(meta)} kcal hoy."
            + (f"\n\n{advertencia_cantidad}" if advertencia_cantidad else "")
            + (f"\n\n{advertencia_momento}" if advertencia_momento else "")
        ),
        "alerta_dieta": alerta_dieta,
    }


async def registrar_ejercicio_llm(
    mensaje: str,
    perfil,
    db: Session,
    ia_engine,
) -> dict:
    """Registra UNO O VARIOS ejercicios del mensaje con kcal por LLM."""
    peso_kg = float(getattr(perfil, "weight", 70) or 70)
    prompt = _PROMPT_EJERCICIO.format(mensaje=mensaje, peso_kg=peso_kg)
    # Más tokens para mensajes con múltiples ejercicios
    _max = 600 if len(mensaje.split()) > 15 else 300
    raw = await ia_engine._llamar_groq(prompt, max_tokens=_max, temp=0.0)
    resultado = _parse_json(raw)

    # Normalizar: acepta tanto lista como objeto único
    if isinstance(resultado, dict):
        ejercicios_raw = [resultado]
    elif isinstance(resultado, list):
        ejercicios_raw = resultado
    else:
        ejercicios_raw = []

    # Filtrar ejercicios válidos
    ejercicios_raw = [e for e in ejercicios_raw
                      if e.get("encontrado", True) and e.get("ejercicio")]

    if not ejercicios_raw:
        return {
            "success": False,
            "tipo_detectado": "no_identificado",
            "mensaje": f"No identifiqué ningún ejercicio, {perfil.first_name}. ¿Qué entrenamiento hiciste?",
        }

    from app.core.utils import get_peru_date
    from app.models.historial import ProgresoCalorias
    hoy = get_peru_date()

    kcal_total = 0.0
    ejercicios_guardados = []

    for datos in ejercicios_raw:
        nombre   = datos["ejercicio"]
        duracion = float(datos.get("duracion_min", 0) or 0)
        series   = datos.get("series")
        reps     = datos.get("reps")
        peso_ej  = datos.get("peso_kg")
        met      = float(datos.get("met", 5.0) or 5.0)
        intensidad = datos.get("intensidad") or ("Alta" if met >= 8 else ("Media" if met >= 5 else "Baja"))
        kcal_formula = round(met * peso_kg * 3.5 / 200 * duracion, 1)
        kcal_llm     = round(float(datos.get("kcal_quemadas", 0) or 0), 1)
        kcal = kcal_formula if kcal_llm > kcal_formula * 2.5 or kcal_llm < kcal_formula * 0.3 else kcal_llm

        try:
            db.execute(text("""
                INSERT INTO workout_logs
                    (client_id, ejercicio, series, reps,
                     peso_kg, calorias_quemadas, intensity, session_duration_min, created_at)
                VALUES
                    (:cid, :nombre, :series, :reps,
                     :peso, :kcal, :intensity, :sdm, NOW())
            """), {
                "cid":      perfil.id,
                "nombre":   nombre,
                "series":   int(series) if series else 0,
                "reps":     int(reps)   if reps   else 0,
                "peso":     float(peso_ej) if peso_ej else None,
                "kcal":     round(kcal, 1),
                "intensity": intensidad,
                "sdm":      round(duracion, 1),
            })
            kcal_total += kcal
            detalle = f"{series}×{reps}" if series and reps else f"{int(duracion)}min"
            if peso_ej and series and reps:
                detalle += f" @{peso_ej}kg"
            ejercicios_guardados.append({"nombre": nombre, "kcal": kcal, "detalle": detalle})
        except Exception as e:
            logger.error("[llm_registro] Error guardando ejercicio %s: %s", nombre, e)

    if not ejercicios_guardados:
        db.rollback()
        return {"success": False, "tipo_detectado": "error", "mensaje": "Error al guardar el ejercicio."}

    # Actualizar calorias_quemadas en progreso
    prog = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == perfil.id,
        ProgresoCalorias.fecha == hoy,
    ).first()
    if prog:
        prog.calorias_quemadas = round((prog.calorias_quemadas or 0) + kcal_total, 1)
    db.commit()

    quemado_total = round(float(prog.calorias_quemadas if prog else kcal_total), 1)

    # Construir mensaje de confirmación
    if len(ejercicios_guardados) == 1:
        ex = ejercicios_guardados[0]
        msg = f"✅ Registré: {ex['nombre']} | {ex['detalle']} — {round(ex['kcal'])} kcal quemadas."
        nombre_pill = ex['nombre']
        detalle_pill = ex['detalle']
    else:
        nombres = " + ".join(e["nombre"] for e in ejercicios_guardados)
        msg = f"✅ Registré {len(ejercicios_guardados)} ejercicios ({nombres}) — {round(kcal_total)} kcal totales."
        nombre_pill = nombres
        detalle_pill = f"{len(ejercicios_guardados)} ejercicios"

    return {
        "success": True,
        "tipo_detectado": "ejercicio",
        "datos": {
            "nombre":       nombre_pill,
            "kcal_quemadas": kcal_total,
            "duracion_min": sum(float(e.get("duracion_min", 0) or 0) for e in ejercicios_raw),
            "series":       ejercicios_guardados[0]["detalle"] if len(ejercicios_guardados) == 1 else None,
            "ejercicios":   ejercicios_guardados,
        },
        "balance_actualizado": {"quemado": quemado_total},
        "mensaje": msg,
    }


def _persistir_historial_recomendaciones(db, perfil, momento: str, platos: list) -> None:
    """Guarda los platos recomendados (con macros reales) en HistorialRecomendacion
    para que las próximas 48h los excluya el candidato KNN y el LLM no los repita.
    plato_id queda en NULL: son platos generados por LLM, no del catálogo."""
    if db is None or not platos:
        return
    try:
        from app.models.historial_recomendacion import HistorialRecomendacion

        for nombre, kcal, prot, carb, gras in platos:
            db.add(HistorialRecomendacion(
                client_id=perfil.id,
                plato_id=None,
                nombre_plato=nombre,
                calorias=kcal,
                proteinas_g=prot,
                carbohidratos_g=carb,
                grasas_g=gras,
                momento_dia=momento.lower() if momento else None,
                fue_consumido=False,
            ))
        db.commit()
    except Exception as e:
        logger.warning("[Reco] No se pudo persistir HistorialRecomendacion: %s", e)
        db.rollback()


async def respuesta_recomendacion_llm(
    mensaje: str,
    perfil,
    consumido: float,
    meta: float,
    quemado: float,
    ia_engine,
    modo: str = "comida",
    historial: list = None,
    db: Session = None,
    plan_macros: dict = None,
    consumido_macros: dict = None,
) -> str:
    """Genera recomendación vía LLM. Para comida: cachea los macros exactos de
    cada plato recomendado → cuando el usuario lo registre, se usarán los mismos
    valores (consistencia perfecta recomendación ↔ registro)."""
    restante = max(0.0, meta - consumido + quemado)  # igual que la UI: suma quemadas
    objetivo = getattr(perfil, "goal", "mantener peso") or "mantener peso"
    dieta    = getattr(perfil, "diet_type", "Normal") or "Normal"
    condiciones = ", ".join(getattr(perfil, "medical_conditions", None) or []) or "ninguna"

    if modo == "ejercicio":
        prompt = _PROMPT_RECOMENDACION_EJERCICIO.format(
            nombre=perfil.first_name,
            objetivo=objetivo,
            condiciones=condiciones,
            mensaje=mensaje,
        )
        return await ia_engine._llamar_groq(prompt, max_tokens=300, temp=0.7)

    # ── Recomendación de COMIDA: generada por LLM con contexto real ──────────────
    import re as _re_reco
    from app.core.utils import get_peru_now as _get_peru_now_reco

    # 0. Detectar momento del día PRIMERO — necesario para filtrar candidatos KNN
    #    antes de pasarlos al LLM (evita sugerir ingredientes inapropiados por horario)
    # Orden de prioridad: CENA → MERIENDA → ALMUERZO → DESAYUNO
    # ALMUERZO va antes que DESAYUNO para que "ya entrené en la mañana, necesito almorzar"
    # matchee "almorzar" (ALMUERZO) antes de matchear "mañana" (DESAYUNO).
    # "tarde" solo en MERIENDA — "snack en la tarde" → MERIENDA, no ALMUERZO.
    _MOMENTO_KEYWORDS_RECO = {
        "CENA":      ["cenar", "cena", "noche", "nocturno"],
        "MERIENDA":  ["merienda", "snack", "media tarde", "media mañana", "antojo", "tarde"],
        "ALMUERZO":  ["almorzar", "almuerzo", "mediodía", "mediodia"],
        "DESAYUNO":  ["desayunar", "desayuno", "mañana", "madrugada"],
    }
    _msg_low_reco = mensaje.lower() if mensaje else ""
    momento_reco = None
    for _m_key, _kws in _MOMENTO_KEYWORDS_RECO.items():
        if any(kw in _msg_low_reco for kw in _kws):
            momento_reco = _m_key
            break
    if not momento_reco:
        _hora = _get_peru_now_reco().hour
        if 5 <= _hora < 10:
            momento_reco = "DESAYUNO"
        elif 10 <= _hora < 15:
            momento_reco = "ALMUERZO"
        elif 15 <= _hora < 19:
            momento_reco = "MERIENDA"
        else:
            momento_reco = "CENA"

    # 1. KNN — candidatos del catálogo INS/CENAN por similitud coseno con el déficit real.
    _candidatos_knn: list = []
    _excluidos_48h: list[str] = []
    if db is not None:
        try:
            from datetime import datetime, timedelta

            from app.models.historial_recomendacion import HistorialRecomendacion
            from app.services.ml_service import ml_recomendador

            _desde_48h = datetime.utcnow() - timedelta(hours=48)
            _excluidos_48h = [
                row[0] for row in db.query(HistorialRecomendacion.nombre_plato)
                .filter(
                    HistorialRecomendacion.client_id == perfil.id,
                    HistorialRecomendacion.created_at >= _desde_48h,
                )
                .all()
                if row[0]
            ]

            _plan_macros = plan_macros or {}
            _cons_macros = consumido_macros or {}
            _prot_falt = max(0.0, (_plan_macros.get("proteinas_g") or 0) - (_cons_macros.get("proteinas") or 0))
            _carb_falt = max(0.0, (_plan_macros.get("carbohidratos_g") or 0) - (_cons_macros.get("carbohidratos") or 0))
            _gras_falt = max(0.0, (_plan_macros.get("grasas_g") or 0) - (_cons_macros.get("grasas") or 0))

            _candidatos_knn = ml_recomendador.obtener_recomendaciones(
                restante, _prot_falt, _carb_falt, _gras_falt,
                n_recomendaciones=3,
                excluir_nombres=_excluidos_48h,
                contexto=mensaje,
            )
        except Exception as e:
            logger.warning("[Reco] KNN candidatos no disponibles: %s", e)

    # 1.5. Evaluador LLM — valida culturalmente los candidatos KNN para el momento del día.
    #      El prompt varía por momento para rechazar ingredientes que generarían platos
    #      inapropiados aunque el ingrediente en sí no esté prohibido (ej: "Lisa" es un
    #      pez válido, pero con él el LLM haría un sudado → plato de almuerzo, no desayuno).
    #      Si ninguno calza → _top_knn = None → los 3 platos serán full LLM.
    _EVAL_CONTEXTO_MOMENTO = {
        "DESAYUNO": (
            "Para el DESAYUNO en Perú solo son válidos ingredientes con los que se "
            "preparan desayunos reales: lácteos (leche, yogur, queso fresco), cereales "
            "(avena, kiwicha, quinua), frutas, pan, huevos, palta, plátano, granola. "
            "RECHAZA SIEMPRE: pescado, carne de res, pollo, cerdo, mariscos — con "
            "ellos se hacen platos de almuerzo o cena, nunca desayunos. "
            "RECHAZA también legumbres/menestras (frejol, lenteja, arveja seca, soja, "
            "garbanzo) y especias/hierbas solas (anís, orégano, comino, canela)."
        ),
        "MERIENDA": (
            "Para la MERIENDA (snack, 80-300 kcal) solo son válidos: frutas, frutos "
            "secos, lácteos, avena, pan integral, palta, maní, granola, yogur. "
            "RECHAZA: pescado, carne, pollo, mariscos, legumbres, arroz — con ellos "
            "se preparan platos completos de almuerzo o cena, no meriendas. "
            "RECHAZA especias/hierbas solas (anís, orégano, comino) que no anclan plato."
        ),
        "CENA": (
            "Para la CENA (platos ligeros, máx 520 kcal) son válidos: pescado magro, "
            "pollo a la plancha, huevos, vegetales, menestras ligeras, sopas. "
            "RECHAZA: ingredientes que solo generan platos muy calóricos (chicharrón, "
            "panceta) y especias/hierbas solas (anís, orégano, comino, canela) que "
            "no pueden ser el ingrediente principal de un plato."
        ),
        "ALMUERZO": (
            "Para el ALMUERZO son válidos casi todos los ingredientes de la gastronomía "
            "peruana: carnes, pescados, aves, mariscos, legumbres, cereales, tubérculos. "
            "RECHAZA únicamente especias/hierbas como ingrediente PRINCIPAL (anís, "
            "orégano, comino, canela, culantro seco) que no pueden anclar un plato completo."
        ),
    }
    _top_knn = None
    if _candidatos_knn:
        _nombres_knn = [c["alimento"] for c in _candidatos_knn]
        _ctx_eval = _EVAL_CONTEXTO_MOMENTO.get(momento_reco, "")
        _prompt_eval = (
            f"Eres nutricionista peruano. {_ctx_eval} "
            f"Lista del catálogo MINSA/INS: {', '.join(_nombres_knn)}. "
            f"¿Cuáles de estos alimentos se usarían habitualmente para preparar "
            f"un plato de {momento_reco} en Lambayeque? "
            f"Responde SOLO los nombres apropiados separados por coma. "
            f"Si ninguno encaja responde exactamente: ninguno"
        )
        try:
            _resp_eval = await ia_engine._llamar_groq(_prompt_eval, max_tokens=60, temp=0.0)
            if _resp_eval and "ninguno" not in _resp_eval.lower():
                _resp_low = _resp_eval.lower()
                for c in _candidatos_knn:
                    _palabras = [p for p in c["alimento"].lower().split() if len(p) > 3]
                    if any(p in _resp_low for p in _palabras):
                        _top_knn = c
                        logger.info("[KNN Eval] Aprobado: %s para %s", c["alimento"], momento_reco)
                        break
            else:
                logger.info("[KNN Eval] Ningún candidato válido para %s → 3 platos full LLM", momento_reco)
        except Exception as e:
            logger.warning("[KNN Eval] Evaluador falló: %s — sin ancla KNN", e)

    # 2. Restricciones por momento del día
    _RESTRICCIONES_MOMENTO_RECO = {
        "DESAYUNO": (
            "Rango: 250-450 kcal. Primera comida del día, rápida y simple. "
            "Típico peruano: avena con leche, pan con palta o queso, huevos revueltos, "
            "yogur con granola, fruta con cereal, quinua con leche. "
            "⛔ PROHIBIDO: sopas, chupes, caldos, pescado, carnes, cebiches, causas, arroces guisados."
        ),
        "ALMUERZO": (
            "Rango: 550-850 kcal — porción real de adulto, MÍNIMO 550 kcal. "
            "Plato de fondo conocido: seco de pollo, arroz con pollo, ceviche, sudado, lomo saltado. "
            "⛔ No repitas el mismo tipo de proteína en los 3 platos."
        ),
        "CENA": (
            "Rango: 200-520 kcal. Plato ligero para la noche: "
            "sopa, ensalada con proteína, pescado a la plancha, menestra. "
            "⛔ Evita frituras y guisos pesados — esos son de almuerzo."
        ),
        "MERIENDA": (
            "Rango: 80-280 kcal. Refrigerio rápido sin cocción elaborada. "
            "Válido: fruta, pan con palta, yogur, frutos secos, huevo sancochado. "
            "⛔ PROHIBIDO: pescado, mariscos, carnes, causas, cebiches, arroces, guisos."
        ),
    }
    restricciones_momento_reco = _RESTRICCIONES_MOMENTO_RECO.get(momento_reco, "")

    # 3. Detectar preferencia de ingrediente específico en el mensaje
    _ing_match = _re_reco.search(
        r'(?:con|de|que\s+tenga|a\s+base\s+de)\s+([a-záéíóúüñ][a-záéíóúüñ\s]{1,25})',
        _msg_low_reco,
    )
    pref_ingrediente_reco = ""
    if _ing_match:
        _ing_detectado = _ing_match.group(1).strip().rstrip('.,?')
        _PALABRAS_IGNORAR = {"hoy", "comer", "ti", "mi", "algo", "uno", "plato", "poco"}
        if _ing_detectado not in _PALABRAS_IGNORAR and len(_ing_detectado) > 2:
            pref_ingrediente_reco = (
                f"El usuario pidió algo con: **{_ing_detectado}**. "
                f"Al menos 1 de los 3 platos debe incluir ese ingrediente."
            )

    # 3.5. Detectar objetivo de PROTEÍNA en el mensaje
    _objetivo_proteina_match = _re_reco.search(
        r'prote[ií]na|prote[ií]co|masa muscular|ganar m[uú]sculo|aumentar m[uú]sculo|volumen muscular',
        _msg_low_reco,
    )
    objetivo_proteina_reco = (
        "OBJETIVO PROTEÍNA: el usuario quiere AUMENTAR SU CONSUMO DE PROTEÍNA. "
        "Los 3 platos DEBEN tener una fuente proteica principal y abundante "
        "(pollo, pescado, res, huevo, menestras, quinua, lácteos) — mínimo ~20g de proteína cada uno. "
        "PROHIBIDO proponer ensaladas o guarniciones sin proteína significativa "
        "(ej: ensalada de solo lechuga/tomate/papa, pachamanca solo de verduras)."
    ) if _objetivo_proteina_match else ""

    # 3.6. Detectar objetivo de masa muscular/volumen → override calórico
    _masa_muscular_match = _re_reco.search(
        r'masa muscular|ganar m[uú]sculo|aumentar m[uú]sculo|volumen muscular|'
        r'bulking|ganar peso|subir de peso|aumentar peso',
        _msg_low_reco,
    )
    # Si el restante es muy bajo pero el objetivo es ganar músculo, mostrar mínimo 500 kcal
    # para que el LLM no recomiende snacks ridículos (el LLM usa el valor como referencia, no límite duro)
    _restante_display = max(restante, 500.0) if _masa_muscular_match else restante
    _masa_muscular_txt = (
        "OBJETIVO MASA MUSCULAR: para ganar masa muscular se requiere un aporte calórico ALTO. "
        "Propón 3 platos completos de 400-700 kcal cada uno con ALTA proteína (≥25g por plato). "
        "Una ingesta calórica ligeramente superior al mantenimiento diario es CORRECTA y deseable "
        "para este objetivo — NO limites los platos al déficit restante del día. "
        "Usa fuentes de proteína magra: pollo a la plancha, pescado, res magra, huevos, "
        "menestras con quinua. Incluye carbohidratos de calidad (arroz, papa, quinua) como "
        "fuente de energía para el entrenamiento."
    ) if _masa_muscular_match else ""

    # 4. Restricción de dieta + condiciones médicas → reglas concretas para el LLM
    _condiciones_list_reco = getattr(perfil, "medical_conditions", None) or []
    _condiciones_str_reco = " ".join(_condiciones_list_reco).lower()
    es_vegano_reco = (
        "vegano" in dieta.lower() or "vegetariano" in dieta.lower()
        or "vegano" in _condiciones_str_reco or "vegetariano" in _condiciones_str_reco
    )
    restriccion_dieta_reco = (
        "VEGANO/VEGETARIANO: PROHIBIDO carnes, pollo, pescado, mariscos, lácteos animales. "
        "Solo plantas, legumbres, granos, frutas, tofu, soja, hongos."
    ) if es_vegano_reco else ""

    # Condiciones médicas → micro-llamada Groq que traduce cualquier condición
    # a restricciones dietéticas concretas. Sin hardcoding: funciona para Diabetes,
    # Hipertensión, Lactosa, Gota, Enfermedad Renal, Asma o cualquier condición futura.
    _condiciones_medicas_txt = ""
    if condiciones and condiciones.lower() != "ninguna":
        try:
            _prompt_med = (
                f"Eres nutricionista clínico. El paciente tiene: {condiciones}.\n"
                f"Lista en máximo 5 líneas las restricciones dietéticas CONCRETAS "
                f"para estas condiciones. Sé ESPECÍFICO con cada alimento individual "
                f"(leche, yogur, queso, crema, miel, azúcar, etc.) — menciona explícitamente "
                f"si se debe evitar o si existe una versión permitida (ej. deslactosada, sin azúcar).\n"
                f"Formato estricto — solo esto, sin explicaciones:\n"
                f"• [condición]: evitar [lista exacta], permitido solo [versiones seguras]\n"
                f"Responde SOLO las líneas con •. Nada más."
            )
            _restricciones_raw = await ia_engine._llamar_groq(
                _prompt_med, max_tokens=200, temp=0.0
            )
            if _restricciones_raw and _restricciones_raw.strip():
                _condiciones_medicas_txt = (
                    f"⛔ RESTRICCIONES MÉDICAS OBLIGATORIAS — aplica en los 3 platos:\n"
                    f"{_restricciones_raw.strip()}\n"
                    f"⚠️ Aplica cada restricción SOLO si el plato normalmente lleva ese ingrediente. "
                    f"No añadas lácteos, azúcares ni sustitutos a platos que no los necesitan "
                    f"(ej. no pongas leche en una menestra o sopa de verduras).\n\n"
                )
        except Exception as _e_med:
            logger.warning("[Reco] No se pudo generar restricciones médicas: %s", _e_med)

    # 5. Combinar platos ya recomendados: historial de la conversación actual
    #    (corto plazo) + HistorialRecomendacion de las últimas 48h (persistente,
    #    real, vía BD) para evitar repetición entre sesiones/días.
    _ya_vistos: list[str] = []
    if historial:
        _RE_BULLET_HIST = _re_reco.compile(r'-\s*([^\(]+)\s*\(~?\d+\s*kcal\)', _re_reco.IGNORECASE)
        for _hm in (historial or [])[-10:]:
            _ya_vistos += _RE_BULLET_HIST.findall(_hm.get("content", ""))
    _ya_vistos += _excluidos_48h

    _ya_sugeridos_txt = ""
    if _ya_vistos:
        _vistos_unicos = list(dict.fromkeys(v.strip() for v in _ya_vistos if v and v.strip()))
        _ya_sugeridos_txt = (
            f"PLATOS YA RECOMENDADOS (NO repetir): {', '.join(_vistos_unicos[:8])}.\n\n"
        )

    # 5.5. Estructura híbrida KNN + LLM:
    #      Plato 1 → ingrediente ancla del KNN (filtrado por momento), LLM crea nombre natural.
    #      Platos 2 y 3 → LLM libre, guiado solo por las restricciones del momento.
    _knn_candidatos_txt = ""
    if _top_knn:
        _alim_knn = _top_knn["alimento"]
        _kcal_knn = _top_knn["calorias_100g"]
        _knn_candidatos_txt = (
            f"GUÍA NUTRICIONAL (modelo KNN): el alimento '{_alim_knn}' "
            f"(~{_kcal_knn:.0f} kcal/100g) tiene el perfil nutricional más afín al déficit actual. "
            f"Si existe un plato CONOCIDO y COMÚN en Perú que lo lleve, úsalo para el Plato 1. "
            f"Si no hay un plato conocido que lo incluya para este momento, "
            f"ignora la guía y elige un plato libre igualmente conocido.\n\n"
        )

    # 6. Referencia de platos del día a día por momento — ejemplos de ESTILO, no lista cerrada.
    #    El LLM puede adaptar según condiciones médicas y KNN, pero dentro de este universo.
    _PLATOS_REFERENCIA = {
        "DESAYUNO": (
            "avena con leche, quinua con leche, pan con palta, pan con queso, "
            "huevos revueltos, huevos sancochados, tostada con mermelada, "
            "yogur con granola, fruta con cereal, mazamorra de maíz"
        ),
        "ALMUERZO": (
            "seco de pollo, seco de res, arroz con pollo, lomo saltado, ají de gallina, "
            "ceviche de pescado, sudado de pescado, carapulcra, chicharrón de pollo, "
            "menestra con arroz, tallarines verdes, arroz con mariscos, causa rellena, "
            "chaufa de pollo, estofado de pollo, sopa a la minuta"
        ),
        "CENA": (
            "sopa de pollo, caldo de gallina, sopa de fideos, sopa de lentejas, "
            "pollo a la plancha con ensalada, pescado a la plancha, tortilla de verduras, "
            "arroz con huevo, menestra sencilla, crema de zapallo, sopa de quinua"
        ),
        "MERIENDA": (
            "fruta sola, yogur con granola, pan con palta, galletas con queso, "
            "puñado de frutos secos, huevo sancochado, vaso de leche, "
            "avena preparada, mazamorra de maíz pequeña"
        ),
    }
    _ref_platos = _PLATOS_REFERENCIA.get(momento_reco, "")

    # 7. Prompt al LLM — condiciones médicas al final (recency bias: LLM las lee último)
    _prompt_reco_comida = (
        f"Eres nutricionista del Gimnasio World Light Lambayeque.\n"
        f"Propón EXACTAMENTE 3 platos para {perfil.first_name} — "
        f"recetas peruanas reales y conocidas del día a día.\n\n"
        f"PERFIL:\n"
        f"- Objetivo: {objetivo}\n"
        f"- Momento: {momento_reco}\n"
        f"- Calorías disponibles hoy: {round(_restante_display)} kcal\n\n"
        + (f"{restriccion_dieta_reco}\n\n" if restriccion_dieta_reco else "")
        + f"PARA EL {momento_reco}:\n{restricciones_momento_reco}\n\n"
        + f"PLATOS — escoge entre recetas conocidas del día a día peruano, como: {_ref_platos}. "
        + f"Puedes sugerir variantes o platos similares con nombre real que cualquier peruano reconoce. "
        + f"Si sugieres pescado, usa especies de Lambayeque (Caballa, Lisa, Mero, Tollo).\n"
        + f"⛔ SEMÁNTICA: Caballa, Lisa, Mero, Tollo son PESCADOS — nunca son 'mariscos'. "
        + f"No escribas 'Mariscos de Caballa' ni 'Mariscos de Lisa' — son categorías distintas. "
        + f"Di 'Arroz con Caballa' O 'Arroz con Mariscos', nunca ambos combinados.\n\n"
        + (f"{_masa_muscular_txt}\n\n" if _masa_muscular_txt else "")
        + (f"{objetivo_proteina_reco}\n\n" if objetivo_proteina_reco else "")
        + (f"PREFERENCIA: {pref_ingrediente_reco}\n\n" if pref_ingrediente_reco else "")
        + _ya_sugeridos_txt
        + _knn_candidatos_txt
        + _condiciones_medicas_txt  # ← justo antes del formato: máxima prioridad LLM
        + "FORMATO — exactamente 3 líneas:\n"
        "- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
        "- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
        "- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n\n"
        "P/C/G coherentes con kcal (4×P + 4×C + 9×G ≈ kcal). "
        "⛔ SOLO las 3 líneas. Sin recetas, sin texto extra."
    )

    respuesta_llm_reco = await ia_engine._llamar_groq(
        _prompt_reco_comida, max_tokens=180, temp=0.5
    )

    # Guard: si el LLM ignoró el formato y devolvió receta, reintentar con prompt mínimo
    _RECIPE_MARKERS = ("ingredientes:", "preparación:", "preparacion:", "pasos:", "instrucciones:")
    if any(m in (respuesta_llm_reco or "").lower() for m in _RECIPE_MARKERS):
        logger.warning("[Reco] LLM devolvió receta en vez de bullets — reintentando")
        _prompt_retry = (
            f"Lista 3 opciones de {momento_reco.lower()} peruanas "
            f"({round(restante)} kcal disponibles). SOLO este formato exacto:\n"
            f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
            f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
            f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
            f"Sin frases extra, sin ingredientes, sin pasos."
        )
        respuesta_llm_reco = await ia_engine._llamar_groq(_prompt_retry, max_tokens=120, temp=0.2)

    # 6. Parsear bullets del LLM y cachear macros reales (no hardcodeados)
    # Filtro de colas temporales: "Huevos con Espinacas y Mañana" → "Huevos con Espinacas"
    # Colas temporales al final del nombre
    _RE_COLA_TEMPORAL = _re_reco.compile(
        r'\s+(?:y\s+|con\s+|para\s+)?'
        r'(?:mañana|hoy|tarde|noche|esta\s+mañana|esta\s+noche|esta\s+tarde|hoy\s+día)\s*$',
        _re_reco.IGNORECASE,
    )
    # Palabras de contexto que el LLM inserta en cualquier posición del nombre
    _RE_CONTEXTO_MEDIO = _re_reco.compile(
        r'\s+(?:con|de|y|al)\s+(?:entrenamiento|ejercicio|workout|post[\s\-]?entrenamiento)\b',
        _re_reco.IGNORECASE,
    )

    def _limpiar_nombre(n: str) -> str:
        n = _re_reco.sub(r'^[\s\-•*\d.\)]+|[\s*]+$', '', n).strip()
        n = _RE_COLA_TEMPORAL.sub('', n).strip()
        n = _RE_CONTEXTO_MEDIO.sub('', n).strip()
        return n

    # Intento 1: el LLM incluyó kcal + P/C/G en el mismo bullet.
    _RE_BULLET_MACROS = _re_reco.compile(
        r'([^()\n]{3,80}?)\s*\(~?(\d+(?:\.\d+)?)\s*kcal[,;]?\s*'
        r'P\s*:?\s*(\d+(?:\.\d+)?)\s*g[,;]?\s*'
        r'C\s*:?\s*(\d+(?:\.\d+)?)\s*g[,;]?\s*'
        r'G\s*:?\s*(\d+(?:\.\d+)?)\s*g\)',
        _re_reco.IGNORECASE
    )
    _platos_con_macros = _RE_BULLET_MACROS.findall(respuesta_llm_reco or "")

    # Límites duros por momento — cap y floor post-procesados por si el LLM ignora los rangos
    _KCAL_CAP_MOMENTO   = {"CENA": 520, "ALMUERZO": 850, "MERIENDA": 280, "DESAYUNO": 450}
    _KCAL_FLOOR_MOMENTO = {"ALMUERZO": 550, "MERIENDA": 80, "DESAYUNO": 200, "CENA": 120}
    _kcal_cap   = _KCAL_CAP_MOMENTO.get(momento_reco, 850)
    _kcal_floor = _KCAL_FLOOR_MOMENTO.get(momento_reco, 0)

    # Post-procesado de lácteos: si el usuario tiene intolerancia a lactosa, asegurar
    # que yogur/leche/queso en los nombres lleven el calificador "deslactosado/a".
    # No hardcodea la condición: detecta "lactosa" como subcadena de condiciones.
    _tiene_intolerancia_lactosa = "lactosa" in condiciones.lower()
    _LACTEOS_REGEX = _re_reco.compile(
        r'\b(yogur|leche|queso|crema de leche|mantequilla)\b(?!\s+deslact)',
        _re_reco.IGNORECASE,
    )

    def _aplicar_deslactosado(nombre: str) -> str:
        if not _tiene_intolerancia_lactosa:
            return nombre
        return _LACTEOS_REGEX.sub(
            lambda m: m.group(0) + " deslactosado" if m.group(1).lower() in ("yogur", "queso", "mantequilla")
            else m.group(0) + " deslactosada",
            nombre,
        )

    if _platos_con_macros:
        _platos_limpios = []
        for _nombre_p, _kcal_p, _p_p, _c_p, _g_p in _platos_con_macros[:3]:
            _nombre_p = _limpiar_nombre(_nombre_p)
            _nombre_p = _aplicar_deslactosado(_nombre_p)
            p_f, c_f, g_f = float(_p_p), float(_c_p), float(_g_p)
            k_f = round(4 * p_f + 4 * c_f + 9 * g_f, 1) or float(_kcal_p)
            # Cap duro: si el LLM ignoró el límite superior, escalar macros proporcionalmente
            if k_f > _kcal_cap:
                _factor = _kcal_cap / k_f
                p_f = round(p_f * _factor, 1)
                c_f = round(c_f * _factor, 1)
                g_f = round(g_f * _factor, 1)
                k_f = float(_kcal_cap)
                logger.info("[Reco] Cap MAX aplicado a '%s': →%.0f kcal (%s)", _nombre_p, k_f, momento_reco)
            # Floor duro: si el LLM fue demasiado conservador, escalar al mínimo del momento
            elif _kcal_floor and k_f < _kcal_floor:
                _factor = _kcal_floor / k_f if k_f > 0 else 1.0
                p_f = round(p_f * _factor, 1)
                c_f = round(c_f * _factor, 1)
                g_f = round(g_f * _factor, 1)
                k_f = float(_kcal_floor)
                logger.info("[Reco] Floor MIN aplicado a '%s': →%.0f kcal (%s)", _nombre_p, k_f, momento_reco)
            cache_macros(_nombre_p, {
                "nombre": _nombre_p,
                "kcal": k_f,
                "prot_g": p_f,
                "carb_g": c_f,
                "grasa_g": g_f,
            })
            _platos_limpios.append((_nombre_p, k_f, p_f, c_f, g_f))
        _persistir_historial_recomendaciones(db, perfil, momento_reco, _platos_limpios)
        bullets = '\n'.join(
            f'- {n} (~{k:.0f} kcal)' for n, k, *_ in _platos_limpios
        )
        return f"Opciones para ti:\n{bullets}"

    # Intento 2 (fallback): el LLM no incluyó P/C/G — extraer solo nombre+kcal
    # y estimar macros reales con _PROMPT_COMIDA por cada plato (sin valores
    # hardcodeados).
    _RE_BULLET_RECO = _re_reco.compile(
        r'([^()\n]{3,80}?)\s*\(~?(\d+(?:\.\d+)?)\s*kcal\)', _re_reco.IGNORECASE
    )
    _platos_parseados = _RE_BULLET_RECO.findall(respuesta_llm_reco or "")

    if _platos_parseados:
        _platos_limpios = []
        for _nombre_p, _kcal_p in _platos_parseados[:3]:
            _nombre_p = _limpiar_nombre(_nombre_p)
            _kcal_f = float(_kcal_p)
            p_f = c_f = g_f = 0.0
            try:
                _raw_macro = await ia_engine._llamar_groq(
                    _PROMPT_COMIDA.format(mensaje=_nombre_p), max_tokens=300, temp=0.0
                )
                _d_macro = _parse_json(_raw_macro)
                _items = (_d_macro or {}).get("alimentos") or []
                if _items:
                    p_f = float(_items[0].get("prot_g", 0) or 0)
                    c_f = float(_items[0].get("carb_g", 0) or 0)
                    g_f = float(_items[0].get("grasa_g", 0) or 0)
                    _kcal_f = round(4 * p_f + 4 * c_f + 9 * g_f, 1) or _kcal_f
            except Exception as e:
                logger.warning("[Reco] No se pudo estimar macros de '%s': %s", _nombre_p, e)
            cache_macros(_nombre_p, {
                "nombre": _nombre_p,
                "kcal": _kcal_f,
                "prot_g": p_f,
                "carb_g": c_f,
                "grasa_g": g_f,
            })
            _platos_limpios.append((_nombre_p, _kcal_f, p_f, c_f, g_f))
        _persistir_historial_recomendaciones(db, perfil, momento_reco, _platos_limpios)
        bullets = '\n'.join(
            f'- {n} (~{k:.0f} kcal)' for n, k, *_ in _platos_limpios
        )
        return f"Opciones para ti:\n{bullets}"

    # 7. Fallback: retornar respuesta del LLM tal como vino
    return respuesta_llm_reco or "No pude generar recomendaciones en este momento."


async def respuesta_chat_llm(
    mensaje: str,
    perfil,
    consumido: float,
    meta: float,
    quemado: float,
    historial: list,
    ia_engine,
) -> str:
    """Respuesta conversacional corta vía LLM."""
    pct = round(consumido / meta * 100) if meta > 0 else 0
    hist_txt = "\n".join(
        f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content'][:120]}"
        for m in historial[-4:]
    ) or "(inicio de conversación)"

    prompt = _PROMPT_CHAT.format(
        nombre=perfil.first_name,
        consumido=round(consumido),
        meta=round(meta),
        pct=pct,
        quemado=round(quemado),
        historial=hist_txt,
        mensaje=mensaje,
        dieta=getattr(perfil, "diet_type", "Normal") or "Normal",
        condiciones=", ".join(getattr(perfil, "medical_conditions", None) or []) or "ninguna",
        objetivo=getattr(perfil, "goal", "mantener peso") or "mantener peso",
    )
    # ── Intercept "qué hora es" — hora real de Perú, sin pasar por el LLM ────────
    _m_lower_hora = mensaje.lower().strip()
    if any(k in _m_lower_hora for k in ("qué hora es", "que hora es", "qué hora son", "que hora son")):
        from app.core.utils import get_peru_now
        return f"Son las {get_peru_now().strftime('%H:%M')} (hora de Perú)."

    # ── Intercept "puedo comer/tomar X?" — respuesta corta sin receta ────────────
    import re as _re_puedo
    _RE_PUEDO_COMER = _re_puedo.compile(
        r'^(puedo|se\s+puede|puedo\s+yo|puede\s+uno)\s+'
        r'(comer|tomar|beber|ingerir|comerme|tomarme)\s+\S',
        _re_puedo.IGNORECASE,
    )
    if _RE_PUEDO_COMER.match(mensaje.strip()):
        _conds_raw   = getattr(perfil, "medical_conditions", None) or []
        _objetivo    = getattr(perfil, "goal", "mantener peso") or "mantener peso"
        # Detectar restricciones dietéticas desde medical_conditions (donde realmente viven)
        _es_vegano   = any("vegano" in c.lower() for c in _conds_raw)
        _es_vegetariano = any("vegetariano" in c.lower() for c in _conds_raw)
        _tiene_diabetes = any("diabetes" in c.lower() for c in _conds_raw)
        # Construir bloque de restricciones claro y explícito
        _restricciones = []
        if _es_vegano:
            _restricciones.append("VEGANO: PROHIBIDO todo producto animal (pollo, carne, pescado, huevo, lácteos)")
        elif _es_vegetariano:
            _restricciones.append("VEGETARIANO: PROHIBIDO carne, pollo y pescado")
        if _tiene_diabetes:
            _restricciones.append("DIABETES: evitar azúcares refinados y alimentos de alto índice glucémico")
        otras = [c for c in _conds_raw if not any(
            k in c.lower() for k in ("vegano", "vegetariano", "diabetes")
        )]
        if otras:
            _restricciones.append(f"Otras condiciones: {', '.join(otras)}")
        _bloque_restricciones = "\n".join(f"- {r}" for r in _restricciones) or "- Sin restricciones especiales"

        _prompt_perm = (
            f"Eres un coach nutricional amigable. Perfil del usuario:\n"
            f"{_bloque_restricciones}\n"
            f"Objetivo: {_objetivo}\n\n"
            f"Pregunta del usuario: '{mensaje}'\n\n"
            f"Responde como un amigo que sabe de nutrición: tono cálido y directo.\n"
            f"Máximo 2 frases cortas (≤35 palabras en total):\n"
            f"  Frase 1: respuesta clara (sí/no/con moderación) + razón según su perfil.\n"
            f"  IMPORTANTE: si el alimento viola una restricción (vegano/vegetariano/diabetes),\n"
            f"  esa restricción es la razón principal.\n"
            f"  Frase 2 (opcional): alternativa concreta — DEBE cumplir TODAS las mismas restricciones.\n"
            f"  Si el usuario es vegano y el alimento es pescado, NO sugieras otro tipo de pescado ni carne.\n"
            f"  Sugiere solo alternativas 100% compatibles con su dieta (ej: tofu, legumbres, soja).\n\n"
            f"PROHIBIDO: recetas, listas, pasos de preparación, párrafos largos.\n"
            f"PROHIBIDO: comillas dobles o simples alrededor de las frases.\n"
            f"PROHIBIDO: mencionar el nombre del usuario."
        )
        _raw_perm = await ia_engine._llamar_groq(_prompt_perm, max_tokens=100, temp=0.5)
        _resultado_perm = _limpiar_markdown(_raw_perm)
        # El LLM a veces ignora "PROHIBIDO mencionar el nombre" — quitar
        # "{Nombre}, " si quedó al inicio de la respuesta.
        _nombre_escaped = _re_puedo.escape(perfil.first_name or "")
        if _nombre_escaped:
            _resultado_perm = _re_puedo.sub(
                rf'^{_nombre_escaped},\s*', '', _resultado_perm, flags=_re_puedo.IGNORECASE
            )
            if _resultado_perm:
                _resultado_perm = _resultado_perm[0].upper() + _resultado_perm[1:]
        return _resultado_perm

    # Recetas y técnicas de ejercicio requieren más tokens para una respuesta completa
    _m_lower = mensaje.lower()
    _es_receta = any(k in _m_lower for k in (
        "como se hace", "cómo se hace", "como se prepara", "cómo se prepara",
        "como hacer", "cómo hacer", "ingredientes de", "receta de",
        "como cocinar", "cómo cocinar",
    ))
    _es_tecnica = any(k in _m_lower for k in (
        "tecnica de", "técnica de",
        "como hacer una", "cómo hacer una",
        "como hacer el", "cómo hacer el",
        "como hago el", "cómo hago el",
        "como hago una", "cómo hago una",
        "como se hace el", "cómo se hace el",
        "como realizar", "cómo realizar",
        "como ejecutar", "como ejecutar",
        "pasos para", "forma correcta",
        "explicame como", "explícame cómo",
        "ensenme como", "enséñame cómo",
    ))
    # Detección de consulta calórica: "cuántas kcal tiene X", "cuánto tiene de X"
    # Para estas preguntas, calculamos con _PROMPT_COMIDA (mismo engine que registro)
    # y cacheamos el resultado → consistencia perfecta con registro posterior.
    _es_consulta_kcal = any(k in _m_lower for k in (
        "cuantas calorias tiene", "cuántas calorías tiene",
        "cuanto tiene de", "cuánto tiene de",
        "cuantas kcal tiene", "cuántas kcal tiene",
        "cuantos gramos tiene", "cuántos gramos tiene",
        "cuanta proteina tiene", "cuánta proteína tiene",
        "valor nutricional de", "macros de",
    ))

    if _es_consulta_kcal:
        # Extraer el alimento de la pregunta y calcular con _PROMPT_COMIDA
        alimento_query = mensaje  # el LLM interpretará la pregunta como alimento
        raw_macros = await ia_engine._llamar_groq(
            _PROMPT_COMIDA.format(mensaje=alimento_query),
            max_tokens=400, temp=0.0
        )
        d_macros = _parse_json(raw_macros)
        if d_macros and d_macros.get("alimentos"):
            # Cachear para consistencia futura
            for item in d_macros["alimentos"]:
                p_i = float(item.get("prot_g", 0) or 0)
                c_i = float(item.get("carb_g", 0) or 0)
                g_i = float(item.get("grasa_g", 0) or 0)
                k_i = round(4*p_i + 4*c_i + 9*g_i, 1) or float(item.get("kcal", 0) or 0)
                if item.get("nombre") and k_i > 0:
                    cache_macros(item["nombre"], {
                        "nombre": item["nombre"], "kcal": k_i,
                        "prot_g": p_i, "carb_g": c_i, "grasa_g": g_i
                    })
            # Construir respuesta con los valores exactos
            primer = d_macros["alimentos"][0]
            p_r = float(primer.get("prot_g", 0) or 0)
            c_r = float(primer.get("carb_g", 0) or 0)
            g_r = float(primer.get("grasa_g", 0) or 0)
            k_r = round(4*p_r + 4*c_r + 9*g_r, 1)
            grm = float(primer.get("porcion_g", 100) or 100)
            nombre_r = primer.get("nombre", "")
            return (
                f"{nombre_r} ({grm:.0f}{'ml' if 'ml' in mensaje.lower() or 'jugo' in mensaje.lower() or 'leche' in mensaje.lower() else 'g'}) "
                f"tiene {k_r:.0f} kcal — P:{p_r:.1f}g C:{c_r:.1f}g G:{g_r:.1f}g."
            )

    _max_tok = 500 if (_es_receta or _es_tecnica) else 200
    raw = await ia_engine._llamar_groq(prompt, max_tokens=_max_tok, temp=0.7)
    resultado = _limpiar_markdown(raw)

    # Para recetas: añadir saltos de línea antes de secciones clave
    if _es_receta:
        import re as _re_fmt
        # El LLM a veces ignora "NO intro" y antepone un resumen del historial.
        # Si "Ingredientes:" no está al inicio, descartar todo lo anterior.
        _idx_ing = resultado.lower().find("ingredientes:")
        if _idx_ing > 0:
            resultado = resultado[_idx_ing:]
        # Separar "Ingredientes:" y "Preparación:" en líneas propias
        resultado = _re_fmt.sub(r'\s*(Ingredientes:)', r'\n\nIngredientes:', resultado)
        resultado = _re_fmt.sub(r'\s*(Preparaci[oó]n:)', r'\n\nPreparación:', resultado)
        # Cada paso numerado en su propia línea
        resultado = _re_fmt.sub(r'\.?\s*(\d+\.)\s+', r'\n\1 ', resultado)
        resultado = resultado.strip()

    return resultado


# ── Caché de macros (consistencia recomendación → registro) ──────────────────
# Cuando el asistente recomienda un plato calcula sus macros exactos y los
# guarda aquí. Si el usuario registra ese plato en la misma sesión, se usan
# los mismos valores → consistencia perfecta sin BD hardcodeada.

import time as _time
import unicodedata as _ud2
import re as _re2

_macro_cache: dict = {}
_CACHE_TTL = 7200  # 2 horas


_SINONIMOS_ALIMENTOS = {
    # quinua — todas las variantes de voz y ortografía
    "quinoa": "quinua", "kinua": "quinua", "kino":  "quinua",
    "quino":  "quinua", "kinoa": "quinua", "quinuoa":"quinua",
    "quinuo": "quinua", "quínoa":"quinua", "quínua": "quinua",
    "kinwa":  "quinua", "kinwua":"quinua",
    # otros sinónimos peruanos comunes
    "palta": "aguacate", "aguacate": "palta",
    "choclo": "maiz",  "maiz": "choclo",
    "camote": "batata", "batata": "camote",
}


def _normalizar_nombre(nombre: str) -> str:
    """Normaliza nombre: quita tildes, minúsculas, aplica sinónimos."""
    n = nombre.lower().strip()
    n = "".join(c for c in _ud2.normalize("NFD", n) if _ud2.category(c) != "Mn")
    n = _re2.sub(r"\s+", " ", n)
    # Reemplazar sinónimos token a token
    tokens = n.split()
    tokens = [_SINONIMOS_ALIMENTOS.get(t, t) for t in tokens]
    return " ".join(tokens)


def _cache_key(nombre: str) -> str:
    return _normalizar_nombre(nombre)


def cache_macros(nombre: str, macros: dict) -> None:
    """Guarda macros en caché con TTL de 2 horas."""
    key = _cache_key(nombre)
    _macro_cache[key] = {**macros, "_ts": _time.time()}
    logger.info("[MacroCache] Guardado: %s → %s kcal", nombre, macros.get("kcal", "?"))


def get_cached_macros(nombre: str) -> dict | None:
    """Retorna macros cacheados o None si no existe / expiró."""
    key = _cache_key(nombre)
    entry = _macro_cache.get(key)
    if entry and (_time.time() - entry.get("_ts", 0)) < _CACHE_TTL:
        return {k: v for k, v in entry.items() if k != "_ts"}
    return None


def _buscar_en_cache(mensaje: str) -> dict | None:
    """Busca en caché con:
    1. Coincidencia exacta normalizada (quinoa → quinua → mismo key)
    2. Fuzzy matching con difflib (umbral 0.75) para errores de voz como Kino, equino"""
    from difflib import SequenceMatcher
    msg_norm = _normalizar_nombre(mensaje)
    ahora = _time.time()
    mejor_ratio = 0.0
    mejor_entry = None

    for key, entry in list(_macro_cache.items()):
        if (ahora - entry.get("_ts", 0)) >= _CACHE_TTL:
            continue
        # 1. Coincidencia exacta: la key está contenida en el mensaje normalizado
        if key in msg_norm:
            logger.info("[MacroCache] Hit exacto: '%s'", key)
            return {k: v for k, v in entry.items() if k != "_ts"}
        # 2. Fuzzy: comparar key con substrings del mensaje de longitud similar
        ratio = SequenceMatcher(None, key, msg_norm).ratio()
        if ratio > mejor_ratio:
            mejor_ratio = ratio
            mejor_entry = entry

    if mejor_ratio >= 0.72 and mejor_entry:
        logger.info("[MacroCache] Hit fuzzy (ratio=%.2f)", mejor_ratio)
        return {k: v for k, v in mejor_entry.items() if k != "_ts"}
    return None


# ── Helpers privados ──────────────────────────────────────────────────────────

_ANIMAL_VEGANO = frozenset({
    "mariscos", "camarones", "camaron", "pulpo", "calamar", "langostino", "langosta",
    "atun", "salmón", "salmon", "pescado", "trucha", "caballa", "merluza", "tilapia",
    "ceviche", "tiradito", "jalea", "chicharron de pescado",
    "pollo", "pechuga", "gallina", "pavo", "pato", "cuy",
    "carne", "res", "lomo", "bistec", "cerdo", "chancho", "chicharron",
    "huevo", "tortilla de huevo", "huevo frito", "huevo sancochado",
    "leche", "queso", "yogur", "mantequilla", "crema",
})
_ANIMAL_VEGETARIANO = frozenset({
    "mariscos", "camaron", "pulpo", "calamar", "atun", "salmon", "pescado",
    "trucha", "caballa", "merluza", "ceviche", "tiradito",
    "pollo", "pechuga", "gallina", "pavo", "pato", "cuy",
    "carne", "res", "lomo", "bistec", "cerdo", "chancho", "chicharron de carne",
})

def _detectar_conflicto_dieta(nombres: list, diet_type: str) -> str | None:
    """Detecta si algún alimento registrado no corresponde a la dieta del usuario."""
    if not diet_type or not nombres:
        return None
    dieta = diet_type.lower().strip()
    prohibidos = set()
    if "vegano" in dieta:
        prohibidos = _ANIMAL_VEGANO
    elif "vegetariano" in dieta:
        prohibidos = _ANIMAL_VEGETARIANO
    if not prohibidos:
        return None
    conflictos = []
    for nombre in nombres:
        n_lower = nombre.lower()
        for p in prohibidos:
            if p in n_lower:
                conflictos.append(nombre)
                break
    if not conflictos:
        return None
    tipo = "vegana" if "vegano" in dieta else "vegetariana"
    items = ", ".join(conflictos[:2])
    return f"⚠️ {items} no es parte de tu dieta {tipo}. Registrado igual para mantener tu historial."


def _limpiar_markdown(texto: str) -> str:
    """Elimina markdown y patrones de intro/cierre del LLM."""
    import re as _re_md
    t = texto
    # Eliminar negrita/cursiva: **texto** → texto, *texto* → texto
    t = _re_md.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', t)
    # Eliminar encabezados: # Título → (eliminado)
    t = _re_md.sub(r'^#{1,6}\s+.*$', '', t, flags=_re_md.MULTILINE)
    # Eliminar líneas solo con guiones (separadores)
    t = _re_md.sub(r'^-{3,}$', '', t, flags=_re_md.MULTILINE)

    # Eliminar párrafo de intro si la primera línea es relleno
    # "Leonardo, me alegra...", "Leonardo, para hacer X, necesitas..."
    lineas = t.split('\n')
    _INTRO_PATS = _re_md.compile(
        r'^[A-Za-záéíóúÁÉÍÓÚñÑ]+,\s*(me\s+alegra|qué\s+buena|es\s+un\s+placer|'
        r'para\s+hacer\s+\w+.*necesitas|para\s+preparar|para\s+realizar\s+una)',
        _re_md.IGNORECASE
    )
    if lineas and _INTRO_PATS.match(lineas[0].strip()):
        lineas = lineas[1:]  # eliminar primera línea de intro
    t = '\n'.join(lineas)

    # Limpiar líneas vacías múltiples
    t = _re_md.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def _parse_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        # 1. Eliminar comentarios JavaScript: // texto  y  /* texto */
        cleaned = re.sub(r'//[^\n\r"]*', '', cleaned)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        # 2. Evaluar expresiones aritméticas simples: 2 * 50 → 100, 3 * 55 → 165
        cleaned = re.sub(
            r'(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)',
            lambda m: str(round(float(m.group(1)) * float(m.group(2)), 1)),
            cleaned
        )
        # 3. Eliminar unidades pegadas a números: 8g→8, 10ml→10, 420kcal→420
        cleaned = re.sub(r'(\d+(?:\.\d+)?)\s*(?:g|ml|kcal|kg|mg|cc)(?=\s*[,}\]])', r'\1', cleaned)
        # 4. Eliminar trailing commas antes de } o ]
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        # Intentar dict primero (comida, ejercicio único, recomendación)
        # luego array (ejercicios múltiples)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # Fallback: array JSON (múltiples ejercicios)
        m_arr = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if m_arr:
            try:
                result = json.loads(m_arr.group(0))
                if isinstance(result, list):
                    return result
            except Exception:
                pass
    except Exception:
        pass
    return None


def _get_or_create_progreso(db: Session, client_id: int, fecha, plan_hoy: dict):
    from app.models.historial import ProgresoCalorias
    prog = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == client_id,
        ProgresoCalorias.fecha == fecha,
    ).first()
    if not prog:
        from sqlalchemy import text as sqla_text
        meta = int(plan_hoy.get("calorias_dia", 2000))
        db.execute(sqla_text("""
            INSERT INTO progreso_calorias
                (client_id, fecha, calorias_consumidas, calorias_quemadas,
                 proteinas_consumidas, carbohidratos_consumidos, grasas_consumidas)
            VALUES (:cid, :fecha, 0, 0, 0, 0, 0)
            ON CONFLICT DO NOTHING
        """), {"cid": client_id, "fecha": fecha})
        db.commit()
        prog = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == client_id,
            ProgresoCalorias.fecha == fecha,
        ).first()
    return prog
