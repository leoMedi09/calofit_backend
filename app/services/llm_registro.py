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

⚠️ EXCEPCIÓN DE MÁXIMA PRIORIDAD — DATOS DE ETIQUETA DADOS POR EL USUARIO:
Si el usuario menciona valores nutricionales explícitos de un producto (calorías
y/o macros — sin importar si los da por porción, por 100g/100ml, o como total
de lo que consumió, ni cómo los exprese: "tiene X kcal", "trae Y de proteína
cada 100ml", "la etiqueta dice...", etc.), esos números SIEMPRE tienen prioridad
sobre tu propio conocimiento del producto. NO los reemplaces con tu estimación
de marca/producto genérico — escala esos números EXACTOS a la cantidad real que
consumió (ej. si dice "320 ml" y los valores son "por 100ml", multiplica ×3.2).
Esto aplica a CUALQUIER producto, no solo marcas reconocidas — el usuario puede
estar leyendo la etiqueta real que tiene enfrente, que es más preciso que tu
conocimiento general.

⚠️ PRODUCTOS DE MARCA SIN DATOS DADOS (sin etiqueta mencionada por el usuario):
Si el usuario nombra un producto comercial/de marca específico (peruano o de
cualquier país: Inca Kola, Gloria, Pilsen, Coca-Cola, Sublime, etc.) SIN dar
sus propios números, usa tu conocimiento REAL de ESE producto específico
(sus valores nutricionales típicos reales), NO una estimación genérica de la
categoría ("gaseosa" genérica, "chocolate" genérico). Si reconoces la marca,
sé tan preciso como puedas con sus valores reales conocidos. Si NO reconoces
la marca o no estás seguro de sus valores reales, sí usa una estimación
razonable de su categoría general — pero nunca inventes un valor "de marca"
falso presentándolo como si fuera específico.

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
0. ⚠️ Si el mensaje NO menciona ningún alimento ni bebida (ej. datos personales como
   peso/altura/frecuencia de entrenamiento, una pregunta, un saludo, una instrucción
   sin comida) → NO INVENTES una comida para llenar el JSON. Responde exactamente
   {{"alimentos":[], "prot_total":0, "carb_total":0, "grasa_total":0}}.
   Esto es distinto de "ficticio": aquí no hay NADA que extraer, ni real ni inventado.
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
0. ⚠️ Si el mensaje NO nombra un ejercicio o actividad física ESPECÍFICA (ej. solo
   confirma que ya no hay dolor, que está recuperado, que "entrenó sin problemas"
   sin decir qué hizo, una pregunta, un saludo) → NO INVENTES una "rutina completa"
   genérica. Responde {{"encontrado": false, "ejercicio": null, "kcal_quemadas": 0,
   "duracion_min": 0, "met": 0, "intensidad": "Baja"}}. Solo extrae si el mensaje
   nombra QUÉ hizo (sentadillas, trote, press banca, fútbol, etc.).
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
{contexto_lesion}
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
{bloque_perfil}

Conversación reciente:
{historial}

Mensaje actual: "{mensaje}"

REGLAS DE RESPUESTA:
⛔ REGLAS ABSOLUTAS (se aplican SIEMPRE, sin excepción):
  1. PROHIBIDO cualquier markdown: **negrita**, *cursiva*, # títulos. Solo texto plano.
  1b. PROHIBIDO usar abreviaturas tipo etiqueta para macros: "P:Xg C:Yg G:Zg",
      "kcal:", "prot:". Esos números van en PROSA natural, como los diría una
      persona: "tiene 112 kcal, con 1g de proteína, 27g de carbohidratos y casi
      nada de grasa" — NUNCA pierdas ningún valor numérico ni cambies las
      cantidades por sonar natural, solo cambia CÓMO se presentan (texto
      corrido, no pares clave:valor).
  1c. PROHIBIDO inventar una causa médica para dolor/lesión física basada en
      el balance de calorías del perfil (ej. "te duele la rodilla por exceso
      de calorías/sobrepeso"). El consumo calórico de hoy NO es una causa de
      dolor articular o muscular agudo — son cosas no relacionadas. Si el
      tema es dolor/lesión, responde SOLO sobre eso (causas físicas reales:
      sobrecarga, mala postura, impacto, etc.), aunque tengas datos de
      calorías disponibles en tu contexto — ignóralos para este tipo de
      pregunta.
  2. PROHIBIDO empezar con frases de relleno: "Leonardo, me alegra...", "Qué buena pregunta...", "Es un placer...". Empieza directo al tema.
  3. PROHIBIDO terminar con pregunta: "¿Quieres saber más?", "¿Te gustaría...?". Termina con punto.
     Única excepción: si el usuario pidió "consejo" o "ayuda" de forma EXPLÍCITA con esa palabra,
     puedes terminar con una pregunta corta. En CUALQUIER otro caso, termina con punto.
  4. LÍMITE DE LONGITUD: máximo 3 oraciones cortas en total (cuenta mentalmente antes de responder).
     Si la respuesta te queda más larga, recórtala — elimina la oración menos importante.
     PROHIBIDO repetir o citar el mensaje del usuario al inicio de tu respuesta.

⛔ ADAPTACIÓN DE DIETA (CRÍTICO):
  Si Dieta = "Vegano" o "Vegetariano" → PROHIBIDO ingredientes animales en recetas.
    Para platos con carne/pescado: adapta AUTOMÁTICAMENTE al sustituto vegetal SIN que el usuario lo pida.
    Ceviche vegano → usa palmito o champiñones. Lomo saltado vegano → usa tofu o setas.
    Siempre MENCIONA que es la versión vegana: "Versión vegana: en lugar de pescado, usa palmito..."
  Si Dieta = Normal, Mediterránea, Cetogénica, Diabético u OTRA → PROHIBIDO mencionar
    versiones veganas, sustitutos vegetales ni alternativas veganas en la receta.
    Usa los ingredientes originales del plato sin ofrecer variantes no solicitadas.
  Si Condiciones incluye Diabetes → evita azúcar, miel, carbos refinados en la receta.

CÓMO USAR LA APP ('cómo uso la app', 'cómo registro mi comida/ejercicio',
'dónde veo mi progreso', 'cómo funciona esto'):
  ⚠️ Esto es la ÚNICA fuente real de la interfaz — NO inventes botones, pantallas
  ni pasos que no estén aquí. Si no sabes algo de la app que no esté en esta
  lista, dilo en general sin inventar un botón específico.
  - Registrar comida o ejercicio: se escribe directo en el chat, en lenguaje
    natural (ej. "comí pollo con arroz" o "hice 3 series de 10 sentadillas con
    20kg"). También hay 2 íconos junto al cuadro de texto del chat: uno naranja
    (🍽️) para registro rápido de comida, uno verde (🏋️) para armar una rutina
    de ejercicio. También se puede dictar por voz con el ícono del micrófono
    (el texto aparece en el cuadro para revisar antes de enviar).
  - Navegación: la barra inferior tiene 5 pestañas — Inicio (resumen del día),
    Asistente (este chat), Balance, Seguimiento (progreso histórico), Perfil.
  - No hay botón literal llamado "Registrar comida" en la pantalla principal —
    el registro es conversacional, no un formulario con menú de tipo de comida.

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
  Responde en una frase natural con los 4 valores (kcal y los 3 macros) — ver regla 1b,
  NUNCA en formato de etiqueta.

- CÁLCULO nutricional ('cuánta proteína necesito', 'cuántas calorías necesito al día'):
  ⚠️ Esto es una pregunta de NÚMERO, no una petición de plato — PROHIBIDO responder
  con un alimento o receta aquí.
  Para PROTEÍNA: si el perfil incluye "Meta proteína (plan)", usa EXACTAMENTE ese
  número — es el cálculo real de su plan nutricional, ya considera su objetivo
  específico (puede ser distinto de una fórmula genérica). PROHIBIDO recalcular o
  dar un rango distinto cuando ese dato está disponible.
  Solo si "Meta proteína (plan)" NO aparece en el perfil, calcula con el Peso y el
  Objetivo: 1.6-2.2 g/kg si es ganar músculo · 1.2-1.6 g/kg si es mantener o perder peso.
  Para CALORÍAS: usa las kcal de la meta diaria del perfil si están disponibles.
  Responde con el número y una frase de por qué, en 2 oraciones máximo.
  Si no hay Peso ni Meta proteína disponibles, pide el peso antes de calcular — no inventes un número.

- Recetas peruanas: Causa=PAPA AMARILLA. Ceviche=PESCADO CRUDO. Lomo saltado=RES.
  Vegano: adapta con tofu/palmito manteniendo la base.
- Usa el historial para dar continuidad a la conversación.

- CONTINUIDAD DE TEMA EN RESPUESTAS DE SEGUIMIENTO (CRÍTICO):
  Si el mensaje actual es una respuesta corta a algo que TÚ preguntaste o sugeriste
  en el turno anterior ("sí", "dale", "dame consejo", "cuéntame más", "ok", "claro"),
  CONTINÚA exactamente el mismo tema del turno anterior en 2-3 oraciones — el mismo
  límite de longitud que cualquier pregunta simple. NO agregues temas no relacionados
  (ej. si el tema venía siendo ejercicio/lesión, NO mezcles balance de kcal ni dieta a
  menos que el usuario lo pida explícitamente). Una respuesta de seguimiento corta NO
  es una invitación a dar un resumen general del perfil del usuario.

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
        # Solo para la ruta LLM (no caché): descartar ítems cuyo nombre no
        # tenga ninguna palabra presente en el mensaje real — evita registrar
        # una comida inventada cuando el mensaje no menciona comida alguna
        # (ej. "Peso 80 kg, mido 1.75..." → no debe registrar "Pollo saltado").
        if not goto_save:
            datos["alimentos"] = [
                a for a in datos["alimentos"]
                if _extraccion_tiene_base_textual(a.get("nombre", ""), mensaje)
            ]

    # Validar que hay alimentos con macros
    _items = datos.get("alimentos", [])

    # ── Chequeo de completitud (solo ruta LLM, no caché) ──────────────────────
    # Encontrado en pruebas reales: "arroz con palta y mi taza de gelatina" →
    # el LLM extrajo solo Arroz + Gelatina, omitiendo "palta", a pesar de que
    # la Regla 9 del prompt es clara (dos alimentos completos unidos por "con"
    # sin ser un combo reconocido = dos ítems separados). Pasó igual 2/2 veces
    # con temp=0.0 — no es ruido, es un punto débil estable del modelo para
    # esta frase. Reintento dirigido en vez de confiar en que la regla del
    # prompt baste (mismo patrón que el resto de guards de hoy).
    if not goto_save and _items:
        _faltantes = _palabras_faltantes_en_extraccion(mensaje, _items)
        if _faltantes:
            logger.info("[Registro] Posibles alimentos faltantes: %s — verificando", _faltantes)
            _nombres_ya_registrados = ', '.join(a.get('nombre', '') for a in _items)
            _prompt_faltante = (
                f"Mensaje original: \"{mensaje}\"\n"
                f"Ya se registraron estos alimentos: {_nombres_ya_registrados}.\n"
                f"El mensaje también menciona estas palabras sueltas: {', '.join(_faltantes)}.\n"
                f"Si alguna de esas palabras es un alimento o bebida REAL ADICIONAL "
                f"(no un ingrediente ya incluido en los platos de arriba, no un adjetivo, "
                f"no un verbo o palabra de comando del usuario como 'agrega'/'falta'/'olvidé', "
                f"no una palabra normal de la frase), agrégalo.\n"
                f"⚠️ PROHIBIDO repetir cualquiera de estos alimentos ya registrados: "
                f"{_nombres_ya_registrados} — si la palabra suelta se refiere a algo "
                f"que ya está en esa lista, NO lo incluyas de nuevo.\n"
                f"Si ninguna palabra suelta es un alimento adicional real y distinto, "
                f"responde alimentos vacío.\n"
                f'Responde SOLO JSON: {{"alimentos": [{{"nombre": "...", "es_real": true, '
                f'"cantidad": 1, "porcion_g": numero, "kcal": numero, "prot_g": numero, '
                f'"carb_g": numero, "grasa_g": numero}}]}}'
            )
            _raw_faltante = await ia_engine._llamar_groq(_prompt_faltante, max_tokens=250, temp=0.0)
            _datos_faltante = _parse_json(_raw_faltante)
            if _datos_faltante and _datos_faltante.get("alimentos"):
                # No confiar solo en la instrucción del prompt de "no repitas" —
                # verificar con código que el nombre recuperado no sea ya uno de
                # los registrados (encontrado en pruebas: "Te faltó la palta" +
                # "Agrégalo en el registro" volvía a traer "Palta" duplicada
                # porque "falto" disparó otro reintento sin relación real).
                _nombres_ya_norm = {_normalizar_nombre(a.get("nombre", "")) for a in _items}
                _nuevos = [
                    a for a in _datos_faltante["alimentos"]
                    if a.get("es_real", True) is not False
                    and _extraccion_tiene_base_textual(a.get("nombre", ""), mensaje)
                    and _normalizar_nombre(a.get("nombre", "")) not in _nombres_ya_norm
                ]
                if _nuevos:
                    logger.info(
                        "[Registro] Alimento(s) recuperado(s): %s",
                        [a.get("nombre") for a in _nuevos],
                    )
                    datos["alimentos"].extend(_nuevos)
                    _items = datos["alimentos"]

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

    # Aplica el modificador cuando hay 1 o 2 ítems. Con 2 ítems cubre combos que el
    # LLM separa en dos partes (ej: "un cuarto de pan francés con palta" → pan+palta).
    if _factor_porcion and len(_items) <= 2:
        for _it in _items:
            for _campo in ("porcion_g", "kcal", "prot_g", "carb_g", "grasa_g"):
                if _it.get(_campo) is not None:
                    _it[_campo] = round(float(_it[_campo]) * _factor_porcion, 1)
        for _campo_total in ("prot_total", "carb_total", "grasa_total", "kcal_total"):
            if datos.get(_campo_total) is not None:
                datos[_campo_total] = round(float(datos[_campo_total]) * _factor_porcion, 1)

    # Gramaje EXPLÍCITO en número ("50g de arroz con pollo") — encontrado en
    # pruebas reales: el LLM devolvió porcion_g=350 ignorando los "50g" que
    # el usuario pidió explícitamente (Regla 12 del prompt no es garantía,
    # mismo patrón que el resto de hoy). Solo con 1 ítem, para no adivinar a
    # cuál de varios alimentos se refiere el número.
    _match_gramos_explicitos = re.search(
        r'\b(\d+(?:[.,]\d+)?)\s*(?:gr|grs|gramos?|g)\b', _msg_low_porcion
    )
    if _match_gramos_explicitos and len(_items) == 1 and not _factor_porcion:
        _gramos_pedidos = float(_match_gramos_explicitos.group(1).replace(',', '.'))
        _it0 = _items[0]
        _porcion_actual = float(_it0.get("porcion_g", 0) or 0)
        if _porcion_actual > 0 and abs(_porcion_actual - _gramos_pedidos) > _porcion_actual * 0.15:
            _factor_gramos = _gramos_pedidos / _porcion_actual
            logger.warning(
                "[Registro] '%sg' pedido pero LLM devolvio porcion_g=%s — re-escalando",
                _gramos_pedidos, _porcion_actual,
            )
            for _campo in ("porcion_g", "kcal", "prot_g", "carb_g", "grasa_g"):
                if _it0.get(_campo) is not None:
                    _it0[_campo] = round(float(_it0[_campo]) * _factor_gramos, 1)
            for _campo_total in ("prot_total", "carb_total", "grasa_total", "kcal_total"):
                if datos.get(_campo_total) is not None:
                    datos[_campo_total] = round(float(datos[_campo_total]) * _factor_gramos, 1)

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
    historial: list = None,
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
    # Descartar ejercicios cuyo nombre no tenga base en el mensaje real — evita
    # registrar un entrenamiento inventado cuando el mensaje no describe
    # ninguno (ej. "La molestia ya desapareció, volví a entrenar sin
    # problemas" → no debe registrar "Rutina completa en el gimnasio").
    ejercicios_raw = [
        e for e in ejercicios_raw
        if _extraccion_tiene_base_textual(e.get("ejercicio", ""), mensaje)
    ]

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
        # El LLM a veces devuelve duracion_min=0 pese a haber series/reps reales
        # (encontrado en pruebas: "press banca inclinado 3 por 5 repeticiones"
        # → duracion_min=0). Estimación determinista de respaldo: ~5 min por
        # serie (descanso+setup domina sobre el tiempo de la repetición misma)
        # — calibrado para coincidir con la referencia que el propio prompt ya
        # usa más abajo ("1 ejercicio 3×10 ≈ 15 min" → 5 min/serie), no con un
        # cálculo de segundos por repetición que subestimaba mucho (daba ~3.8
        # min para 3×5 cuando el LLM, al estimar directo, da ~15 min — la
        # inconsistencia entre ambos era visible para el usuario).
        if duracion <= 0 and series:
            duracion = round(int(series) * 5, 1)
        kcal_formula = round(met * peso_kg * 3.5 / 200 * duracion, 1)
        kcal_llm     = round(float(datos.get("kcal_quemadas", 0) or 0), 1)
        # Si kcal_formula es 0 (duracion real 0, sin series/reps para estimar),
        # comparar kcal_llm contra ella siempre "parece" desproporcionado
        # (cualquier valor positivo es ">2.5×0") y el guard lo sobrescribía a 0
        # incluso cuando kcal_llm era razonable. En ese caso degenerado, confiar
        # en kcal_llm si es positivo, no forzar 0.
        if kcal_formula <= 0:
            kcal = kcal_llm if kcal_llm > 0 else 0.0
        else:
            kcal = kcal_formula if kcal_llm > kcal_formula * 2.5 or kcal_llm < kcal_formula * 0.3 else kcal_llm

        # Escribir la duración corregida de vuelta en "datos" — el dict de
        # retorno más abajo vuelve a leer duracion_min desde ejercicios_raw
        # (sum(e.get("duracion_min",0) for e in ejercicios_raw)) y sin esto
        # seguía mostrando 0min en la tarjeta del chat aunque la base de datos
        # ya tenía el valor correcto (encontrado comparando ambos).
        datos["duracion_min"] = duracion

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

    # Aviso (no bloqueo) si el ejercicio registrado coincide con una lesión
    # activa mencionada antes en la conversación — encontrado en pruebas
    # reales: "me duele la rodilla" → el asistente recomienda cuidado →
    # "hice sentadillas 3x10" se registraba sin ningún aviso, ignorando lo
    # que el usuario mismo dijo un turno antes. No se bloquea el registro
    # (el ejercicio sí se hizo, hay que reflejarlo), solo se avisa.
    try:
        from app.services.rutina_service import _LESIONES_SUSTITUCION, _detectar_lesiones, filtrar_lesiones_activas
        _texto_hist_ej_warn = " ".join(str(h.get("content", "")) for h in (historial or []))
        _lesiones_cand_warn = _detectar_lesiones(
            list(getattr(perfil, "medical_conditions", None) or []) + [_texto_hist_ej_warn]
        )
        _lesiones_activas_warn = filtrar_lesiones_activas(_lesiones_cand_warn, historial, mensaje)
        _nombres_normalizados = _normalizar_nombre(" ".join(e["nombre"] for e in ejercicios_guardados))
        for _lesion_w in _lesiones_activas_warn:
            _riesgosos_w = {r for r in _LESIONES_SUSTITUCION[_lesion_w]["sustituir"] if r != "default"}
            if any(r in _nombres_normalizados for r in _riesgosos_w):
                msg += (
                    f" ⚠️ Mencionaste antes molestia en {_lesion_w} — si sentiste dolor "
                    f"al hacerlo, considera una alternativa de bajo impacto la próxima vez."
                )
                break
    except Exception as _e_warn_ej:
        logger.warning("[Registro Ejercicio] Aviso de lesion fallo (no crítico): %s", _e_warn_ej)

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


def _construir_mensaje_natural_reco(
    platos_limpios: list,
    palabra_evitada: str | None,
    condicion_relevante: str | None,
    estilo_evitado_momento: str | None,
) -> str:
    """
    Construye el mensaje final en prosa natural (no lista con guiones) a
    partir de los platos YA validados/parseados — la estructura de datos
    (kcal/macros) ya se extrajo y cacheó antes de llamar a esto, así que
    cambiar la presentación aquí no afecta el registro/caché posterior.
    """
    nombres = [n for n, *_ in platos_limpios]
    if not nombres:
        return "No pude generar recomendaciones en este momento."

    if len(nombres) == 1:
        lista_natural = nombres[0]
    else:
        lista_natural = ", ".join(nombres[:-1]) + " o " + nombres[-1]

    kcals = [k for _, k, *_ in platos_limpios]
    kcal_min, kcal_max = min(kcals), max(kcals)
    kcal_txt = (
        f"~{kcal_min:.0f} kcal" if kcal_min == kcal_max
        else f"entre {kcal_min:.0f} y {kcal_max:.0f} kcal"
    )

    razones = []
    if palabra_evitada and condicion_relevante:
        razones.append(f"por tu condición de {condicion_relevante}, evité incluir {palabra_evitada}")
    if estilo_evitado_momento:
        razones.append("a esta hora evité algo frito o pesado, mejor algo más ligero")

    if razones:
        intro_razones = "; y ".join(razones)
        intro_razones = intro_razones[0].upper() + intro_razones[1:]
        return f"{intro_razones}, así que te recomiendo {lista_natural} — {kcal_txt}."

    return f"Te recomiendo {lista_natural} — {kcal_txt}."


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
        # Guard de seguridad por lesión — detecta la lesión tanto en el PERFIL
        # como en el MENSAJE (el usuario puede mencionarla al vuelo sin tenerla
        # guardada), y reutiliza las sustituciones ya conocidas de
        # rutina_service.py en vez de mantener una lista nueva por separado.
        # Se calcula ANTES de generar (no solo después) para informar al LLM
        # desde el prompt — antes el LLM "nacía a ciegas" y solo se corregía
        # reactivamente si violaba algo; ahora además se le avisa de entrada.
        from app.services.rutina_service import (
            _LESIONES_SUSTITUCION, _detectar_lesiones, filtrar_lesiones_activas,
        )
        _condiciones_lista_ej = list(getattr(perfil, "medical_conditions", None) or [])
        # Incluir TODO el historial: una lesión mencionada varios turnos atrás
        # (ej. "me duele la rodilla" en el turno 1) debe seguir siendo
        # candidata aunque ya no esté en los últimos 2-4 turnos — antes esta
        # ventana corta hacía que la rodilla "desapareciera" de las candidatas
        # y nunca llegara a evaluarse en filtrar_lesiones_activas. No hay
        # riesgo de que esto bloquee para siempre: filtrar_lesiones_activas
        # (abajo) ya resuelve si sigue activa o se recuperó.
        _texto_historial_ej = " ".join(
            str(h.get("content", "")) for h in (historial or [])
        )
        _lesiones_candidatas_ej = _detectar_lesiones(
            _condiciones_lista_ej + [mensaje or "", _texto_historial_ej]
        )
        # Filtrar las que el usuario ya indicó como recuperadas más reciente
        # que la última mención de dolor — si no, "rodilla" bloquea para
        # siempre una vez mencionada, aunque el usuario diga que ya sanó.
        _lesiones_activas = filtrar_lesiones_activas(
            _lesiones_candidatas_ej, historial, mensaje
        )

        _ejercicios_riesgosos: set[str] = set()
        _alternativas_seguras: list[str] = []
        _contexto_lesion = ""
        if _lesiones_activas:
            for _lesion in _lesiones_activas:
                _cfg = _LESIONES_SUSTITUCION[_lesion]
                for _riesgoso, (_id_seguro, _nombre_seguro) in _cfg["sustituir"].items():
                    if _riesgoso != "default":
                        _ejercicios_riesgosos.add(_riesgoso)
                    _alternativas_seguras.append(_nombre_seguro)
            # `sustituir` solo cubre ejercicios de FUERZA por lesión (sentadilla,
            # prensa, press...) — no cardio de impacto. Una lesión de rodilla
            # también prohíbe trote/correr/salto, sin importar la lesión exacta:
            # se sumó tras encontrar "Trote 20 minutos" recomendado como ejercicio
            # "seguro" a alguien que dijo que correr le duele la rodilla.
            if "rodilla" in _lesiones_activas:
                _ejercicios_riesgosos |= set(_ACCIONES_IMPACTO)
            _justif_previa = "; ".join(
                _LESIONES_SUSTITUCION[l]["justificacion"] for l in _lesiones_activas
            )
            _contexto_lesion = (
                f"\n⚠️ LESIÓN ACTIVA DEL USUARIO: {_justif_previa}.\n"
                f"PROHIBIDO sugerir: {', '.join(sorted(_ejercicios_riesgosos))} (ni variantes).\n"
                f"Prioriza alternativas como: {', '.join(dict.fromkeys(_alternativas_seguras))}.\n"
            )

        prompt = _PROMPT_RECOMENDACION_EJERCICIO.format(
            nombre=perfil.first_name,
            objetivo=objetivo,
            condiciones=condiciones,
            contexto_lesion=_contexto_lesion,
            mensaje=mensaje,
        )
        respuesta_ej = await ia_engine._llamar_groq(prompt, max_tokens=300, temp=0.7)

        if _lesiones_activas:
            # _normalizar_nombre quita tildes — sin esto, "jalon" (keyword) nunca
            # coincide con "Jalón" (como el LLM lo escribe naturalmente).
            # Este chequeo sigue existiendo como red de seguridad — el contexto
            # arriba reduce la probabilidad de violación, no la garantiza.
            _viola_lesion = any(
                r in _normalizar_nombre(respuesta_ej or "") for r in _ejercicios_riesgosos
            )

            if _viola_lesion:
                logger.warning(
                    "[Reco-Ejercicio] Ejercicio riesgoso para lesión detectada — reintentando"
                )
                _justif_lesion = "; ".join(
                    _LESIONES_SUSTITUCION[l]["justificacion"] for l in _lesiones_activas
                )
                _prompt_retry_ej = (
                    f"Eres entrenador personal. El usuario tiene: {_justif_lesion}.\n"
                    f"NO sugieras NINGUNO de estos ejercicios ni variantes: "
                    f"{', '.join(sorted(_ejercicios_riesgosos))}.\n"
                    f"En su lugar usa alternativas seguras como: {', '.join(_alternativas_seguras)}.\n"
                    f"Mensaje original del usuario: \"{mensaje}\"\n"
                    f"Responde en 2-3 frases naturales (sin listas, sin numeración, sin pasos "
                    f"de ejecución), mencionando 2-3 ejercicios seguros con series/reps. "
                    f"Sin preguntas al final."
                )
                respuesta_ej = await ia_engine._llamar_groq(
                    _prompt_retry_ej, max_tokens=200, temp=0.3
                )

                # Verificación final + fallback garantizado en prosa natural (no lista)
                _aun_viola_ej = any(
                    r in _normalizar_nombre(respuesta_ej or "") for r in _ejercicios_riesgosos
                )
                if _aun_viola_ej:
                    logger.warning(
                        "[Reco-Ejercicio] Reintento también riesgoso — usando fallback seguro"
                    )
                    _nombres_unicos = list(dict.fromkeys(_alternativas_seguras))[:3]
                    if len(_nombres_unicos) > 1:
                        _lista_segura = ", ".join(_nombres_unicos[:-1]) + " o " + _nombres_unicos[-1]
                    else:
                        _lista_segura = _nombres_unicos[0]
                    respuesta_ej = (
                        f"Por tu lesión, mejor evitamos ejercicios de alto impacto en esa zona. "
                        f"Prueba con {_lista_segura} — 3 series de 12 repeticiones, con peso "
                        f"ligero y movimientos controlados."
                    )

        return respuesta_ej

    # ── Recomendación de COMIDA: generada por LLM con contexto real ──────────────
    import re as _re_reco
    from app.core.utils import get_peru_now as _get_peru_now_reco

    # 0. Detectar momento del día PRIMERO — necesario para filtrar candidatos KNN
    #    antes de pasarlos al LLM (evita sugerir ingredientes inapropiados por horario)
    # Orden de prioridad: CENA → MERIENDA → ALMUERZO → DESAYUNO
    # ALMUERZO va antes que DESAYUNO para que "ya entrené en la mañana, necesito almorzar"
    # matchee "almorzar" (ALMUERZO) antes de matchear "mañana" (DESAYUNO).
    # "tarde" solo en MERIENDA — "snack en la tarde" → MERIENDA, no ALMUERZO.
    # "noche"/"nocturno"/"madrugada" NO están aquí a propósito: son descriptores
    # de horario ambiguos (se pueden decir tanto a las 8pm como a la 1am) y se
    # resuelven más abajo consultando la hora real, en vez de forzar siempre
    # CENA/DESAYUNO sin importar qué hora es de verdad.
    # Incluye conjugaciones de primera persona ("ceno", "meriendo") además del
    # infinitivo/sustantivo — encontrado en pruebas reales: "qué ceno" caía al
    # horario del reloj (a las 16h daba rango de MERIENDA, 80-300 kcal) porque
    # "ceno" no es substring de "cena"/"cenar", ignorando lo que el usuario
    # pidió explícitamente.
    _MOMENTO_KEYWORDS_RECO = {
        "CENA":      ["cenar", "cena", "ceno", "cenare", "cenaré"],
        "MERIENDA":  ["merienda", "meriendo", "merendar", "snack", "media tarde", "media mañana", "antojo", "tarde"],
        "ALMUERZO":  ["almorzar", "almuerzo", "mediodía", "mediodia"],
        "DESAYUNO":  ["desayunar", "desayuno", "mañana"],
    }
    _msg_low_reco = mensaje.lower() if mensaje else ""

    # Si el usuario pidió explícitamente algo que choca con su propia condición
    # médica (ej. "queso" siendo intolerante a la lactosa), lo detectamos aquí
    # para poder justificar la respuesta en vez de devolver un "Opciones para
    # ti:" genérico que ignora el pedido sin explicación. Junta TODAS las
    # coincidencias del mensaje, no solo la primera (ej. "queso y algo dulce"
    # → debe mencionar ambas, no solo "queso").
    from app.services.recomendador_platos import _tokens_prohibidos, _CONDICION_TOKENS
    _condiciones_lista_reco = getattr(perfil, "medical_conditions", None) or []
    _tokens_dieta_msg = _tokens_prohibidos(_condiciones_lista_reco)

    _evitados_msg: list[tuple[str, str]] = []  # [(palabra, condición), ...]
    for _t in _tokens_dieta_msg:
        if _t in _msg_low_reco:
            _cond = next(
                (c for c in _condiciones_lista_reco if _t in _CONDICION_TOKENS.get(c, set())),
                None,
            )
            if _cond:
                _evitados_msg.append((_t, _cond))

    # Sinónimos genéricos: "dulce"/"postre" no son un ingrediente literal, son
    # una CATEGORÍA — pero si el cliente tiene Diabetes, significan lo mismo
    # que "azúcar" para este propósito. Lista corta de categorías comunes,
    # no de alimentos puntuales (eso seguiría siendo hardcoding de verdad).
    _SINONIMOS_CATEGORIA_DIETA = {
        "dulce": ("Diabetes", "azúcar"),
        "postre": ("Diabetes", "azúcar"),
        "azucarado": ("Diabetes", "azúcar"),
        "lacteo": ("Intolerancia a la Lactosa", "lácteos"),
        "lácteo": ("Intolerancia a la Lactosa", "lácteos"),
        "lacteos": ("Intolerancia a la Lactosa", "lácteos"),
        "lácteos": ("Intolerancia a la Lactosa", "lácteos"),
    }
    _conds_ya_cubiertas = {c for _, c in _evitados_msg}
    for _palabra_generica, (_cond_generica, _termino_mostrar) in _SINONIMOS_CATEGORIA_DIETA.items():
        if (
            _palabra_generica in _msg_low_reco
            and _cond_generica in _condiciones_lista_reco
            and _cond_generica not in _conds_ya_cubiertas
        ):
            _evitados_msg.append((_termino_mostrar, _cond_generica))
            _conds_ya_cubiertas.add(_cond_generica)

    # Texto final: "queso" o "queso y azúcar" o "queso, azúcar y lácteos"
    _palabra_evitada_msg = (
        " y ".join([", ".join(p for p, _ in _evitados_msg[:-1]), _evitados_msg[-1][0]])
        if len(_evitados_msg) > 1
        else (_evitados_msg[0][0] if _evitados_msg else None)
    )
    _condicion_relevante_msg = (
        " y ".join(dict.fromkeys(c for _, c in _evitados_msg)) if _evitados_msg else None
    )

    momento_reco = None
    for _m_key, _kws in _MOMENTO_KEYWORDS_RECO.items():
        if any(kw in _msg_low_reco for kw in _kws):
            momento_reco = _m_key
            break

    # Descriptores de horario ambiguos: "noche"/"nocturno" puede ser las 8pm
    # (cena real) o la 1am (antojo ligero) — se resuelve con la hora real.
    # "madrugada" nunca es un desayuno completo, siempre antojo ligero.
    if not momento_reco and any(kw in _msg_low_reco for kw in ("noche", "nocturno")):
        _hora_noche = _get_peru_now_reco().hour
        momento_reco = "CENA" if 18 <= _hora_noche <= 21 else "MERIENDA"
    elif not momento_reco and "madrugada" in _msg_low_reco:
        momento_reco = "MERIENDA"

    if not momento_reco:
        # Rangos alineados con la fuente canónica inferir_momento_dia_peru()
        # (app/core/utils.py). La franja 22:00-04:59 (trasnoche) reutiliza
        # MERIENDA — a esa hora nadie quiere un plato de cena completo, sino
        # un antojo ligero. Antes caía en "CENA" y el LLM sugería platos
        # pesados de almuerzo sin que nada lo restringiera correctamente.
        _hora = _get_peru_now_reco().hour
        if 5 <= _hora <= 9:
            momento_reco = "DESAYUNO"
        elif 10 <= _hora <= 14:
            momento_reco = "ALMUERZO"
        elif 15 <= _hora <= 17:
            momento_reco = "MERIENDA"
        elif 18 <= _hora <= 21:
            momento_reco = "CENA"
        else:  # 22:00-04:59 — trasnoche / antojo nocturno
            momento_reco = "MERIENDA"

    # Si el usuario pidió un estilo de preparación (frito, guisado) que el
    # momento del día no permite (ej. "algo frito" a la hora de la merienda),
    # lo anotamos para explicarlo igual que con las condiciones médicas — no
    # es una restricción de salud, es de horario, pero merece la misma
    # transparencia en vez de ignorar el pedido sin decir nada.
    _ESTILOS_NO_PERMITIDOS_MOMENTO = {
        "DESAYUNO": ("frito", "frita", "guiso", "guisado"),
        "MERIENDA": ("frito", "frita", "guiso", "guisado"),
    }
    _estilo_evitado_momento = next(
        (
            e for e in _ESTILOS_NO_PERMITIDOS_MOMENTO.get(momento_reco, ())
            if e in _msg_low_reco
        ),
        None,
    )

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

            # Filtrar el ancla KNN por condiciones médicas/dietéticas (Vegano,
            # Vegetariano, Lactosa, Celíaco, Diabetes...) — sin esto, el KNN puede
            # anclar el Plato 1 en un alimento prohibido (ej. pescado para un
            # cliente vegano) y el LLM termina usándolo igual.
            from app.services.recomendador_platos import _tokens_prohibidos
            _tokens_dieta_reco_knn = _tokens_prohibidos(
                getattr(perfil, "medical_conditions", None) or []
            )
            if _tokens_dieta_reco_knn:
                _candidatos_knn = [
                    c for c in _candidatos_knn
                    if not any(t in c["alimento"].lower() for t in _tokens_dieta_reco_knn)
                ]
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

    # 1.5. Detectar Vegano/Vegetariano TEMPRANO — los ejemplos de plato por momento
    #      (paso 2) deben ser distintos si el cliente no come carne/pescado/lácteos/huevo.
    _condiciones_list_reco = getattr(perfil, "medical_conditions", None) or []
    _condiciones_str_reco = " ".join(_condiciones_list_reco).lower()
    es_vegano_reco = (
        "vegano" in dieta.lower() or "vegetariano" in dieta.lower()
        or "vegano" in _condiciones_str_reco or "vegetariano" in _condiciones_str_reco
    )

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
    # Variante Vegano/Vegetariano: mismos rangos, ejemplos sin carne/pescado/lácteos/huevo.
    _RESTRICCIONES_MOMENTO_VEGANO = {
        "DESAYUNO": (
            "Rango: 250-450 kcal. Primera comida del día, rápida y simple. "
            "Típico vegano/vegetariano: avena con leche vegetal, pan con palta, "
            "quinua con leche vegetal, fruta con granola, tostadas con mermelada. "
            "⛔ PROHIBIDO: sopas, chupes, caldos, pescado, carnes, huevos, lácteos animales, cebiches, causas."
        ),
        "ALMUERZO": (
            "Rango: 550-850 kcal — porción real de adulto, MÍNIMO 550 kcal. "
            "Plato de fondo vegano/vegetariano conocido: menestra con arroz, tallarines con verduras, "
            "arroz con lentejas, quinua con verduras salteadas, tofu salteado con verduras. "
            "⛔ PROHIBIDO carnes, pescado, mariscos, huevos, lácteos animales. "
            "No repitas la misma proteína vegetal en los 3 platos."
        ),
        "CENA": (
            "Rango: 200-520 kcal. Plato ligero para la noche: "
            "sopa de verduras, ensalada con menestra o tofu, crema de zapallo, menestra sencilla. "
            "⛔ PROHIBIDO pescado, carnes, huevos, lácteos animales. Evita frituras y guisos pesados."
        ),
        "MERIENDA": (
            "Rango: 80-280 kcal. Refrigerio rápido sin cocción elaborada. "
            "Válido: fruta, pan con palta, frutos secos, batido con leche vegetal. "
            "⛔ PROHIBIDO: pescado, mariscos, carnes, huevos, lácteos animales, causas, cebiches, arroces, guisos."
        ),
    }
    restricciones_momento_reco = (
        _RESTRICCIONES_MOMENTO_VEGANO if es_vegano_reco else _RESTRICCIONES_MOMENTO_RECO
    ).get(momento_reco, "")

    # 3. Detectar preferencia de ingrediente específico en el mensaje
    # Tope de 2 palabras (no 25 caracteres libres) — sin esto, "con quinua
    # para el almuerzo" capturaba la frase completa en vez de solo "quinua",
    # y el LLM recibía una instrucción imposible de cumplir.
    _ing_match = _re_reco.search(
        r'(?:con|de|que\s+tenga|a\s+base\s+de)\s+([a-záéíóúüñ]+(?:\s+[a-záéíóúüñ]+)?)'
        r'(?=\s+(?:para|en|hoy|ahora|al|por)\b|[.,?]|$)',
        _msg_low_reco,
    )
    pref_ingrediente_reco = ""
    if _ing_match:
        _ing_detectado = _ing_match.group(1).strip().rstrip('.,?')
        _PALABRAS_IGNORAR = {"hoy", "comer", "ti", "mi", "algo", "uno", "plato", "poco"}
        # Encontrado en pruebas reales: "qué como ... si tengo dolor DE rodilla"
        # capturaba "rodilla" como ingrediente pedido (el patrón "de + palabra"
        # no distingue comida de cuerpo/lesión) y lo inyectaba literalmente en
        # los nombres de plato ("Lomo Saltado con Rodilla"). _TEMA_EJERCICIO_KW
        # ya lista body parts/lesión — ninguna de esas palabras es un ingrediente.
        _palabras_ing = set(_ing_detectado.split())
        _es_no_comida = bool(_palabras_ing & set(_TEMA_EJERCICIO_KW))
        if _ing_detectado not in _PALABRAS_IGNORAR and not _es_no_comida and len(_ing_detectado) > 2:
            pref_ingrediente_reco = (
                f"⚠️ El usuario pidió ESPECÍFICAMENTE algo con: **{_ing_detectado}**. "
                f"Esto tiene prioridad sobre la variedad: los 3 platos DEBEN incluir "
                f"'{_ing_detectado}' de alguna forma (como ingrediente principal o "
                f"visible en la preparación) — no lo menciones en uno solo y dejes "
                f"los otros 2 libres."
            )

    # 3.2. Detectar NEGACIÓN/exclusión puntual en el mensaje ("no quiero comer
    # carne hoy") — encontrado en pruebas reales: el motor KNN seguía
    # recomendando "Arroz con Pollo" pese a la negación explícita, porque no
    # existía ningún mecanismo que la detectara (la negación no es una
    # condición médica guardada en el perfil, es una preferencia puntual del
    # mensaje). Reutiliza los mismos tokens por categoría que ya existen para
    # Vegano/Vegetariano/Lactosa/Celíaco/Diabetes — no se inventa una lista
    # nueva de alimentos, solo se reusa la ya construida.
    from app.services.recomendador_platos import _CONDICION_TOKENS as _COND_TOKENS_NEG
    _NEG_CATEGORIA_A_TOKENS = {
        "carne": _COND_TOKENS_NEG["Vegetariano"],
        "carnes": _COND_TOKENS_NEG["Vegetariano"],
        "pescado": {"pescado", "salmon", "salmón", "atun", "atún", "trucha",
                     "caballa", "corvina", "cachema", "lisa", "mero", "tollo", "anchoveta"},
        "mariscos": {"mariscos", "camaron", "camarón", "langostino", "pulpo", "calamar"},
        "lacteos": _COND_TOKENS_NEG["Intolerancia a la Lactosa"],
        "lácteos": _COND_TOKENS_NEG["Intolerancia a la Lactosa"],
        "gluten": _COND_TOKENS_NEG["Celíaco"],
        "dulce": _COND_TOKENS_NEG["Diabetes"],
        "azucar": _COND_TOKENS_NEG["Diabetes"],
        "azúcar": _COND_TOKENS_NEG["Diabetes"],
    }
    # Captura de 1 sola palabra (no 2 como en pref_ingrediente_reco) — encontrado
    # en pruebas reales: "no quiero comer carne HOY, que almuerzo" capturaba
    # "carne hoy" (2 palabras) porque "hoy" quedaba atrapado en el grupo
    # opcional antes de que el lookahead lo detectara como filler. Las
    # categorías negadas (carne/pescado/lácteos/gluten/dulce) son casi siempre
    # 1 palabra, así que se prioriza fiabilidad sobre cobertura de frases largas.
    _neg_match = _re_reco.search(
        r'\bno\s+(?:quiero|puedo|deseo)\s+(?:comer\s+)?([a-záéíóúüñ]+)'
        r'(?=\s+(?:hoy|ahora|por\s+favor)\b|[.,?]|$)',
        _msg_low_reco,
    )
    exclusion_reco = ""
    _tokens_exclusion_msg: set[str] = set()
    if _neg_match:
        _excl_detectado = _neg_match.group(1).strip().rstrip('.,?')
        if len(_excl_detectado) > 2:
            _tokens_exclusion_msg = _NEG_CATEGORIA_A_TOKENS.get(_excl_detectado, {_excl_detectado})
            exclusion_reco = (
                f"⚠️ El usuario dijo explícitamente que NO quiere comer: **{_excl_detectado}**. "
                f"Ninguno de los 3 platos debe contener esto ni sus variantes/derivados obvios."
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

    # 4. Restricción de dieta (es_vegano_reco ya se calculó en el paso 1.5)
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
                    f"(ej. no pongas leche en una menestra o sopa de verduras).\n"
                    f"🚨 PRIORIDAD ABSOLUTA: estas restricciones médicas pesan MÁS que cualquier "
                    f"ejemplo de plato mencionado antes en este mensaje (por momento del día, "
                    f"estilo o ingrediente sugerido). Si algún ejemplo anterior contradice esta "
                    f"lista, IGNÓRALO POR COMPLETO y elige otra opción real y conocida que sí cumpla.\n\n"
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
            f"INSPIRACIÓN NUTRICIONAL (sutil, NO obligatoria): el alimento "
            f"'{_alim_knn}' (~{_kcal_knn:.0f} kcal/100g) tiene un perfil afín al "
            f"déficit actual. Si encaja de forma natural con un plato conocido "
            f"y común en Perú, puedes considerarlo como inspiración para UNO "
            f"de los 3 platos — no necesariamente el primero, y no siempre el "
            f"mismo tipo de proteína. Prioriza variedad real entre los 3 "
            f"platos por encima de esta sugerencia; ignórala libremente si no "
            f"aporta variedad.\n\n"
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
        + (
            (
                f"Si sugieres pescado, usa especies de Lambayeque (Caballa, Lisa, Mero, Tollo).\n"
                f"⛔ SEMÁNTICA: Caballa, Lisa, Mero, Tollo son PESCADOS — nunca son 'mariscos'. "
                f"No escribas 'Mariscos de Caballa' ni 'Mariscos de Lisa' — son categorías distintas. "
                f"Di 'Arroz con Caballa' O 'Arroz con Mariscos', nunca ambos combinados.\n\n"
            ) if not es_vegano_reco else "\n"
        )
        + (f"{_masa_muscular_txt}\n\n" if _masa_muscular_txt else "")
        + (f"{objetivo_proteina_reco}\n\n" if objetivo_proteina_reco else "")
        + (f"PREFERENCIA: {pref_ingrediente_reco}\n\n" if pref_ingrediente_reco else "")
        + (f"EXCLUSIÓN: {exclusion_reco}\n\n" if exclusion_reco else "")
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

    # Guard: si el LLM ignoró una restricción médica/dietética (Vegano, Vegetariano,
    # Lactosa, Celíaco, Diabetes...), reintentar UNA vez. Dos chequeos complementarios:
    #   1. Palabra clave literal (rápido, sin costo de API) — detecta "pollo", "leche", etc.
    #   2. Juicio de Groq (sin hardcoding) — detecta platos que violan la condición aunque
    #      el nombre no contenga la palabra prohibida (ej. "Picarones" para un diabético:
    #      Groq sabe que llevan miel sin que tengamos que mantener una lista de postres).
    from app.services.recomendador_platos import _tokens_prohibidos, _detectar_dieta_en_mensaje
    _condiciones_dieta_check = list(getattr(perfil, "medical_conditions", None) or [])
    # Sumar restricciones dichas en el mensaje actual aunque no estén en el
    # perfil — ej. "Soy vegano y me duele la rodilla..." nunca se detectaba
    # porque "Vegano" no vivía en medical_conditions de este usuario.
    for _cond_msg in _detectar_dieta_en_mensaje(mensaje):
        if _cond_msg not in _condiciones_dieta_check:
            _condiciones_dieta_check.append(_cond_msg)
    _tokens_dieta_check = _tokens_prohibidos(_condiciones_dieta_check) | _tokens_exclusion_msg

    # Calificadores que vuelven SEGURO un alimento normalmente prohibido
    # (ej. "queso deslactosado" no debe disparar el filtro de Lactosa).
    # Se evalúa por línea/plato — si la misma línea trae el calificador, no cuenta.
    _CALIFICADORES_SEGUROS_DIETA = (
        "deslactosado", "deslactosada", "sin lactosa", "sin azúcar", "sin azucar",
        "sin gluten", "light", "diet",
    )

    def _linea_viola_dieta(linea: str) -> bool:
        _linea_low = linea.lower()
        if any(c in _linea_low for c in _CALIFICADORES_SEGUROS_DIETA):
            return False
        return any(t in _linea_low for t in _tokens_dieta_check)

    _viola_token = bool(_tokens_dieta_check) and any(
        _linea_viola_dieta(linea) for linea in (respuesta_llm_reco or "").split("\n")
    )

    _viola_juicio_groq = False
    if not _viola_token and condiciones and condiciones.lower() != "ninguna":
        try:
            _prompt_validacion_dieta = (
                f"Eres nutricionista clínico ESTRICTO. Condiciones del paciente: {condiciones}.\n"
                f"Platos propuestos:\n{respuesta_llm_reco}\n"
                f"Revisa cada plato UNO POR UNO, pensando en la receta tradicional completa, "
                f"no solo en las palabras del nombre. Postres y dulces tradicionales peruanos "
                f"(picarones, mazamorra, alfajor, suspiro, cocada, tres leches, turrón, etc.) "
                f"SIEMPRE llevan azúcar o miel aunque el nombre no lo diga — son inadecuados "
                f"para Diabetes. Quesos/lácteos sin la palabra 'deslactosado' son inadecuados "
                f"para Intolerancia a la Lactosa.\n"
                f"¿Hay AL MENOS UN plato inadecuado para alguna de las condiciones del paciente? "
                f"Responde SOLO 'SI' o 'NO'."
            )
            _resp_validacion = await ia_engine._llamar_groq(
                _prompt_validacion_dieta, max_tokens=5, temp=0.0
            )
            _viola_juicio_groq = bool(_resp_validacion) and _resp_validacion.strip().lower().startswith("si")
        except Exception as _e_val:
            logger.warning("[Reco] Validación dietética con Groq falló: %s", _e_val)

    if _viola_token or _viola_juicio_groq:
        # Capturar EXACTAMENTE qué palabras dispararon el chequeo, para que el
        # reintento las excluya por nombre — un aviso genérico ("evita lo que
        # corresponda") no es suficiente, el LLM tiende a repetir el mismo plato.
        _tokens_detectados = sorted({
            t for linea in (respuesta_llm_reco or "").split("\n")
            for t in _tokens_dieta_check
            if t in linea.lower() and not any(
                c in linea.lower() for c in _CALIFICADORES_SEGUROS_DIETA
            )
        })
        logger.warning(
            "[Reco] Plato inadecuado por condición médica (token=%s detectados=%s, groq=%s) — reintentando",
            _viola_token, _tokens_detectados, _viola_juicio_groq,
        )
        _restriccion_explicita = (
            f"NO uses NINGUNO de estos ingredientes/platos, ni variantes: "
            f"{', '.join(_tokens_detectados)}.\n"
            if _tokens_detectados else ""
        )
        _prompt_retry_dieta = (
            f"Lista EXACTAMENTE 3 platos de {momento_reco.lower()} peruanos "
            f"({round(restante)} kcal disponibles) apropiados para un paciente con: {condiciones}.\n"
            f"{_restriccion_explicita}"
            f"Piensa como nutricionista clínico: evita además cualquier otro ingrediente o "
            f"receta tradicional incompatible con esas condiciones, incluso si el nombre del "
            f"plato no lo menciona explícitamente. Verifica cada plato dos veces.\n"
            f"SOLO este formato:\n"
            f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
            f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
            f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
            f"Sin frases extra, sin ingredientes, sin pasos."
        )
        respuesta_llm_reco = await ia_engine._llamar_groq(_prompt_retry_dieta, max_tokens=150, temp=0.1)

        # Garantía final: si el reintento TAMBIÉN viola (el LLM puede introducir
        # una violación nueva al regenerar), no se intenta una 3ra vez sin certeza
        # — se usa un fallback determinista 100% seguro (sin carne/pescado/lácteos/
        # gluten/azúcar, válido para cualquiera de las 5 condiciones principales).
        _aun_viola = bool(_tokens_dieta_check) and any(
            _linea_viola_dieta(linea) for linea in (respuesta_llm_reco or "").split("\n")
        )
        if _aun_viola:
            logger.warning(
                "[Reco] Reintento también violó la condición médica — usando fallback seguro garantizado"
            )
            _FALLBACK_SEGURO_MOMENTO = {
                "DESAYUNO": (
                    "- Fruta sola (~80 kcal, P:1g C:20g G:0g)\n"
                    "- Avena con agua (~150 kcal, P:5g C:27g G:3g)\n"
                    "- Tostada con palta (~120 kcal, P:3g C:15g G:7g)"
                ),
                "ALMUERZO": (
                    "- Menestra sencilla con verduras (~350 kcal, P:18g C:55g G:5g)\n"
                    "- Sopa de verduras con quinua (~300 kcal, P:10g C:50g G:5g)\n"
                    "- Ensalada de verduras con palta (~280 kcal, P:5g C:30g G:15g)"
                ),
                "CENA": (
                    "- Sopa de verduras (~180 kcal, P:6g C:25g G:4g)\n"
                    "- Ensalada de verduras con palta (~220 kcal, P:4g C:20g G:14g)\n"
                    "- Menestra ligera (~250 kcal, P:14g C:35g G:4g)"
                ),
                "MERIENDA": (
                    "- Fruta sola (~80 kcal, P:1g C:20g G:0g)\n"
                    "- Puñado de frutos secos (~150 kcal, P:5g C:8g G:12g)\n"
                    "- Tostada con palta (~120 kcal, P:3g C:15g G:7g)"
                ),
            }
            respuesta_llm_reco = _FALLBACK_SEGURO_MOMENTO.get(
                momento_reco, _FALLBACK_SEGURO_MOMENTO["ALMUERZO"]
            )

    # Guard de ingrediente pedido explícitamente ("qué puedo comer con palta",
    # "algo con quinua") — encontrado en pruebas reales: la instrucción del
    # prompt ("al menos 1 de 3") no siempre se respetaba (0/3 platos con
    # quinua en una prueba). Verificación de código + reintento dirigido,
    # mismo patrón que el guard dietético de arriba.
    if pref_ingrediente_reco:
        _ing_norm = _normalizar_nombre(_ing_detectado)
        _lineas_reco = [l for l in (respuesta_llm_reco or "").split("\n") if l.strip()]
        # Por LÍNEA (plato), no "en algún lugar del texto" — si solo 1 de 3
        # platos lo menciona, sigue sin cumplir lo que el usuario pidió.
        _n_con_ingrediente = sum(1 for l in _lineas_reco if _ing_norm in _normalizar_nombre(l))
        _tiene_ingrediente = bool(_lineas_reco) and _n_con_ingrediente == len(_lineas_reco)
        if not _tiene_ingrediente:
            logger.warning(
                "[Reco] Ingrediente pedido '%s' ausente de los 3 platos — reintentando",
                _ing_detectado,
            )
            _prompt_retry_ing = (
                f"Lista EXACTAMENTE 3 platos de {momento_reco.lower()} peruanos "
                f"({round(restante)} kcal disponibles) que incluyan '{_ing_detectado}' "
                f"como ingrediente — LOS 3, no solo uno. Si '{_ing_detectado}' no calza "
                f"de forma natural en un plato de fondo, inclúyelo como acompañamiento "
                f"o guarnición de ese plato.\n"
                f"SOLO este formato:\n"
                f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
                f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
                f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
                f"Sin frases extra, sin ingredientes, sin pasos."
            )
            respuesta_llm_reco = await ia_engine._llamar_groq(
                _prompt_retry_ing, max_tokens=150, temp=0.2
            )
            # Segundo intento si el primero tampoco lo logró (ingredientes
            # "difíciles" como palta/quinua en platos de fondo tradicionales):
            # instrucción más explícita, indicando CÓMO encajarlo si no calza
            # como ingrediente principal.
            _lineas_retry1 = [l for l in (respuesta_llm_reco or "").split("\n") if l.strip()]
            _n_retry1 = sum(1 for l in _lineas_retry1 if _ing_norm in _normalizar_nombre(l))
            if not (_lineas_retry1 and _n_retry1 == len(_lineas_retry1)):
                logger.warning(
                    "[Reco] Reintento 1 tampoco logró '%s' en los 3 platos — 2do reintento",
                    _ing_detectado,
                )
                _prompt_retry_ing2 = (
                    f"Lista EXACTAMENTE 3 platos de {momento_reco.lower()} peruanos "
                    f"({round(restante)} kcal disponibles). REGLA OBLIGATORIA: cada uno "
                    f"de los 3 nombres de plato debe mencionar literalmente la palabra "
                    f"'{_ing_detectado}' — agrégala como guarnición/acompañamiento si no "
                    f"es el ingrediente principal (ej. 'Lomo Saltado con {_ing_detectado}', "
                    f"'Ensalada de {_ing_detectado}', 'Sopa con {_ing_detectado}').\n"
                    f"SOLO este formato:\n"
                    f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
                    f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
                    f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
                    f"Sin frases extra."
                )
                respuesta_llm_reco = await ia_engine._llamar_groq(
                    _prompt_retry_ing2, max_tokens=150, temp=0.2
                )
            # No hay fallback determinista aquí (no se puede inventar un plato
            # con un ingrediente arbitrario sin hardcodear) — si ningún
            # reintento lo logra, se acepta el resultado igual, mejor que
            # cortar la recomendación por completo.

    # Guard de coherencia culinaria: el LLM a veces combina palabras reales
    # de forma inventada (ej. "Pachamanca de quinoa" — la pachamanca es con
    # carnes, nunca de quinoa; "Lomo saltado de pescado" — el lomo saltado es
    # de carne/pollo). Sin hardcodear nombres de platos, le pedimos a Groq que
    # juzgue con su propio conocimiento culinario si algo suena inventado.
    try:
        _prompt_coherencia = (
            f"Eres experto en gastronomía peruana. Te paso 3 platos:\n"
            f"{respuesta_llm_reco}\n"
            f"Analiza CADA plato uno por uno, en voz alta, antes de concluir:\n"
            f"Plato 1: ¿es una receta real que existe tal cual en la cocina "
            f"peruana? ¿Sí o no, y por qué?\n"
            f"Plato 2: lo mismo.\n"
            f"Plato 3: lo mismo.\n"
            f"Sé ESTRICTO: muchos platos peruanos tienen un ingrediente "
            f"principal FIJO por tradición (ej. la pachamanca SIEMPRE es con "
            f"carnes — pollo, cerdo, cordero — JAMÁS con quinua o verduras "
            f"solas; el lomo saltado SIEMPRE es con carne de res o pollo, "
            f"JAMÁS con pescado). Si un plato cambia ese ingrediente fijo por "
            f"otro, NO es una receta real, es una combinación inventada.\n"
            f"Termina tu respuesta exactamente con la palabra SI (si al menos "
            f"un plato es inventado) o NO (si los 3 son reales), en la última línea."
        )
        _resp_coherencia = await ia_engine._llamar_groq(
            _prompt_coherencia, max_tokens=220, temp=0.0
        )
        _ultima_linea_coherencia = (
            (_resp_coherencia or "").strip().splitlines()[-1].strip().lower().rstrip(".")
            if _resp_coherencia else ""
        )
        _incoherente = _ultima_linea_coherencia in ("si", "sí")
    except Exception as _e_coh:
        _incoherente = False
        logger.warning("[Reco] Validación de coherencia culinaria falló: %s", _e_coh)

    if _incoherente:
        logger.warning("[Reco] Plato con combinación inventada detectado — reintentando")
        _prompt_retry_coherencia = (
            f"Lista EXACTAMENTE 3 platos de {momento_reco.lower()} peruanos "
            f"({round(restante)} kcal disponibles) que sean REALES y "
            f"conocidos en la gastronomía peruana — NO inventes combinaciones "
            f"nuevas de ingredientes que no se preparan juntos tradicionalmente. "
            f"Usa solo platos que cualquier peruano reconocería de inmediato.\n"
            f"SOLO este formato:\n"
            f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
            f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
            f"- Nombre del plato (~XXX kcal, P:Xg C:Yg G:Zg)\n"
            f"Sin frases extra, sin ingredientes, sin pasos."
        )
        respuesta_llm_reco = await ia_engine._llamar_groq(
            _prompt_retry_coherencia, max_tokens=150, temp=0.2
        )

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
        return _construir_mensaje_natural_reco(
            _platos_limpios, _palabra_evitada_msg, _condicion_relevante_msg, _estilo_evitado_momento
        )

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
        return _construir_mensaje_natural_reco(
            _platos_limpios, _palabra_evitada_msg, _condicion_relevante_msg, _estilo_evitado_momento
        )

    # 7. Fallback: retornar respuesta del LLM tal como vino
    return respuesta_llm_reco or "No pude generar recomendaciones en este momento."


# Gestor de contexto liviano para la conversación libre: decide qué parte del
# perfil es relevante según el tema reciente, en vez de mandar todo siempre y
# confiar en que el LLM ignore lo que no aplica (eso fallaba con frecuencia).
_TEMA_EJERCICIO_KW = (
    "ejercicio", "entren", "gym", "gimnasio", "rutina", "lesion", "lesión",
    "dolor", "rodilla", "espalda", "hombro", "codo", "muscul", "correr",
    "trotar", "nadar", "pesas", "cardio", "estiramiento", "pecho", "pierna",
    "piernas", "bicep", "bícep", "tricep", "trícep", "abdomen", "core",
)
_TEMA_NUTRICION_KW = (
    "comida", "comer", "comí", "comi ", "almuerzo", "cena", "desayuno",
    "dieta", "kcal", "caloria", "caloría", "macros", "proteina", "proteína",
    "carbohidrato", "grasa", "nutricion", "nutrición", "alimento", "plato",
    "merienda", "postre",
)
_PALABRAS_LESION_GENERICA = (
    "lesion", "lesión", "dolor", "molestia", "me duele", "me lastime", "me lastimé",
)


def _lesion_mencionada_sin_tipo(mensaje: str, historial: list) -> bool:
    """True si se menciona una lesión/dolor de forma genérica (sin decir
    rodilla/espalda/hombro/codo) — no hay suficiente información para un
    consejo de ejercicio seguro, así que hay que preguntar antes de generar."""
    _texto = (mensaje or "").lower()
    if historial:
        # Solo turnos del USUARIO — las respuestas del asistente usan palabras
        # como "evitar lesiones" o "pierna lesionada" como lenguaje genérico de
        # seguridad, no como el usuario reportando una lesión nueva. Mezclarlas
        # causaba falsos positivos: "Comí pollo con quinua" disparaba "¿qué
        # lesión tienes?" porque la respuesta anterior del asistente decía
        # "evitar lesiones", aunque el usuario sí había nombrado la zona antes.
        _turnos_usuario = [
            str(h.get("content", "")) for h in historial if h.get("role") == "user"
        ]
        _texto += " " + " ".join(_turnos_usuario[-4:]).lower()
    if not any(p in _texto for p in _PALABRAS_LESION_GENERICA):
        return False
    from app.services.rutina_service import _detectar_lesiones
    return len(_detectar_lesiones([_texto])) == 0


_PALABRAS_PETICION_AMBIGUA = (
    "dame un consejo", "dame consejo", "dame una recomendacion", "dame algo",
    "ayudame", "ayúdame", "que hago", "qué hago", "y entonces", "y ahora que",
    "y ahora qué", "recomiendame algo", "recomiéndame algo",
)


def _es_peticion_ambigua(mensaje: str) -> bool:
    """True si el mensaje pide ayuda/consejo de forma genérica, sin nombrar
    ningún tema — solo se llama cuando _detectar_tema_chat ya devolvió
    'general' (es decir, ni el mensaje ni el historial reciente tienen señal
    de nutrición/ejercicio), así que no se dispara con mensajes que sí tienen
    contexto claro."""
    _low = (mensaje or "").lower().strip()
    return any(p in _low for p in _PALABRAS_PETICION_AMBIGUA)


def _detectar_tema_chat(mensaje: str, historial: list) -> str:
    """'ejercicio' | 'nutricion' | 'general' — según el mensaje actual y los
    últimos 2 turnos. Si hay señales de ambos temas a la vez, se prefiere
    incluir todo el contexto (más seguro que omitir algo relevante)."""
    _texto = (mensaje or "").lower()
    if historial:
        _texto += " " + " ".join(str(h.get("content", "")) for h in historial[-2:]).lower()
    _es_ejercicio = any(k in _texto for k in _TEMA_EJERCICIO_KW)
    _es_nutricion = any(k in _texto for k in _TEMA_NUTRICION_KW)
    if _es_ejercicio and not _es_nutricion:
        return "ejercicio"
    if _es_nutricion and not _es_ejercicio:
        return "nutricion"
    return "general"


async def validate_and_retry(
    respuesta: str,
    ia_engine,
    es_invalida,
    construir_prompt_retry,
    fallback: str,
    max_tokens_retry: int = 200,
    temp_retry: float = 0.3,
) -> str:
    """
    Ciclo genérico de control de calidad para respuestas de LLM:
    generar → validar → reintentar con instrucción específica → respaldo
    garantizado si el reintento también falla.

    Formaliza el patrón que se repitió 5 veces hoy en este archivo (dieta,
    coherencia culinaria, lesión×2, formato) en una sola función reusable —
    en vez de copiar el bloque de detectar/reintentar/respaldar cada vez que
    aparece un caso nuevo.

    ``es_invalida(texto) -> bool``        — qué cuenta como violación.
    ``construir_prompt_retry() -> str``   — prompt del reintento (closure con
                                             el contexto que necesite).
    ``fallback``                          — texto garantizado si el reintento
                                             también falla (nunca None).
    """
    if not es_invalida(respuesta):
        return respuesta

    _prompt_retry = construir_prompt_retry()
    _respuesta_retry = await ia_engine._llamar_groq(
        _prompt_retry, max_tokens=max_tokens_retry, temp=temp_retry
    )
    if es_invalida(_respuesta_retry or ""):
        return fallback
    return _respuesta_retry


def _repite_mensaje_usuario(respuesta: str, mensaje: str) -> bool:
    """True si la respuesta empieza citando/repitiendo el mensaje del usuario."""
    _msg_clean = (mensaje or "").strip().lower().rstrip("?.!¿¡")
    if len(_msg_clean) < 5:
        return False
    _prefijo = _msg_clean[:min(len(_msg_clean), 20)]
    return respuesta.strip().lower().startswith(_prefijo)


def _menciona_tema_no_relacionado(respuesta: str, tema: str) -> bool:
    """True si, dado el tema activo, la respuesta menciona el tema contrario
    sin que nadie lo haya pedido (ej. hablar de dieta en una pregunta de
    lesión, o de ejercicio en una pregunta de nutrición)."""
    _low = respuesta.lower()
    if tema == "ejercicio":
        return any(k in _low for k in _TEMA_NUTRICION_KW)
    if tema == "nutricion":
        return any(k in _low for k in _TEMA_EJERCICIO_KW)
    return False


_FRASES_CAUTELA_IMPACTO = (
    "evita", "evitar", "no es recomendable", "sin impacto", "bajo impacto",
    "no te recomiendo", "no deberías", "no deberias",
)
_ACCIONES_IMPACTO = (
    "trota", "trotar", "trote", "correr", "corre ", "saltar", "salto",
    "sprint", "saltos",
)


def _tiene_contradiccion_impacto(respuesta: str) -> bool:
    """True si la respuesta dice 'evita el impacto' y en la MISMA respuesta
    igual sugiere correr/trotar/saltar — contradicción interna, no mezcla de
    tema. Distinto de _menciona_tema_no_relacionado: aquí el tema SÍ es
    correcto (ejercicio), el problema es que se contradice a sí misma."""
    _low = respuesta.lower()
    _tiene_cautela = any(f in _low for f in _FRASES_CAUTELA_IMPACTO)
    _tiene_impacto = any(a in _low for a in _ACCIONES_IMPACTO)
    return _tiene_cautela and _tiene_impacto


def _filtrar_resultado_chat(texto: str, tema: str) -> str:
    """Respaldo determinista: en vez de un mensaje genérico de disculpa,
    conserva las oraciones del LLM que SÍ están en tema y descarta las que
    se desviaron — aprovecha lo bueno que ya generó en vez de descartarlo todo."""
    import re as _re_filtro
    _oraciones = _re_filtro.split(r'(?<!\d)(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])', texto.strip())
    _oraciones = [o.strip() for o in _oraciones if o.strip()]
    _kw_evitar = _TEMA_NUTRICION_KW if tema == "ejercicio" else (
        _TEMA_EJERCICIO_KW if tema == "nutricion" else ()
    )
    _filtradas = [o for o in _oraciones if not any(k in o.lower() for k in _kw_evitar)]
    if not _filtradas:
        return "Cuéntame un poco más para poder ayudarte mejor con eso."
    _resultado = " ".join(_filtradas).strip()
    if _resultado and _resultado[-1] not in ".!":
        _resultado += "."
    return _resultado


async def respuesta_chat_llm(
    mensaje: str,
    perfil,
    consumido: float,
    meta: float,
    quemado: float,
    historial: list,
    ia_engine,
    plan_macros: dict = None,
) -> str:
    """Respuesta conversacional corta vía LLM."""
    pct = round(consumido / meta * 100) if meta > 0 else 0
    hist_txt = "\n".join(
        f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content'][:120]}"
        for m in historial[-4:]
    ) or "(inicio de conversación)"

    objetivo = getattr(perfil, "goal", "mantener peso") or "mantener peso"
    dieta = getattr(perfil, "diet_type", "Normal") or "Normal"
    condiciones = ", ".join(getattr(perfil, "medical_conditions", None) or []) or "ninguna"

    # Detector de información insuficiente: si se menciona una lesión/dolor
    # de forma genérica (sin decir rodilla/espalda/hombro/codo), no hay datos
    # suficientes para dar un consejo de ejercicio seguro — no es que el LLM
    # "desobedezca", es que falta la variable que decide qué es seguro o no.
    # Cortar aquí, sin llamar al LLM, y preguntar antes de recomendar nada.
    if _lesion_mencionada_sin_tipo(mensaje, historial):
        return (
            "¿Qué lesión tienes exactamente? ¿Es en la rodilla, espalda, hombro, "
            "codo u otra zona? Así te doy un consejo seguro y específico."
        )

    # Gestor de contexto: solo se incluye dieta/kcal/condiciones si el tema
    # reciente es de comida/nutrición. Si es de ejercicio/lesión, se omite —
    # así el código decide qué es relevante en vez de pedirle al LLM que
    # "ignore" datos irrelevantes (eso fallaba ~50% de las veces en pruebas).
    _tema_chat = _detectar_tema_chat(mensaje, historial)

    # Intención desconocida: si NI el mensaje NI el historial reciente tienen
    # señal de tema (_detectar_tema_chat ya devolvió "general" considerando
    # ambos), y el mensaje es una petición vaga de ayuda/consejo, preguntar en
    # vez de inventar un tema. Calibrado para NO dispararse cuando sí hay señal
    # clara (ej. "tengo dolor de rodilla" ya cae en tema "ejercicio", no aquí).
    if _tema_chat == "general" and _es_peticion_ambigua(mensaje):
        return (
            "¿Sobre qué tema necesitas ayuda? Puedo ayudarte con nutrición, "
            "ejercicio o seguimiento de tu progreso."
        )

    if _tema_chat == "ejercicio":
        # Sin "Objetivo" aquí a propósito: hasta "mantener peso" por sí solo
        # bastaba para que el LLM se desviara a hablar de dieta sin que se
        # preguntara — en temas de lesión/ejercicio, el nombre alcanza.
        bloque_perfil = f"- Nombre: {perfil.first_name}"
    else:
        _peso = getattr(perfil, "weight", None)
        _meta_prot = (plan_macros or {}).get("proteinas_g")
        bloque_perfil = (
            f"- Nombre: {perfil.first_name}\n"
            f"- {round(consumido)}/{round(meta)} kcal consumidas ({pct}%)  |  {round(quemado)} kcal quemadas hoy\n"
            f"- Dieta: {dieta}  |  Condiciones: {condiciones}  |  Objetivo: {objetivo}"
            + (f"  |  Peso: {_peso:.0f} kg" if _peso else "")
            # Meta real de proteína del plan calculado — fuente única de verdad,
            # evita que el LLM recalcule con una fórmula g/kg genérica que puede
            # quedar muy por debajo del valor real del plan (visto en prod: plan
            # real 224g vs fórmula genérica 80-110g para el mismo usuario).
            + (f"  |  Meta proteína (plan): {round(_meta_prot)} g" if _meta_prot else "")
        )

    prompt = _PROMPT_CHAT.format(
        bloque_perfil=bloque_perfil,
        historial=hist_txt,
        mensaje=mensaje,
    )
    # ── Intercept "qué hora es" — hora real de Perú, sin pasar por el LLM ────────
    _m_lower_hora = mensaje.lower().strip()
    if any(k in _m_lower_hora for k in ("qué hora es", "que hora es", "qué hora son", "que hora son")):
        from app.core.utils import get_peru_now
        return f"Son las {get_peru_now().strftime('%H:%M')} (hora de Perú)."

    # ── Intercept "cómo uso la app / cómo registro mi comida/ejercicio" ──────────
    # Respuesta fija, sin pasar por el LLM: probado que el modelo inventaba
    # botones y pantallas que no existen (ej. recomendó apps externas como
    # MyFitnessPal, o describió un "botón Registrar comida" inexistente) —
    # para una pregunta de FAQ con respuesta conocida, la regla en el prompt
    # no bastó, igual que pasó con otros casos hoy.
    _m_norm_app = _normalizar_nombre(mensaje)
    _pregunta_uso_app = (
        ("como uso" in _m_norm_app or "como funciona" in _m_norm_app)
        and ("app" in _m_norm_app or "esto" in _m_norm_app or "aplicacion" in _m_norm_app)
    )
    # "registr" (no "como registro") cubre registro/registrar/registrando/
    # registra — "como registro" exacto no coincidía con "cómo registrAR"
    # (infinitivo), que es como lo escribió el usuario real que encontró el bug.
    _pregunta_registro_comida = "registr" in _m_norm_app and "comida" in _m_norm_app
    _pregunta_registro_ejercicio = "registr" in _m_norm_app and "ejercicio" in _m_norm_app
    _pregunta_progreso = "donde veo" in _m_norm_app and ("progreso" in _m_norm_app or "balance" in _m_norm_app)

    if _pregunta_registro_comida:
        return (
            "Para registrar comida solo escribe en este chat lo que comiste, por "
            "ejemplo \"comí pollo con arroz\" — también puedes tocar el ícono "
            "naranja 🍽️ junto al cuadro de texto para un registro rápido, o dictarlo "
            "por voz con el ícono del micrófono."
        )
    if _pregunta_registro_ejercicio:
        return (
            "Para registrar ejercicio escribe en este chat qué hiciste, por "
            "ejemplo \"hice 3 series de 10 sentadillas con 20kg\" — también puedes "
            "tocar el ícono verde 🏋️ junto al cuadro de texto para armar una rutina, "
            "o dictarlo por voz con el ícono del micrófono."
        )
    if _pregunta_progreso:
        return "Tu progreso histórico está en la pestaña \"Seguimiento\" de la barra inferior."
    if _pregunta_uso_app:
        return (
            "Puedes escribirme directo en este chat (o dictar por voz con el ícono "
            "del micrófono) para registrar comidas o ejercicios, o pedirme "
            "recomendaciones. La barra inferior tiene Inicio, Asistente, Balance, "
            "Seguimiento y Perfil para navegar la app."
        )

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
            _unidad_r = 'ml' if 'ml' in mensaje.lower() or 'jugo' in mensaje.lower() or 'leche' in mensaje.lower() else 'g'
            # Prosa natural en vez de "P:Xg C:Yg G:Zg" — este return es un
            # template fijo en código, no pasa por el LLM ni por el recorte de
            # abajo, así que el formato se arregla aquí directamente.
            _partes_r = []
            for _val, _nom in ((p_r, "proteína"), (c_r, "carbohidratos"), (g_r, "grasa")):
                _partes_r.append(
                    f"casi nada de {_nom}" if _val < 0.5 else f"{_formato_num(_val)}g de {_nom}"
                )
            return (
                f"{nombre_r} ({grm:.0f}{_unidad_r}) tiene {k_r:.0f} kcal, "
                f"con {_partes_r[0]}, {_partes_r[1]} y {_partes_r[2]}."
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

    # Validador genérico: no repetir el mensaje del usuario + no mezclar el
    # tema contrario (dieta en pregunta de ejercicio, o viceversa). Antes esto
    # solo era una regla de prompt sin garantía — ahora usa el mismo ciclo
    # detectar→reintentar→respaldo que ya funciona al 100% en nutrición/ejercicio.
    if not _es_receta and not _es_tecnica:
        resultado = await validate_and_retry(
            respuesta=resultado,
            ia_engine=ia_engine,
            es_invalida=lambda t: (
                _repite_mensaje_usuario(t, mensaje)
                or _menciona_tema_no_relacionado(t, _tema_chat)
                or _tiene_contradiccion_impacto(t)
            ),
            construir_prompt_retry=lambda: (
                "Tu respuesta anterior tuvo un problema: "
                + (
                    "repitió/citó el mensaje del usuario al inicio. "
                    if _repite_mensaje_usuario(resultado, mensaje) else ""
                )
                + (
                    f"mencionó un tema no relacionado (la conversación es sobre "
                    f"{'ejercicio/lesión' if _tema_chat == 'ejercicio' else 'nutrición'}, "
                    f"no menciones {'dieta/kcal/nutrición' if _tema_chat == 'ejercicio' else 'ejercicio/entrenamiento'} "
                    f"a menos que el usuario lo pida explícitamente). "
                    if _menciona_tema_no_relacionado(resultado, _tema_chat) else ""
                )
                + (
                    "se contradijo a sí misma: dijo que evitaras impacto y luego "
                    "recomendó correr/trotar/saltar de todas formas. "
                    if _tiene_contradiccion_impacto(resultado) else ""
                )
                + f"\nMensaje del usuario: \"{mensaje}\"\n"
                + f"Conversación reciente:\n{hist_txt}\n"
                + "Responde de nuevo corrigiendo eso, máximo 3 oraciones, sin preguntas, sin citar el mensaje. "
                + "Si mencionas evitar impacto, NO sugieras correr/trotar/saltar en la misma respuesta."
            ),
            fallback=_filtrar_resultado_chat(resultado, _tema_chat),
        )

    # Guard de seguridad por lesión — la conversación libre también puede acabar
    # sugiriendo ejercicios (ej. "¿puedo correr con esta lesión?" → "mejor haz
    # sentadillas..."), y antes este guard solo protegía RECOMENDAR_EJERCICIO.
    # Misma lógica reutilizada: detecta lesión en perfil o mensaje/historial,
    # reintenta si sugiere algo riesgoso, fallback seguro en prosa si persiste.
    from app.services.rutina_service import (
        _LESIONES_SUSTITUCION, _detectar_lesiones, filtrar_lesiones_activas,
    )
    _condiciones_lista_chat = list(getattr(perfil, "medical_conditions", None) or [])
    # Historial completo, no solo los últimos 2 turnos — mismo motivo que en
    # respuesta_recomendacion_llm: si la zona se mencionó hace varios turnos,
    # debe seguir siendo candidata para que filtrar_lesiones_activas decida
    # si sigue activa o ya se recuperó (antes "desaparecía" de candidatas).
    _texto_hist_chat = " ".join(str(h.get("content", "")) for h in (historial or []))
    _lesiones_candidatas_chat = _detectar_lesiones(
        _condiciones_lista_chat + [mensaje or "", _texto_hist_chat]
    )
    # Descartar lesiones que el usuario ya indicó como recuperadas más
    # reciente que la última mención de dolor (ver filtrar_lesiones_activas).
    _lesiones_activas_chat = filtrar_lesiones_activas(
        _lesiones_candidatas_chat, historial, mensaje
    )
    if _lesiones_activas_chat:
        _riesgosos_chat: set[str] = set()
        _alternativas_chat: list[str] = []
        for _lesion in _lesiones_activas_chat:
            _cfg = _LESIONES_SUSTITUCION[_lesion]
            for _riesgoso, (_id_seguro, _nombre_seguro) in _cfg["sustituir"].items():
                if _riesgoso != "default":
                    _riesgosos_chat.add(_riesgoso)
                _alternativas_chat.append(_nombre_seguro)

        if any(r in _normalizar_nombre(resultado) for r in _riesgosos_chat):
            logger.warning(
                "[Chat] Ejercicio riesgoso para lesión detectada en conversación libre — reintentando"
            )
            _justif_chat = "; ".join(
                _LESIONES_SUSTITUCION[l]["justificacion"] for l in _lesiones_activas_chat
            )
            _prompt_retry_chat = (
                f"Eres entrenador y nutricionista. El usuario tiene: {_justif_chat}.\n"
                f"NO sugieras NINGUNO de estos ejercicios ni variantes: "
                f"{', '.join(sorted(_riesgosos_chat))}.\n"
                f"Si vas a sugerir actividad física, usa alternativas seguras como: "
                f"{', '.join(_alternativas_chat)}.\n"
                f"Mensaje del usuario: \"{mensaje}\"\n"
                f"Responde en máximo 3 oraciones naturales, sin listas ni preguntas."
            )
            resultado = await ia_engine._llamar_groq(_prompt_retry_chat, max_tokens=200, temp=0.3)
            resultado = _limpiar_markdown(resultado)

            if any(r in _normalizar_nombre(resultado) for r in _riesgosos_chat):
                logger.warning("[Chat] Reintento también riesgoso — usando fallback seguro")
                _nombres_unicos_chat = list(dict.fromkeys(_alternativas_chat))[:2]
                _alt_txt = " o ".join(_nombres_unicos_chat) if _nombres_unicos_chat else "estiramientos suaves"
                resultado = (
                    f"Por tu lesión, mejor evita esfuerzos que la sobrecarguen. "
                    f"Prueba con {_alt_txt} mientras te recuperas, y consulta con un profesional de salud."
                )

    # Garantía determinista de formato: el LLM no siempre respeta "máx 3 oraciones"
    # ni "sin pregunta al final" pese a tenerlo en el prompt — esto recorta el
    # TEXTO ya generado (no cambia el contenido/sentido), nunca depende de que
    # el LLM "se acuerde" cada vez. No se aplica a recetas/técnica (tienen su
    # propio formato de pasos numerados).
    if not _es_receta and not _es_tecnica:
        resultado = _recortar_respuesta_chat(resultado, mensaje)

    # Garantía determinista de formato (igual razón que el recorte de arriba):
    # pese a la regla del prompt, el LLM sigue usando "P:Xg C:Yg G:Zg" en vez
    # de prosa natural — esto reescribe el patrón sin tocar ningún número.
    resultado = _naturalizar_macros(resultado)

    return resultado


_RE_MACROS_ETIQUETA = re.compile(
    r'[.\s]*[—\-]?\s*P:\s*([\d.]+)\s*g\s*C:\s*([\d.]+)\s*g\s*G:\s*([\d.]+)\s*g\.?',
    re.IGNORECASE,
)


def _formato_num(n: float) -> str:
    return str(int(n)) if n == int(n) else f"{n:.1f}".rstrip("0").rstrip(".")


def _naturalizar_macros(texto: str) -> str:
    """Reescribe "P:Xg C:Yg G:Zg" (formato de etiqueta) a prosa natural, sin
    cambiar ningún valor numérico — solo CÓMO se presentan. El prompt ya pide
    esto (regla 1b de _PROMPT_CHAT) pero el LLM no lo respeta de forma
    confiable, mismo patrón que el resto de hoy: regla de prompt + garantía
    de código."""
    def _reemplazar(m: re.Match) -> str:
        p, c, g = float(m.group(1)), float(m.group(2)), float(m.group(3))
        partes = []
        for valor, nombre in ((p, "proteína"), (c, "carbohidratos"), (g, "grasa")):
            if valor < 0.5:
                partes.append(f"casi nada de {nombre}")
            else:
                partes.append(f"{_formato_num(valor)}g de {nombre}")
        return f", con {partes[0]}, {partes[1]} y {partes[2]}."
    return _RE_MACROS_ETIQUETA.sub(_reemplazar, texto)


def _recortar_respuesta_chat(texto: str, mensaje_usuario: str, max_oraciones: int = 3) -> str:
    """Recorta a un máximo de oraciones y quita la pregunta final si el usuario
    no pidió 'consejo'/'ayuda' explícitamente — recorte de formato puro, no
    cambia el contenido de lo que el LLM ya dijo."""
    import re as _re_trim
    _oraciones = _re_trim.split(r'(?<!\d)(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])', texto.strip())
    _oraciones = [o.strip() for o in _oraciones if o.strip()]
    if len(_oraciones) > max_oraciones:
        _oraciones = _oraciones[:max_oraciones]

    _pidio_consejo = any(
        p in mensaje_usuario.lower() for p in ("consejo", "ayuda", "ayudame", "ayúdame")
    )
    if len(_oraciones) > 1 and _oraciones[-1].rstrip().endswith("?") and not _pidio_consejo:
        _oraciones = _oraciones[:-1]

    _resultado = " ".join(_oraciones).strip()
    if _resultado and _resultado[-1] not in ".!":
        _resultado += "."
    return _resultado or texto.strip()


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


_STOPWORDS_BASE_TEXTUAL = frozenset({
    "de", "la", "el", "los", "las", "con", "y", "en", "del", "al", "un", "una",
})


# Vocabulario nutricional genérico — NUNCA es en sí mismo el nombre de un
# alimento, sin importar el producto ("kcal", "etiqueta", "proteína" no son
# comida, son palabras que describen comida). No es una lista de productos/
# marcas (eso seguiría siendo conversacional) — es un cierre léxico chico y
# fijo del propio dominio de nutrición, igual que "el/la/de" son stopwords de
# español. Encontrado en pruebas: al dar datos de etiqueta ("160 kcal, 6g de
# proteína por 100ml..."), el checker de completitud marcaba "kcal" como
# posible alimento faltante y el LLM del reintento lo aceptó como real.
_PALABRAS_NO_ALIMENTO_GENERICAS = frozenset({
    "kcal", "cal", "calorias", "calorías", "caloria", "caloría",
    "proteina", "proteína", "proteinas", "proteínas",
    "carbohidrato", "carbohidratos", "carbohidrato",
    "grasa", "grasas", "macro", "macros", "macronutriente", "macronutrientes",
    "etiqueta", "etiquetas", "nutricional", "nutricionales", "nutricion",
    "nutrición", "porcion", "porción", "porciones", "racion", "ración",
    "informacion", "información", "valor", "valores", "dice", "indica",
    "segun", "según", "cada", "tiene", "trae", "contiene",
})


def _extraccion_tiene_base_textual(nombre_extraido: str, mensaje_original: str) -> bool:
    """Red de seguridad determinista (sin costo de tokens) contra alucinaciones
    del LLM: verifica que el nombre extraído tenga al menos una palabra
    significativa presente en el mensaje real del usuario. La Regla 0 del
    prompt ya le pide al LLM no inventar nada cuando no hay alimento/ejercicio
    real, pero esa instrucción no es garantía (validado en pruebas: el mismo
    bug reapareció con mensajes distintos) — esto la respalda con código."""
    _nombre_norm = _normalizar_nombre(nombre_extraido or "")
    palabras = [
        p for p in _nombre_norm.split()
        if len(p) > 3 and p not in _STOPWORDS_BASE_TEXTUAL
    ]
    if not palabras:
        return True  # nombre muy corto/genérico para verificar — no bloquear
    # Si TODAS las palabras del nombre son vocabulario nutricional genérico
    # ("kcal", "etiqueta"...), no es un alimento real sin importar que la
    # palabra literalmente aparezca en el mensaje.
    if all(p in _PALABRAS_NO_ALIMENTO_GENERICAS for p in palabras):
        return False
    msg_norm = _normalizar_nombre(mensaje_original or "")
    return any(p in msg_norm for p in palabras)


# Stopwords para el chequeo de COMPLETITUD (lo opuesto a _extraccion_tiene_base_textual:
# en vez de validar que lo extraído venga del mensaje, valida que el mensaje no
# tenga algo que el LLM dejó fuera). Deliberadamente conservador — palabras de
# 4+ letras fuera de esta lista chica, para minimizar falsos positivos.
_STOPWORDS_COMPLETITUD = frozenset({
    "hola", "buenas", "buenos", "comi", "come", "comer", "tome", "tomo",
    "bebi", "cene", "almorce", "desayune", "hoy", "ayer", "con", "para",
    "por", "mis", "tus", "sus", "una", "unos", "unas", "taza", "vaso",
    "plato", "porcion", "racion", "rebanada", "tajada", "lonja", "rodaja",
    "trozo", "pedazo", "cucharada", "cucharadita", "puñado", "copa",
    "botella", "lata", "jarra", "gramos", "litros", "cantidad", "tambien",
    # Verbos de comando ("agrega que comí X") — sin esto, "agrega" se
    # detectaba como un posible alimento faltante y disparaba un reintento
    # que terminaba duplicando el ítem real (ej. "Huevo frito" x2).
    "agrega", "agregalo", "agregale", "agregar", "anota", "anotalo",
    "registra", "registralo", "incluye", "incluyelo", "guarda", "guardalo",
    "ponme", "apunta", "apuntalo", "registro", "sumale", "suma",
    # Modificadores de porción ("medio plato", "un cuarto de") — ya se manejan
    # aparte (factor de escala 0.5/0.25/etc. más abajo); sin excluirlos aquí,
    # "medio" se marcaba como posible alimento faltante y el reintento
    # alucinó un ítem fantasma "Plato" para "comí medio plato de lomo saltado".
    "medio", "media", "mitad", "cuarto", "cuarta", "chico", "chica",
    "pequeño", "pequeña", "grande", "mediano", "mediana", "doble",
    "falto", "faltaba", "faltó", "falta", "olvide", "olvidé", "olvido",
})


def _palabras_faltantes_en_extraccion(mensaje: str, alimentos: list[dict]) -> list[str]:
    """Detecta palabras del mensaje que parecen alimentos pero no aparecen en
    ningún nombre extraído — ej. "arroz con palta y gelatina" → si solo se
    extrajo "Arroz" y "Gelatina", "palta" queda detectada como posible
    omisión. No es prueba definitiva (puede ser un adjetivo o palabra normal),
    por eso el caller debe verificar con un reintento antes de aceptarla."""
    palabras_extraidas = set(
        _normalizar_nombre(" ".join(a.get("nombre", "") for a in alimentos)).split()
    )
    _PUNTUACION_BORDE = ",.;:!?()¡¿\"'"
    palabras_msg = [
        p for p in (
            t.strip(_PUNTUACION_BORDE) for t in _normalizar_nombre(mensaje or "").split()
        )
        if len(p) >= 4
        and p not in _STOPWORDS_COMPLETITUD
        and p not in _STOPWORDS_BASE_TEXTUAL
        and p not in _PALABRAS_NO_ALIMENTO_GENERICAS
        and not any(c.isdigit() for c in p)  # "320ml", "100ml" no son alimentos
    ]
    return [p for p in palabras_msg if p not in palabras_extraidas]


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
