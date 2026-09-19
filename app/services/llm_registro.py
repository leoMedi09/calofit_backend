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
import asyncio
import httpx
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


from app.core.objetivo_utils import es_superavit as _es_superavit_goal
from app.core.user_context import UserContext
from app.core.mets_gym import tabla_prompt_texto as _tabla_met_texto

_RX_CANTIDAD_AMBIGUA = re.compile(
    r'\b(?:no\s+(?:estoy\s+seguro|se|sé)\s+si|creo\s+que|tal\s+vez|quiz[aá]s|mas\s+o\s+menos|más\s+o\s+menos|aprox|entre\s+\d+\s+y\s+\d+|o\s+(?:media|una|dos)|un\s+poco(?:\s+de)?|algo(?:\s+de)?|bastante)\b',
    re.IGNORECASE
)

_RX_CORRECCION_REGISTRO = re.compile(
    r'\bcorrecci[oó]n\b|\bme\s+equivoqu[eé]\b|\ben\s+realidad\b|\bno\s+fueron\b|'
    r'\bfueron\s+m[aá]s\b|\bfueron\s+menos\b|\bno\s+fue\b|\bfue\s+m[aá]s\b|\bfue\s+menos\b|'
    r'\b(?:era|eran)\b',
    re.IGNORECASE,
)

_RX_NO_ALIMENTO_PELIGROSO = re.compile(
    r'\b(?:'
    r'vidrio|cristal|vidrios|'
    r'tierra(?!\s+(?:de\s+)?(?:man[ií]|ma[ií]z|trigo))|'
    r'metal(?:es)?|hierro(?!\s+(?:fundido|forjado))|acero|aluminio|'
    r'cemento|hormig[o\u00f3]n|cal(?:\s+viva)?|yeso|argamasa|'
    r'pintura|barniz|thinner|solvente|aguarr[a\u00e1]s|'
    r'pl[a\u00e1]stico|caucho|goma\s+de\s+borrar|'
    r'clavos|tornillos|tuercas|alambre|cables?\s+el[e\u00e9]ctricos?|'
    r'gasolina|di[e\u00e9]sel|kerosene|aceite\s+de\s+motor|'
    r'detergente|lej[i\u00ed]a|cloro(?!\s+de\s+piscina)|amoniaco|sosa\s+c[a\u00e1]ustica|'
    r'veneno|raticida|pesticida|insecticida|herbicida|'
    r'medicamento[s]?\s+(?:en\s+exceso|mezclados?)|'
    r'tiza|gis|carb[o\u00f3]n(?:\s+mineral)?|carbon(?:cillo)?\s+mineral'
    r')\b',
    re.IGNORECASE,
)

_NO_ALIMENTO_CONTEXTOS_EXCLUIDOS = [
    re.compile(r'\baj[i\u00ed]\s+(?:\w+\s+)?de\s+vidrio\b', re.IGNORECASE),
    re.compile(r'\b(?:olla|sart[e\u00e9]n|bandeja|molde|papel)\s+(?:de\s+)?aluminio\b', re.IGNORECASE),
    re.compile(r'\baluminio\s+(?:de\s+)?(?:cocina|olla|sart[e\u00e9]n|molde|papel)\b', re.IGNORECASE),
    re.compile(r'\baluminio\s+en\s+(?:olla|sart[e\u00e9]n|cocina)\b', re.IGNORECASE),
]


def _mensaje_contiene_no_alimento(mensaje: str):
    """Devuelve el término peligroso detectado, o None si el mensaje es seguro.
    Aplica exclusiones de contexto para evitar falsos positivos."""
    match = _RX_NO_ALIMENTO_PELIGROSO.search(mensaje or "")
    if not match:
        return None
    for rx_excl in _NO_ALIMENTO_CONTEXTOS_EXCLUIDOS:
        if rx_excl.search(mensaje):
            return None
    return match.group(0)


_FALLBACKS_OPCIONES = {
    "DESAYUNO": [
        "- Fruta picada con chía (~90 kcal, P:2g C:20g G:1g)",
        "- Avena tibia con agua (~150 kcal, P:5g C:27g G:3g)",
        "- Tostada integral con palta (~130 kcal, P:3g C:16g G:7g)",
        "- Plátano con frutos secos (~180 kcal, P:4g C:28g G:7g)",
        "- Batido de frutas sin azúcar (~110 kcal, P:2g C:25g G:1g)",
    ],
    "ALMUERZO": [
        "- Lentejas guisadas con verduras (~350 kcal, P:18g C:55g G:5g)",
        "- Sopa de verduras con quinua (~300 kcal, P:10g C:50g G:5g)",
        "- Ensalada de garbanzos y palta (~280 kcal, P:8g C:35g G:12g)",
        "- Tofu a la plancha con brócoli y arroz (~380 kcal, P:22g C:45g G:10g)",
        "- Crema de zapallo y arvejas (~290 kcal, P:9g C:42g G:8g)",
    ],
    "CENA": [
        "- Sopa de verduras ligera (~150 kcal, P:5g C:22g G:4g)",
        "- Ensalada de espinacas y palta (~200 kcal, P:4g C:18g G:13g)",
        "- Menestra de frejol ligera (~240 kcal, P:12g C:38g G:4g)",
        "- Tostadas con hummus y pepino (~180 kcal, P:6g C:26g G:6g)",
        "- Tortilla vegetal de champiñones (~220 kcal, P:10g C:12g G:15g)",
    ],
    "MERIENDA": [
        "- Fruta de estación (~80 kcal, P:1g C:20g G:0g)",
        "- Puñado de almendras y nueces (~160 kcal, P:5g C:8g G:13g)",
        "- Tostada integral con palta (~120 kcal, P:3g C:15g G:6g)",
        "- Rodajas de manzana con canela (~70 kcal, P:1g C:17g G:0g)",
        "- Taza de quinua cocida ligera (~130 kcal, P:4g C:24g G:2g)",
    ]
}

def obtener_fallback_aleatorio(momento: str) -> str:
    import random
    m = (momento or "ALMUERZO").upper().strip()
    if m not in _FALLBACKS_OPCIONES:
        m = "ALMUERZO"
    opciones = _FALLBACKS_OPCIONES[m]
    seleccion = opciones[:min(3, len(opciones))]
    return "\n".join(seleccion)


async def _llamar_groq_con_excepciones(ia_engine, prompt: str, max_tokens: int = 800, temp: float = 0.7, model: str = None) -> str:
    import asyncio
    import httpx
    raw = await ia_engine._llamar_groq(prompt, max_tokens=max_tokens, temp=temp, model=model)
    if not raw or not raw.strip():
        raise ValueError("Respuesta vacía del LLM")
    raw_lower = raw.lower()
    if "timed out" in raw_lower or "timeout" in raw_lower or "no pudimos completar la respuesta a tiempo" in raw_lower:
        raise asyncio.TimeoutError("Groq Timeout Error")
    if "429" in raw_lower or "rate_limit" in raw_lower or "rate limit" in raw_lower or "límite de consultas alcanzado" in raw_lower:
        raise ConnectionError("Groq Rate Limit Error (429)")
    if raw.startswith("[Error:") or "[error:" in raw_lower:
        raise ConnectionError(f"Groq API connection/HTTP error: {raw}")
    return raw


def _obtener_fallback_chat_seguro(perfil, tema: str) -> str:
    conds = list(getattr(perfil, "medical_conditions", None) or [])
    res_txt = "Lo siento, hubo un inconveniente al conectar con el servidor. "
    if tema == "ejercicio":
        res_txt += "Te sugiero realizar una caminata ligera de 15 minutos o estiramientos suaves para mantenerte activo de forma segura."
    else:
        res_txt += "Te sugiero opciones seguras locales como avena tibia con agua, tostadas integrales con palta o fruta picada."
        if conds:
            res_txt += f" (Teniendo en cuenta tus condiciones: {', '.join(conds)})."
    return res_txt




def _asegurar_contexto(perfil, consumido: float = 0.0, quemado: float = 0.0, plan_hoy: dict = None, ctx: Optional[UserContext] = None) -> UserContext:
    if ctx is not None:
        return ctx
    return UserContext.build(perfil, consumido, quemado, plan_hoy or {})

def obtener_fallback_restricciones_medicas(condiciones: list[str]) -> str:
    restricciones = []
    for cond in condiciones:
        cond_lower = cond.lower().strip()
        if "vegano" in cond_lower:
            restricciones.append("VEGANO: PROHIBIDO todo producto animal (carne, pollo, pescado, huevo, leche, queso, yogur, mantequilla, miel).")
        elif "vegetariano" in cond_lower:
            restricciones.append("VEGETARIANO: PROHIBIDO todo tipo de carne, pollo y pescado.")
        elif "diabetes" in cond_lower or "diabético" in cond_lower or "diabetico" in cond_lower:
            restricciones.append("DIABETES: evitar azúcares refinados, miel, dulces y alimentos de alto índice glucémico (pan blanco, arroz blanco en exceso). Preferir carbohidratos complejos y fibra.")
        elif "hipertensión" in cond_lower or "hipertension" in cond_lower or "hipertenso" in cond_lower:
            restricciones.append("HIPERTENSIÓN: limitar sodio, sal de mesa añadida, embutidos, enlatados y alimentos ultraprocesados.")
        elif "lactosa" in cond_lower or "intolerancia a la lactosa" in cond_lower:
            restricciones.append("INTOLERANCIA A LA LACTOSA: PROHIBIDO lácteos enteros. Permitido versiones deslactosadas o alternativas vegetales.")
        elif "celiaco" in cond_lower or "celíaco" in cond_lower or "gluten" in cond_lower:
            restricciones.append("CELIAQUÍA / SIN GLUTEN: PROHIBIDO trigo, cebada, centeno y derivados. Solo alimentos certificados gluten-free.")
        else:
            restricciones.append(f"{cond}: Seguir las pautas de alimentación recomendadas por tu médico para esta condición.")
    
    if restricciones:
        return (
            "⛔ RESTRICCIONES MÉDICAS OBLIGATORIAS (FALLBACK LOCAL) — aplica en los 3 platos:\n"
            + "\n".join(f"• {r}" for r in restricciones)
            + "\n⚠️ Aplica cada restricción SOLO si el plato normalmente lleva ese ingrediente.\n"
            + "🚨 PRIORIDAD ABSOLUTA: estas restricciones médicas pesan MÁS que cualquier ejemplo anterior.\n\n"
        )
    return ""

_RX_SUPERAVIT_MENSAJE = re.compile(
    r'masa muscular|ganar m[uú]sculo|aumentar m[uú]sculo|volumen muscular|bulking',
    re.IGNORECASE,
)


def _calcular_balance_meta(
    consumido: float, meta: float, quemado: float, objetivo: str = "", mensaje: str = "",
) -> dict:
    """Calcula el balance calórico real del día — mismo criterio que la UI:
    total disponible = meta + quemado (las calorías quemadas se suman al
    presupuesto). Devuelve también el bloque de texto ya armado para el
    prompt y la advertencia en lenguaje natural, así ningún caller recalcula
    ni redacta la regla por su cuenta.
    """
    _total_disponible = meta + quemado
    restante = max(0.0, _total_disponible - consumido)
    excedido = max(0.0, consumido - _total_disponible)
    pct = round(consumido / _total_disponible * 100) if _total_disponible > 0 else 0
    es_masa_muscular = (
        _es_superavit_goal(objetivo)
        or bool(_RX_SUPERAVIT_MENSAJE.search(mensaje or ""))
    )

    bloque_balance = ""
    advertencia_natural = None
    if es_masa_muscular and excedido > 0:
        bloque_balance = (
            "ℹ️ BALANCE VS META: el objetivo del usuario es ganar masa muscular — un consumo "
            "por encima de la meta calórica es ESPERADO y CORRECTO para este objetivo. "
            "PROHIBIDO decir que \"excedió su meta\" o sugerir que coma menos por este motivo.\n\n"
        )
        advertencia_natural = (
            "Como tu objetivo es ganar masa muscular, estar en superávit es esperado y correcto."
        )
    elif excedido > 0:
        bloque_balance = (
            f"⚠️ BALANCE VS META: el usuario YA EXCEDIÓ su meta calórica de hoy por "
            f"{round(excedido)} kcal ({pct}% de lo disponible, ya sumadas las calorías "
            f"quemadas por ejercicio). Dilo de forma directa y sin contradicciones.\n\n"
        )
        advertencia_natural = (
            f"Ya superaste tu meta de hoy por {round(excedido)} kcal — "
            f"si aún quieres comer algo, mejor que sea ligero."
        )


    return {
        "total_disponible": _total_disponible,
        "restante": restante,
        "excedido": excedido,
        "pct": pct,
        "es_masa_muscular": es_masa_muscular,
        "bloque_balance": bloque_balance,
        "advertencia_natural": advertencia_natural,
    }


_IDENTIDAD = """Eres el Asistente CaloFit del gimnasio World Light Lambayeque — un profesional con doble especialización:
• Nutricionista Clínico y Deportivo certificado con dominio completo de la composición nutricional de alimentos de TODO el mundo: gastronomía peruana (Lambayeque, Chiclayo, Lima y regiones), latinoamericana, internacional, fast food, comida asiática, europea, árabe, japonesa, china, italiana, etc. Tu conocimiento equivale al de un nutricionista experto que ha estudiado la Tabla Peruana de Composición de Alimentos (INS/CENAN), USDA FoodData Central, la FAO/OMS y múltiples fuentes científicas internacionales — pero no consultas bases de datos externas en tiempo real: aplicas ese conocimiento acumulado directamente.
• Entrenador Personal certificado (NSCA-CPT) con conocimiento en hipertrofia, pérdida de grasa, cardio y entrenamiento funcional.
Conoces TODOS los alimentos: cebiches, causas, secos, arroces, menestras, caldos, frituras, dulces, bebidas, frutas tropicales, sushi, ramen, pizza, döner kebab, pad thai, falafel, bowls, proteínas en polvo, suplementos, snacks procesados de cualquier marca/país, y cualquier otro alimento real del mundo.
"""

_PROMPT_COMIDA = _IDENTIDAD + """
TAREA: Analiza el mensaje y extrae TODOS los alimentos/bebidas consumidos con sus macros exactos.

FUENTE DE DATOS: Aplica tu conocimiento profesional de nutricionista clínico para estimar valores realistas y científicamente coherentes. Conoces la composición nutricional de cualquier alimento del mundo — peruano, latinoamericano, asiático, europeo, fast food, internacional — con la misma precisión que un nutricionista experto que ha estudiado múltiples tablas de composición alimentaria (INS/CENAN, USDA, FAO/OMS, tablas europeas).
NO estás limitado a ninguna base de datos específica: si el alimento existe en el mundo real y es comestible, PUEDES y DEBES estimarlo con valores realistas basados en tu conocimiento clínico.
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
   Paso 1 — macros: ¿Son los valores de prot_g/carb_g/grasa_g coherentes con tu conocimiento
   nutricional de ESE alimento? Un huevo tiene grasa, el arroz tiene carbos, el pollo tiene
   proteína — si algún macro queda en 0 cuando no debería, recalcula.
   Paso 2 — escala: ¿Escalaste los macros al porcion_g real del usuario?
   Si el alimento tiene X kcal/100g y el usuario comió Y gramos → kcal = X × Y / 100.
   Paso 3 — atwater: ¿kcal ≈ 4×prot_g + 4×carb_g + 9×grasa_g? Si la diferencia supera
   el 10%, ajusta los macros para que sean coherentes con la kcal real de ese alimento.

VALIDACIÓN: Antes de calcular, determina si cada alimento es real.
Un alimento es real si existe en el mundo físico y es comestible — no importa si es peruano, japonés, árabe, o de cualquier otro origen. Tu conocimiento como nutricionista abarca TODOS los alimentos del mundo.
NO son reales: ingredientes ficticios, mitológicos, inventados (unicornio, dragón, zarblak, florbonix). Tampoco materiales incomestibles (vidrio, metal, cemento).
Son reales aunque sean inusuales o poco conocidos: maca, sachatomate, ceviche de champiñones, saltado de tofu, sushi, ramen, falafel, döner, pad thai, kimchi, injera, etc.

⛔ MATERIALES INCOMESTIBLES Y PELIGROSOS (MÁXIMA PRIORIDAD):
Si el mensaje menciona materiales NO comestibles y peligrosos como vidrio, cristal, tierra, metal, cemento, pintura, plástico, gasolina, veneno, detergente, u otros materiales o sustancias químicas peligrosas:
- Si el item mencionado ES el material peligroso (ej. "comí vidrio", "tomé tierra"): es_real = false.
- Si aparece junto a un alimento real (ej. "pan con vidrio"): el PLATO COMPLETO es inválido.
  En ese caso, responde exactamente: {{"alimentos":[], "prot_total":0, "carb_total":0, "grasa_total":0, "contiene_no_alimento": true}}
  El campo "contiene_no_alimento": true indica que el mensaje describió algo no comestible/peligroso.
  NUNCA registres el alimento real sin el peligroso.
⚠️ IMPORTANTE: Solo clasifica como peligroso/incomestible si es un material o sustancia nociva real de los mencionados (vidrio, metal, veneno, etc.). Si el usuario menciona una palabra desconocida, inventada, un plato raro o un error de escritura (ej. "umas", "zarblak", "florbonix"), NO actives "contiene_no_alimento": true ni marques todo el plato como no comestible. Trátalo únicamente bajo la regla de VALIDACIÓN (marca es_real: false para el ítem desconocido y extrae el resto de alimentos normales como de costumbre).

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

━━ DISAMBIGUACIÓN CRÍTICA (NUNCA CONFUNDIR) ━━
- "palta" = aguacate/avocado (NUNCA confundir con "pata" = extremidad/pierna).
- "ceviche" = plato de pescado crudo marinado en limón (NUNCA cocido ni horneado).
- Usa tu conocimiento general para interpretar libre y correctamente expresiones informales de comida ("me zampé", "le di duro a", "me eché un", "picoté", etc.) y modismos peruanos u otros dialectos.

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
6. CANTIDADES: "dos panes con pollo" → UN solo ítem {{nombre:"Pan con Pollo", cantidad:2, kcal: de una sola unidad}}. NUNCA separes en Pan ×2 + Pollo por separado — el "con" indica un combo, no ingredientes sueltos. Los macros, kcal y porcion_g en el JSON deben ser siempre POR UNIDAD (para cantidad=1), NO el total acumulado de todas las unidades. El sistema se encargará de multiplicarlo por la cantidad automáticamente. Nombre siempre en singular.
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
   · OLLUCO (también escrito "oyuco"/"oluco") CON cualquier carne (chancho, charqui,
     carne seca) es UN guiso tradicional — SIEMPRE UN solo ítem "Olluco con [carne]"
     con TODA la carne ya incluida en sus macros. NUNCA generes "Olluco" y "Chancho"
     como dos ítems separados — la carne NO es un alimento aparte en este plato.
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
13. MODIFICADORES DE TAMAÑO Y FRACCIONES: "medio/media" → ~50%; "un cuarto de" → ~25%; "porción/plato pequeño(a)" → ~60-70%; "porción/plato grande" → ~130-160%; "porción/plato mediano(a)" → 100% (base normal). Esto es un CONCEPTO GENERAL, no una lista cerrada: CUALQUIER fracción expresada en palabras o número ("un tercio", "dos tercios", "tres cuartos", "un quinto", "2/5", "60%", etc.) de UNA unidad se interpreta IGUAL — convierte la fracción a decimal y multiplica "porcion_g" y kcal/prot_g/carb_g/grasa_g de UNA SOLA unidad estándar de ese alimento por ese decimal.
   ⚠️ ERROR CRÍTICO A EVITAR: el numerador de una fracción NUNCA es "cantidad". "tres cuartos de palta" significa 0.75 de UNA palta (cantidad=1, porcion_g≈0.75×porción normal) — NO son 3 paltas (cantidad=3). Antes de responder, pregúntate: ¿el número que veo modifica CUÁNTAS unidades hay, o QUÉ FRACCIÓN de una unidad se comió? Si va seguido de "cuartos/tercios/quintos/de un(a)", es fracción de una unidad, jamás cantidad de unidades.
   Aplica el factor TANTO a "porcion_g" COMO a kcal/prot_g/carb_g/grasa_g de forma proporcional (ej: "medio vaso de leche" → ~120ml y la mitad de las kcal/macros de un vaso completo; "un tercio de palta" → ~33% del porcion_g y kcal de una palta normal, cantidad=1).
14. CONSISTENCIA: para un mismo alimento y la misma porción, usa SIEMPRE los mismos valores
    nutricionales basados en tu conocimiento profesional — NO improvises valores distintos cada
    vez. Si tienes duda entre varias preparaciones, usa la versión más común/estándar en el
    contexto del usuario (peruano por defecto, o el país de origen del plato si es extranjero).
15. AMBIGÜEDAD DE CANTIDADES: Si el usuario expresa incertidumbre o duda sobre la cantidad (ej. "no estoy seguro de si era media o una taza", "creo que eran dos huevos"), estima la cantidad de forma razonable y conservadora basada en el contexto y usa el valor medio o el más probable.
16. ⚠️ NOMBRES LIMPIOS SIN MEDIDAS: Evita incluir palabras que indiquen contenedores, porciones o medidas (como "taza de", "vaso de", "plato de", "unidad de", "ración de", "poción de") en el campo "nombre". El nombre debe ser únicamente el del alimento o plato en sí (ej: "Café" en lugar de "Taza de café", "Leche" en lugar de "Vaso de leche", "Arroz con pollo" en lugar de "Plato de arroz con pollo", "Plátano" en lugar de "Unidad de plátano").
17. ⚠️ VERIFICACIÓN DE DENSIDAD POR CATEGORÍA (antes de responder, para cada
    ítem): ubica el alimento en su categoría amplia y revisa que tu kcal/100g
    (o /100ml) sea coherente con ella — no son alimentos específicos, son
    rangos físicos típicos de CUALQUIER alimento de ese tipo:
    - Agua/infusión sin azúcar: 0-5. Bebida azucarada/gaseosa: 35-50.
      Leche o lácteo líquido: 55-70. Jugo natural: 40-60.
    - Vegetal o fruta fresca: 15-90.
    - Proteína magra cocida (carnes, pescado, huevo): 100-250.
    - Carbohidrato/almidón cocido (arroz, papa, pan, menestras): 100-200.
    - Grasa, aceite o fruto seco: 500-900 (el máximo físico real es 900).
    Si tu primer cálculo se sale mucho del rango esperado para la categoría
    de ese alimento (por arriba O por abajo), está mal — vuelve a calcularlo
    antes de responder, no lo dejes así.
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
""" + _tabla_met_texto() + """

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
5. Traduce de manera flexible cualquier término coloquial, sinónimo o jerga al nombre oficial del ejercicio correspondiente. Por ejemplo, "muertos" → "Peso Muerto", "jalones" → "Jalón al Pecho", "pichanga" → "Fútbol", "tiré pecho" o "entrené pecho" → rutina completa de pecho. Confía en tu conocimiento general para interpretar correctamente expresiones informales y modismos de entrenamiento.
   Si el usuario no especifica el ejercicio exacto pero dice el grupo muscular
   ("entrené pecho", "hice pierna", "trabajé espalda"), infiere una rutina típica
   de ese grupo (3-4 ejercicios, ~45 min, intensidad Media-Alta) y calcula las kcal.
"""

_PROMPT_MET_DESCONOCIDO = _IDENTIDAD + """
TAREA: Como Entrenador Personal certificado, estima el valor MET (intensidad
metabólica) del ejercicio "{nombre}" — no está en mi catálogo de referencia,
así que usa tu conocimiento profesional, calibrado con esta tabla:

""" + _tabla_met_texto() + """

Responde SOLO con JSON, sin texto extra: {{"met": numero_decimal}}
"""


async def estimar_met_desconocido(nombre: str, ia_engine) -> float:
    """MET por LLM para un ejercicio que no está en METS_GYM (fast-path local).
    Cubre cualquier ejercicio sin necesidad de agregarlo a mano al catálogo."""
    try:
        prompt = _PROMPT_MET_DESCONOCIDO.format(nombre=nombre)
        raw = await _llamar_groq_con_excepciones(ia_engine, prompt, max_tokens=40, temp=0.0)
        datos = _parse_json(raw)
        met = float(datos.get("met")) if datos and datos.get("met") else 0.0
        return met if met > 0 else 5.0
    except Exception as e:
        logger.warning("[Ejercicio manual] Fallback MET por LLM falló para '%s': %s", nombre, e)
        return 5.0


_PROMPT_RECOMENDACION_COMIDA = """Eres un clasificador de platos. Responde SOLO con una lista. Nada más.

Dieta: {dieta}. Calorías disponibles: {restante} kcal. Restricciones: {condiciones}.

Escribe EXACTAMENTE 3 líneas en este formato (sin introducción, sin conclusión):
- NombrePlato1 (~XXX kcal)
- NombrePlato2 (~YYY kcal)
- NombrePlato3 (~ZZZ kcal)

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
  0. PROHIBIDO hacer referencia explícita al historial de conversación. NUNCA uses frases como
     "como mencionaste antes", "como dijiste", "como recordarás", "anteriormente dijiste",
     "en tu mensaje anterior", "ya me contaste". Usa el contexto internamente para ser coherente,
     pero NO lo anuncies ni lo cites. Responde como si fuera una conversación natural continua.
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
  Usa tu conocimiento profesional de nutricionista para dar el valor realista de cualquier
  alimento. No estás limitado a una lista — responde con precisión para el alimento que
  el usuario mencione, sea peruano, internacional, marca comercial, o cualquier otro.
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

{bloque_balance_meta}⛔ BALANCE VS META (CRÍTICO):
  Usa EXCLUSIVAMENTE la advertencia de balance indicada arriba (si la hay) — NO calcules tú
  mismo si excedió la meta a partir de los números del perfil, ni mezcles "meta" con "lo
  disponible" (lo disponible ya incluye lo quemado). Si arriba no aparece ninguna advertencia
  de balance, NO digas que excedió su meta ni menciones un exceso — puedes decir cuánto le
  queda disponible. NUNCA digas que "está cumpliendo su objetivo" si SÍ hay advertencia de
  exceso arriba — son afirmaciones contradictorias.
  ⚠️ EXCEPCIÓN DE MÁXIMA PRIORIDAD PARA GANAR MASA / MÚSCULO:
  Si el objetivo del usuario del perfil es "SUPERAVIT", "ganar_leve" o "ganar masa" (ganar masa muscular / superávit / volumen), y el usuario pregunta si debe/puede seguir comiendo o si ya pasó su meta, debes felicitarlo o alentarlo de forma neutral a seguir comiendo. Explícale que para ganar músculo o estar en volumen es necesario y correcto estar en superávit calórico (comer más de la meta). PROHIBIDO decirle que evite comer, prohibido decirle que comer más dificultará sus objetivos, y prohibido sugerirle que coma menos. Debe comer más para lograr su meta de ganancia muscular.
"""


async def registrar_comida_llm(
    mensaje: str,
    perfil,
    plan_hoy: dict,
    db: Session,
    ia_engine,
    historial: list = None,
    ctx: Optional[UserContext] = None,
) -> dict:
    """Registra comida con macros estimados por LLM. Sin lookup de BD."""
    ctx = _asegurar_contexto(perfil, plan_hoy=plan_hoy, ctx=ctx)
    _mensaje_low_cache0 = (mensaje or "").lower()
    _tiene_varios_alimentos = (
        " y " in _mensaje_low_cache0
        or "," in _mensaje_low_cache0
        or " más " in _mensaje_low_cache0
        or " mas " in _mensaje_low_cache0
    )
    cached = None if _tiene_varios_alimentos else _buscar_en_cache(mensaje)
    if cached:
        kcal  = round(float(cached.get("kcal", 0)), 1)
        prot  = round(float(cached.get("prot_g", 0)), 1)
        carb  = round(float(cached.get("carb_g", 0)), 1)
        grasa = round(float(cached.get("grasa_g", 0)), 1)
        nombre_cached = cached.get("nombre", mensaje)
        if kcal > 0 or prot > 0 or carb > 0 or grasa > 0:
            logger.info("[Registro] Usando macros cacheados para '%s': %s kcal", nombre_cached, kcal)
            datos = {
                "alimentos": [{"nombre": nombre_cached, "es_real": True,
                                "kcal": kcal, "prot_g": prot, "carb_g": carb, "grasa_g": grasa}],
                "prot_total": prot, "carb_total": carb, "grasa_total": grasa,
                "kcal_total": kcal,
            }
            goto_save = True
        else:
            datos = None
            goto_save = False
    else:
        datos = None
        goto_save = False

    _filas_previas_correccion: list | None = None
    if not goto_save and _RX_CORRECCION_REGISTRO.search(mensaje or ""):
        from app.core.utils import get_peru_date as _get_hoy_correccion
        from app.models.comida_registro import ComidaRegistro as _CRCorreccion
        _hoy_correccion = _get_hoy_correccion()
        _ultimo_cr = (
            db.query(_CRCorreccion)
            .filter(_CRCorreccion.client_id == perfil.id, _CRCorreccion.fecha == _hoy_correccion)
            .order_by(_CRCorreccion.id.desc())
            .first()
        )
        if _ultimo_cr:
            _filas_previas_correccion = (
                db.query(_CRCorreccion)
                .filter(
                    _CRCorreccion.client_id == perfil.id,
                    _CRCorreccion.fecha == _hoy_correccion,
                    _CRCorreccion.nombre_alimento == _ultimo_cr.nombre_alimento,
                    _CRCorreccion.texto_original == _ultimo_cr.texto_original,
                )
                .all()
            )
            mensaje = f"{_ultimo_cr.nombre_alimento} - {mensaje}"
            logger.info(
                "[Registro] Corrección detectada — reescribiendo mensaje con alimento previo '%s'",
                _ultimo_cr.nombre_alimento,
            )

    if not goto_save:
        _prematch_peligroso = _RX_NO_ALIMENTO_PELIGROSO.search(mensaje or "")
        if _prematch_peligroso:
            _item_pre = _prematch_peligroso.group(0)
            logger.warning(
                "[Registro] Guard pre-LLM: item incomestible detectado: '%s'", _item_pre
            )
            return {
                "success": False,
                "tipo_detectado": "no_alimento",
                "mensaje": (
                    f"⚠️ '{_item_pre.capitalize()}' no es un alimento — no puedo registrarlo. "
                    f"Si crees que fue un error de escritura, corrígelo e intenta de nuevo."
                ),
            }

    if not goto_save:
        try:
            prompt = _PROMPT_COMIDA.format(mensaje=mensaje)
            raw = await _llamar_groq_con_excepciones(ia_engine, prompt, max_tokens=700, temp=0.0, model="llama-3.3-70b-versatile")
            datos = _parse_json(raw)
            if datos is None:
                await asyncio.sleep(1)
                raw = await _llamar_groq_con_excepciones(ia_engine, prompt, max_tokens=700, temp=0.0, model="llama-3.3-70b-versatile")
                datos = _parse_json(raw)
        except asyncio.TimeoutError as e:
            logger.error("[LLM Timeout in registrar_comida_llm]: %s", e)
            return {
                "success": False,
                "tipo_detectado": "error",
                "mensaje": "No pude conectar a tiempo con el asistente nutricional. Por favor intenta de nuevo en unos segundos."
            }
        except ConnectionError as e:
            logger.error("[LLM ConnectionError in registrar_comida_llm]: %s", e)
            return {
                "success": False,
                "tipo_detectado": "error",
                "mensaje": "El asistente nutricional está temporalmente ocupado o sin conexión. Por favor intenta de nuevo en unos momentos."
            }
        except Exception as e:
            logger.exception("[General Error in registrar_comida_llm LLM call]: %s", e)
            return {
                "success": False,
                "tipo_detectado": "error",
                "mensaje": "Hubo un inconveniente al procesar tu comida con el asistente. Por favor intenta de nuevo."
            }

    if not datos:
        return {
            "success": False,
            "tipo_detectado": "no_identificado",
            "mensaje": f"No pude procesar todos los alimentos, {perfil.first_name}. ¿Puedes repetirlo dividido por comida? Ej: 'en el desayuno comí X'",
        }

    _item_peligroso = _mensaje_contiene_no_alimento(mensaje)
    if _item_peligroso:
        logger.warning(
            "[Registro] Item incomestible/peligroso detectado: '%s' en mensaje: '%s'",
            _item_peligroso, mensaje[:100]
        )
        return {
            "success": False,
            "tipo_detectado": "no_alimento",
            "mensaje": (
                f"⚠️ '{_item_peligroso.capitalize()}' no es un alimento — no puedo registrarlo. "
                f"Si crees que fue un error de escritura, corrígelo e intenta de nuevo."
            ),
        }
    if datos.get("contiene_no_alimento"):
        logger.warning(
            "[Registro] LLM detectó material no comestible en: '%s'", mensaje[:100]
        )
        return {
            "success": False,
            "tipo_detectado": "no_alimento",
            "mensaje": (
                "⚠️ El mensaje describe algo que no es comestible. "
                "No puedo registrar un alimento mezclado con materiales peligrosos o no comestibles."
            ),
        }

    if datos.get("alimentos"):
        _aplicar_corte_pollo_brasa(datos["alimentos"], mensaje)
        datos["alimentos"] = [
            a for a in datos["alimentos"]
            if a.get("es_real", True) is not False
        ]
        if not goto_save:
            datos["alimentos"] = [
                a for a in datos["alimentos"]
                if _extraccion_tiene_base_textual(a.get("nombre", ""), mensaje)
            ]
            datos["alimentos"] = [
                a for a in datos["alimentos"]
                if not _es_solo_palabra_momento_dia(a.get("nombre", ""))
            ]
            datos["alimentos"] = [
                a for a in datos["alimentos"]
                if not _alimento_es_alucinacion(a, db)
            ]
            datos["alimentos"] = _filtrar_componentes_de_plato_compuesto(datos["alimentos"])
            datos["alimentos"] = _filtrar_contenedor_generico_con_ingredientes(datos["alimentos"], mensaje)
            datos["alimentos"] = _fusionar_alimentos_redundantes(datos["alimentos"])

    _items = datos.get("alimentos", [])

    _es_ambiguo = bool(_RX_CANTIDAD_AMBIGUA.search(mensaje or ""))
    if _es_ambiguo:
        _es_generico = True
        _GENERIC_FOOD_NAMES = {"comida", "alimento", "algo", "cena", "almuerzo", "desayuno", "piqueo", "picar", "cosa", "plato", "nutricion", "menu", "todo"}
        for item in _items:
            name_low = item.get("nombre", "").lower()
            if len(name_low) > 2 and not any(g in name_low for g in _GENERIC_FOOD_NAMES):
                _es_generico = False
                break
        
        if not _items or _es_generico:
            return {
                "success": False,
                "tipo_detectado": "no_identificado",
                "mensaje": f"No logré identificar qué alimento específico consumiste. ¿Podrías indicarme qué comiste exactamente?",
            }
        else:
            for item in _items:
                for field in ("prot_g", "carb_g", "grasa_g"):
                    if item.get(field) is not None:
                        item[field] = round(float(item[field]) * 0.7, 1)
                if item.get("kcal") is not None:
                    item["kcal"] = round(float(item["kcal"]) * 0.7, 1)
                if item.get("porcion_g") is not None:
                    item["porcion_g"] = round(float(item["porcion_g"]) * 0.7, 1)
    for _it_init in _items:
        _it_init["_factor_acumulado"] = 1.0

    _KCAL_MAX_POR_ITEM = 1100
    for _it_cap in _items:
        _p_cap = float(_it_cap.get("prot_g", 0) or 0)
        _c_cap = float(_it_cap.get("carb_g", 0) or 0)
        _g_cap = float(_it_cap.get("grasa_g", 0) or 0)
        _kcal_real_it = 4 * _p_cap + 4 * _c_cap + 9 * _g_cap
        if _kcal_real_it > _KCAL_MAX_POR_ITEM:
            _factor_it = _KCAL_MAX_POR_ITEM / _kcal_real_it
            for _campo_it in ("kcal", "prot_g", "carb_g", "grasa_g", "porcion_g"):
                if _it_cap.get(_campo_it) is not None:
                    _it_cap[_campo_it] = round(float(_it_cap[_campo_it]) * _factor_it, 1)
            logger.warning(
                "[Registro] Item '%s' excedia %s kcal reales (%s, via P/C/G) — re-escalado",
                _it_cap.get("nombre"), _KCAL_MAX_POR_ITEM, round(_kcal_real_it, 1),
            )

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
                f"⚠️ 'Causa' al final de la frase, dirigida a alguien (ej. 'arroz con pollo "
                f"causa', 'oe causa'), es una muletilla peruana informal (equivale a 'amigo'/"
                f"'compadre') — NO es el plato Causa. Solo es el plato si aparece como "
                f"'causa de [ingrediente]' o como el alimento principal descrito.\n"
                f"⚠️ PROHIBIDO repetir cualquiera de estos alimentos ya registrados: "
                f"{_nombres_ya_registrados} — si la palabra suelta se refiere a algo "
                f"que ya está en esa lista, NO lo incluyas de nuevo.\n"
                f"Si ninguna palabra suelta es un alimento adicional real y distinto, "
                f"responde alimentos vacío.\n"
                f'Responde SOLO JSON: {{"alimentos": [{{"nombre": "...", "es_real": true, '
                f'"cantidad": 1, "porcion_g": numero, "kcal": numero, "prot_g": numero, '
                f'"carb_g": numero, "grasa_g": numero}}]}}'
            )
            try:
                _raw_faltante = await _llamar_groq_con_excepciones(ia_engine, _prompt_faltante, max_tokens=250, temp=0.0, model="llama-3.3-70b-versatile")
                _datos_faltante = _parse_json(_raw_faltante)
            except Exception as _e_falt:
                logger.warning("[Registro] Error en llamada secundaria de faltante: %s", _e_falt)
                _datos_faltante = None
            if _datos_faltante and _datos_faltante.get("alimentos"):
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
                    datos["alimentos"] = [
                        a for a in datos["alimentos"]
                        if not _es_solo_palabra_momento_dia(a.get("nombre", ""))
                    ]
                    datos["alimentos"] = [
                        a for a in datos["alimentos"]
                        if not _alimento_es_alucinacion(a, db)
                    ]
                    datos["alimentos"] = _filtrar_componentes_de_plato_compuesto(datos["alimentos"])
                    datos["alimentos"] = _filtrar_contenedor_generico_con_ingredientes(datos["alimentos"], mensaje)
                    datos["alimentos"] = _fusionar_alimentos_redundantes(datos["alimentos"])
                    _items = datos["alimentos"]

    _msg_low_porcion = mensaje.lower() if mensaje else ""
    _factor_porcion = None
    if re.search(r'\bmedi[oa]\b|\bmitad\b', _msg_low_porcion):
        _factor_porcion = 0.5
    elif re.search(r'\btres cuartos\b|\btres cuartas partes\b|\b3/4\b', _msg_low_porcion):
        _factor_porcion = 0.75
    elif re.search(r'\bun cuarto\b|\bcuarta parte\b|\b1/4\b', _msg_low_porcion):
        _factor_porcion = 0.25
    elif re.search(r'\bdos tercios\b|\b2/3\b', _msg_low_porcion):
        _factor_porcion = 0.667
    elif re.search(r'\bun tercio\b|\btercera parte\b|\b1/3\b', _msg_low_porcion):
        _factor_porcion = 0.333
    elif re.search(r'porci[oó]n (chica|pequeñ[ao])|plato (chico|pequeñ[oa])', _msg_low_porcion):
        _factor_porcion = 0.65
    elif re.search(r'porci[oó]n (grande|extra)|plato grande|doble porci[oó]n', _msg_low_porcion):
        _factor_porcion = 1.4

    if _factor_porcion and len(_items) <= 2:
        for _it in _items:
            if _it.get("_corte_pollo_aplicado"):
                continue
            for _campo in ("porcion_g", "kcal", "prot_g", "carb_g", "grasa_g"):
                if _it.get(_campo) is not None:
                    _it[_campo] = round(float(_it[_campo]) * _factor_porcion, 1)
            _it["cantidad"] = 1
            _it["_factor_acumulado"] = _it.get("_factor_acumulado", 1.0) * _factor_porcion
            _it["_porcion_explicita_usuario"] = True
        for _campo_total in ("prot_total", "carb_total", "grasa_total", "kcal_total"):
            if datos.get(_campo_total) is not None:
                datos[_campo_total] = round(float(datos[_campo_total]) * _factor_porcion, 1)

    _match_gramos_explicitos = re.search(
        r'\b(\d+(?:[.,]\d+)?)\s*(?:gr|grs|gramos?|g)\b', _msg_low_porcion
    )
    if _match_gramos_explicitos and len(_items) == 1 and not _factor_porcion:
        _gramos_pedidos = float(_match_gramos_explicitos.group(1).replace(',', '.'))
        _it0 = _items[0]
        _porcion_actual = float(_it0.get("porcion_g") or 100)
        if abs(_porcion_actual - _gramos_pedidos) > _porcion_actual * 0.15:
            _factor_gramos = _gramos_pedidos / _porcion_actual
            logger.warning(
                "[Registro] '%sg' pedido pero LLM devolvio porcion_g=%s — re-escalando",
                _gramos_pedidos, _porcion_actual,
            )
            _it0["porcion_g"] = _porcion_actual
            for _campo in ("porcion_g", "kcal", "prot_g", "carb_g", "grasa_g"):
                if _it0.get(_campo) is not None:
                    _it0[_campo] = round(float(_it0[_campo]) * _factor_gramos, 1)
            _it0["_factor_acumulado"] = _it0.get("_factor_acumulado", 1.0) * _factor_gramos
            _it0["_porcion_explicita_usuario"] = True
            for _campo_total in ("prot_total", "carb_total", "grasa_total", "kcal_total"):
                if datos.get(_campo_total) is not None:
                    datos[_campo_total] = round(float(datos[_campo_total]) * _factor_gramos, 1)

    for _it_cache in _items:
        _cached = get_cached_macros(_it_cache.get("nombre", ""))
        if _cached and _cached.get("porcion_g"):
            _factor_escala = float(_it_cache.get("_factor_acumulado", 1.0))
            for _campo_c in ("kcal", "prot_g", "carb_g", "grasa_g"):
                if _cached.get(_campo_c) is not None:
                    _it_cache[_campo_c] = round(float(_cached[_campo_c]) * _factor_escala, 1)

    def _cantidad_clamp_agg(a: dict) -> float:
        try:
            q = float(a.get("cantidad", 1) or 1)
        except (TypeError, ValueError):
            q = 1.0
        if 0 < q < 1:
            return q
        return max(1, min(int(q), 10))

    _prot_items  = sum(float(a.get("prot_g",  0) or 0) * _cantidad_clamp_agg(a) for a in _items)
    _carb_items  = sum(float(a.get("carb_g",  0) or 0) * _cantidad_clamp_agg(a) for a in _items)
    _grasa_items = sum(float(a.get("grasa_g", 0) or 0) * _cantidad_clamp_agg(a) for a in _items)
    if not _items:
        return {
            "success": False,
            "tipo_detectado": "no_identificado",
            "mensaje": f"No identifiqué ningún alimento, {perfil.first_name}. ¿Qué comiste exactamente?",
        }

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
        kcal = kcal_desde_macros if kcal_desde_macros > 0 else kcal_llm

    _KCAL_MAX_RAZONABLE = 5000
    _factor_cap = 1.0
    advertencia_cantidad = None
    _factor_momento = 1.0
    advertencia_momento = None
    
    _es_ambiguo = bool(_RX_CANTIDAD_AMBIGUA.search(mensaje or ""))
    
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
    elif _es_ambiguo:
        advertencia_cantidad = (
            "⚠️ Detecté cierta duda en la cantidad. Registré una estimación conservadora; si no es correcto, "
            "puedes decirme para ajustarlo."
        )

    _msg_low_momento = mensaje.lower() if mensaje else ""
    _momentos_detectados = set()
    if any(k in _msg_low_momento for k in ("desayuno", "desayuné", "desayune")):
        _momentos_detectados.add("DESAYUNO")
    if any(k in _msg_low_momento for k in ("merienda", "snack")):
        _momentos_detectados.add("MERIENDA")
    if any(k in _msg_low_momento for k in ("cena", "cené", "cene")):
        _momentos_detectados.add("CENA")
    if any(k in _msg_low_momento for k in ("almuerzo", "almorcé", "almorce")):
        _momentos_detectados.add("ALMUERZO")
    _momento_registro = next(iter(_momentos_detectados)) if len(_momentos_detectados) == 1 else None
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

    _SOPA_KW = ("sopa ", "caldo ", "crema de ", "sopa de ", " sopa", "caldito")
    _is_sopa = any(k in _msg_low_momento for k in _SOPA_KW)
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
    def _nombre_con_cantidad(a: dict) -> str:
        n = a.get("nombre", "")
        try:
            q = int(float(a.get("cantidad", 1) or 1))
        except (TypeError, ValueError):
            q = 1
        return f"{n} ×{q}" if q > 1 else n
    nombres = [_nombre_con_cantidad(a) for a in alimentos_raw if a.get("nombre")]

    from app.core.utils import get_peru_date
    hoy = get_peru_date()
    prog = _get_or_create_progreso(db, perfil.id, hoy, plan_hoy)

    if _filas_previas_correccion:
        _kcal_previo  = sum(float(f.kcal or 0) for f in _filas_previas_correccion)
        _prot_previo  = sum(float(f.proteina_g or 0) for f in _filas_previas_correccion)
        _carb_previo  = sum(float(f.carbohidratos_g or 0) for f in _filas_previas_correccion)
        _gras_previo  = sum(float(f.grasas_g or 0) for f in _filas_previas_correccion)
        prog.calorias_consumidas      = max(0, (prog.calorias_consumidas or 0) - _kcal_previo)
        prog.proteinas_consumidas     = max(0.0, round((prog.proteinas_consumidas or 0) - _prot_previo, 1))
        prog.carbohidratos_consumidos = max(0.0, round((prog.carbohidratos_consumidos or 0) - _carb_previo, 1))
        prog.grasas_consumidas        = max(0.0, round((prog.grasas_consumidas or 0) - _gras_previo, 1))
        for _fila_vieja in _filas_previas_correccion:
            db.delete(_fila_vieja)
        logger.info(
            "[Registro] Corrección aplicada — %s fila(s) revertidas (%.0f kcal)",
            len(_filas_previas_correccion), _kcal_previo,
        )

    prog.calorias_consumidas      = int((prog.calorias_consumidas or 0) + kcal)
    prog.proteinas_consumidas     = round((prog.proteinas_consumidas or 0) + prot, 1)
    prog.carbohidratos_consumidos = round((prog.carbohidratos_consumidos or 0) + carb, 1)
    prog.grasas_consumidas        = round((prog.grasas_consumidas or 0) + grasa, 1)

    from app.models.comida_registro import ComidaRegistro
    n_items = max(1, len(alimentos_raw))
    for item in alimentos_raw:
        nombre_item = item.get("nombre", nombres[0] if nombres else "Alimento")
        try:
            _cantidad_cruda = float(item.get("cantidad", 1) or 1)
        except (TypeError, ValueError):
            _cantidad_cruda = 1.0
        _factor_fraccion = _cantidad_cruda if 0 < _cantidad_cruda < 1 else 1.0
        try:
            cantidad_item = int(_cantidad_cruda)
        except (TypeError, ValueError):
            cantidad_item = 1
        cantidad_item = max(1, min(cantidad_item, 10))
        _factor_total = _factor_cap * _factor_momento * _factor_sopa * _factor_fraccion
        p_item = round(float(item.get("prot_g", prot / n_items)) * _factor_total, 1)
        c_item = round(float(item.get("carb_g", carb / n_items)) * _factor_total, 1)
        g_item = round(float(item.get("grasa_g", grasa / n_items)) * _factor_total, 1)
        k_item = round(4 * p_item + 4 * c_item + 9 * g_item, 1)

        _porcion_item_val = float(item.get("porcion_g", 100) or 100)
        if k_item < 0 or p_item < 0 or c_item < 0 or g_item < 0:
            logger.warning(
                "[Registro] Valores negativos para '%s' (kcal=%s p=%s c=%s g=%s) — recortados a 0",
                nombre_item, k_item, p_item, c_item, g_item,
            )
            p_item, c_item, g_item = max(p_item, 0), max(c_item, 0), max(g_item, 0)
            k_item = round(4 * p_item + 4 * c_item + 9 * g_item, 1)
        _densidad_kcal_g = (k_item / _porcion_item_val) if _porcion_item_val > 0 else 0
        if _densidad_kcal_g > 9.5:
            _factor_densidad = 9.5 / _densidad_kcal_g
            logger.warning(
                "[Registro] Densidad calorica imposible para '%s': %.1f kcal/g (max teorico 9) — re-escalado",
                nombre_item, _densidad_kcal_g,
            )
            p_item = round(p_item * _factor_densidad, 1)
            c_item = round(c_item * _factor_densidad, 1)
            g_item = round(g_item * _factor_densidad, 1)
            k_item = round(4 * p_item + 4 * c_item + 9 * g_item, 1)

        if not item.get("_porcion_explicita_usuario"):
            cache_macros(nombre_item, {
                "nombre": nombre_item, "kcal": k_item,
                "prot_g": p_item, "carb_g": c_item, "grasa_g": g_item,
                "porcion_g": _porcion_item_val,
            })
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
    nombres_completos = nombres

    meta      = float(plan_hoy.get("calorias_dia", 2000))
    consumido = float(prog.calorias_consumidas)
    
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

    restante  = max(0.0, meta - consumido + quemado)

    from app.core.notification_scheduler import notificar_si_excede_meta
    notificar_si_excede_meta(perfil, prog, meta, quemado=quemado)
    db.commit()

    _es_vegano_ctx = any("vegano" in c.lower() for c in ctx.condiciones_medicas)
    _es_veg_ctx = any("vegetariano" in c.lower() for c in ctx.condiciones_medicas)
    _dieta_tipo = "Vegano" if _es_vegano_ctx else ("Vegetariano" if _es_veg_ctx else "Normal")
    alerta_dieta = _detectar_conflicto_dieta(nombres, _dieta_tipo, ctx.condiciones_medicas)

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
    _max = 600 if len(mensaje.split()) > 15 else 300
    try:
        raw = await _llamar_groq_con_excepciones(ia_engine, prompt, max_tokens=_max, temp=0.0)
        resultado = _parse_json(raw)
    except asyncio.TimeoutError as e:
        logger.error("[LLM Timeout in registrar_ejercicio_llm]: %s", e)
        return {
            "success": False,
            "tipo_detectado": "error",
            "mensaje": "Hubo un inconveniente de conexión con el servidor (Timeout). No pude registrar tu ejercicio, por favor intenta de nuevo."
        }
    except ConnectionError as e:
        logger.error("[LLM ConnectionError in registrar_ejercicio_llm]: %s", e)
        return {
            "success": False,
            "tipo_detectado": "error",
            "mensaje": "Hubo un inconveniente de conexión con el servidor. No pude registrar tu ejercicio, por favor intenta de nuevo."
        }
    except Exception as e:
        logger.exception("[General Error in registrar_ejercicio_llm]: %s", e)
        return {
            "success": False,
            "tipo_detectado": "error",
            "mensaje": "Hubo un inconveniente con el servidor. No pude registrar tu ejercicio, por favor intenta de nuevo."
        }

    if isinstance(resultado, dict):
        ejercicios_raw = [resultado]
    elif isinstance(resultado, list):
        ejercicios_raw = resultado
    else:
        ejercicios_raw = []

    ejercicios_raw = [e for e in ejercicios_raw
                      if e.get("encontrado", True) and e.get("ejercicio")]
    ejercicios_raw = [
        e for e in ejercicios_raw
        if _extraccion_tiene_base_textual(e.get("ejercicio", ""), mensaje, es_ejercicio=True)
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

    _m_dur_total = re.search(
        r'(\d+(?:[.,]\d+)?)\s*minutos?\s*(?:en\s+total)\b'
        r'|\ben\s+total\b[^.]*?(\d+(?:[.,]\d+)?)\s*minutos?',
        mensaje or "", re.IGNORECASE,
    )
    if _m_dur_total and len(ejercicios_raw) > 1:
        _total_declarado = float((_m_dur_total.group(1) or _m_dur_total.group(2)).replace(',', '.'))
        _duraciones_actuales = [float(e.get("duracion_min", 0) or 0) for e in ejercicios_raw]
        _n_con_total_completo = sum(1 for d in _duraciones_actuales if abs(d - _total_declarado) < 1)
        if _n_con_total_completo >= 2:
            _duracion_repartida = round(_total_declarado / len(ejercicios_raw), 1)
            for _e_dur in ejercicios_raw:
                _e_dur["duracion_min"] = _duracion_repartida
                _e_dur["kcal_quemadas"] = 0
            logger.info(
                "[Registro] Duracion total (%smin) repartida entre %d ejercicios -> %smin cada uno",
                _total_declarado, len(ejercicios_raw), _duracion_repartida,
            )

    kcal_total = 0.0
    ejercicios_guardados = []

    from app.services.asistente.asistente_ejercicio import resolver_met_mets_gym

    for datos in ejercicios_raw:
        nombre   = datos["ejercicio"]
        duracion = float(datos.get("duracion_min", 0) or 0)
        series   = datos.get("series")
        reps     = datos.get("reps")
        peso_ej  = datos.get("peso_kg")
        _cat_key, _cat_met = resolver_met_mets_gym((f"{nombre} {mensaje or ''}").lower())
        if _cat_met:
            met = float(_cat_met)
            _met_determinista = True
        else:
            met = float(datos.get("met", 5.0) or 5.0)
            _met_determinista = False
        intensidad = "Alta" if met >= 8 else ("Media" if met >= 5 else "Baja")
        if duracion <= 0 and series:
            duracion = round(int(series) * 5, 1)
        kcal_formula = round(met * peso_kg * 3.5 / 200 * duracion, 1)
        kcal_llm     = round(float(datos.get("kcal_quemadas", 0) or 0), 1)
        if _met_determinista:
            kcal = kcal_formula
        elif kcal_formula <= 0:
            kcal = kcal_llm if kcal_llm > 0 else 0.0
        else:
            kcal = kcal_formula if kcal_llm > kcal_formula * 2.5 or kcal_llm < kcal_formula * 0.3 else kcal_llm

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

    prog = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == perfil.id,
        ProgresoCalorias.fecha == hoy,
    ).first()
    if prog:
        prog.calorias_quemadas = round((prog.calorias_quemadas or 0) + kcal_total, 1)
    db.commit()

    quemado_total = round(float(prog.calorias_quemadas if prog else kcal_total), 1)

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


_INTROS_RECO = [
    "Para lo que buscas, te van bien",
    "Algo que te puede ir bien esta",
    "Tres opciones que se ajustan a lo que pediste",
    "Basándome en lo que me dijiste, te propongo",
    "Aquí tienes opciones que encajan con lo que buscas",
]

def _construir_mensaje_natural_reco(
    platos_limpios: list,
    palabra_evitada: str | None,
    condicion_relevante: str | None,
    estilo_evitado_momento: str | None,
    advertencia_meta: str | None = None,
) -> str:
    import random
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

    _prefijo_meta = f"{advertencia_meta} " if advertencia_meta else ""

    if palabra_evitada and condicion_relevante:
        return (
            f"{_prefijo_meta}Por tu condición de {condicion_relevante} evité incluir {palabra_evitada}, "
            f"así que te van bien {lista_natural} — {kcal_txt}."
        )

    intro = random.choice(_INTROS_RECO)
    return f"{_prefijo_meta}{intro}: {lista_natural} — {kcal_txt}."


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
    ctx: Optional[UserContext] = None,
) -> str:
    """Genera recomendación vía LLM. Para comida: cachea los macros exactos de
    cada plato recomendado → cuando el usuario lo registre, se usarán los mismos
    valores (consistencia perfecta recomendación ↔ registro)."""
    ctx = _asegurar_contexto(perfil, consumido, quemado, plan_macros, ctx)
    objetivo = ctx.objetivo_normalizado
    _balance_reco = _calcular_balance_meta(consumido, meta, quemado, objetivo, mensaje)
    restante = _balance_reco["restante"]
    _excedido_kcal_reco = _balance_reco["excedido"]
    _pct_reco_balance = _balance_reco["pct"]
    _es_vegano = any("vegano" in c.lower() for c in ctx.condiciones_medicas)
    _es_vegetariano = any("vegetariano" in c.lower() for c in ctx.condiciones_medicas)
    dieta = "Vegano" if _es_vegano else ("Vegetariano" if _es_vegetariano else "Normal")
    condiciones = ", ".join(ctx.condiciones_medicas) or "ninguna"

    if modo == "ejercicio":
        from app.services.rutina_service import (
            _LESIONES_SUSTITUCION, _detectar_lesiones, filtrar_lesiones_activas,
        )
        _condiciones_lista_ej = list(getattr(perfil, "medical_conditions", None) or [])
        _texto_historial_ej = " ".join(
            str(h.get("content", "")) for h in (historial or [])
        )
        _lesiones_candidatas_ej = _detectar_lesiones(
            _condiciones_lista_ej + [mensaje or "", _texto_historial_ej]
        )
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
            nombre=ctx.nombre,
            objetivo=ctx.objetivo_normalizado,
            condiciones=condiciones,
            contexto_lesion=_contexto_lesion,
            mensaje=mensaje,
        )

        try:
            respuesta_ej = await _llamar_groq_con_excepciones(ia_engine, prompt, max_tokens=300, temp=0.7)

            if _lesiones_activas:
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
                    respuesta_ej = await _llamar_groq_con_excepciones(
                        ia_engine, _prompt_retry_ej, max_tokens=200, temp=0.3
                    )

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
        except asyncio.TimeoutError as e:
            logger.error("[LLM Timeout in recomendacion ejercicio]: %s", e)
            respuesta_ej = "Por tu seguridad y debido a un timeout del servidor, te recomiendo realizar una rutina de movilidad suave en casa y caminata de 15 minutos a ritmo cómodo."
        except ConnectionError as e:
            logger.error("[LLM ConnectionError in recomendacion ejercicio]: %s", e)
            respuesta_ej = "Por tu seguridad y debido a un problema de conexión con el servidor, te recomiendo realizar una rutina de movilidad suave en casa y caminata de 15 minutos a ritmo cómodo."
        except Exception as e:
            logger.exception("[General Error in recomendacion ejercicio]: %s", e)
            respuesta_ej = "Por tu seguridad, te recomiendo realizar una rutina de movilidad suave en casa y caminata de 15 minutos a ritmo cómodo."

        return respuesta_ej

    import re as _re_reco
    from app.core.utils import get_peru_now as _get_peru_now_reco

    _MOMENTO_KEYWORDS_RECO = {
        "CENA":      ["cenar", "cena", "ceno", "cenare", "cenaré"],
        "MERIENDA":  ["merienda", "meriendo", "merendar", "snack", "media tarde", "media mañana", "antojo", "tarde"],
        "ALMUERZO":  ["almorzar", "almuerzo", "mediodía", "mediodia"],
        "DESAYUNO":  ["desayunar", "desayuno", "mañana"],
    }
    _msg_low_reco = mensaje.lower() if mensaje else ""

    from app.services.recomendador_platos import _tokens_prohibidos, _CONDICION_TOKENS
    _condiciones_lista_reco = ctx.condiciones_medicas
    _tokens_dieta_msg = _tokens_prohibidos(_condiciones_lista_reco)

    _evitados_msg: list[tuple[str, str]] = []
    for _t in _tokens_dieta_msg:
        if _t in _msg_low_reco:
            _cond = next(
                (c for c in _condiciones_lista_reco if _t in _CONDICION_TOKENS.get(c, set())),
                None,
            )
            if _cond:
                _evitados_msg.append((_t, _cond))

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

    if not momento_reco and any(kw in _msg_low_reco for kw in ("noche", "nocturno")):
        _hora_noche = _get_peru_now_reco().hour
        momento_reco = "CENA" if 18 <= _hora_noche <= 21 else "MERIENDA"
    elif not momento_reco and "madrugada" in _msg_low_reco:
        momento_reco = "MERIENDA"

    if not momento_reco:
        _hora = _get_peru_now_reco().hour
        if 5 <= _hora <= 9:
            momento_reco = "DESAYUNO"
        elif 10 <= _hora <= 14:
            momento_reco = "ALMUERZO"
        elif 15 <= _hora <= 17:
            momento_reco = "MERIENDA"
        elif 18 <= _hora <= 21:
            momento_reco = "CENA"
        else:
            momento_reco = "MERIENDA"

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
                    HistorialRecomendacion.client_id == ctx.perfil_id,
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

            from app.services.recomendador_platos import _tokens_prohibidos
            _tokens_dieta_reco_knn = _tokens_prohibidos(ctx.condiciones_medicas)
            if _tokens_dieta_reco_knn:
                _candidatos_knn = [
                    c for c in _candidatos_knn
                    if not any(t in c["alimento"].lower() for t in _tokens_dieta_reco_knn)
                ]
        except Exception as e:
            logger.warning("[Reco] KNN candidatos no disponibles: %s", e)

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
            _resp_eval = await _llamar_groq_con_excepciones(ia_engine, _prompt_eval, max_tokens=150, temp=0.0)
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

    _condiciones_list_reco = ctx.condiciones_medicas
    _condiciones_str_reco = " ".join(_condiciones_list_reco).lower()
    from app.services.recomendador_platos import _detectar_dieta_en_mensaje
    _condiciones_msg_reco = _detectar_dieta_en_mensaje(mensaje)
    es_vegano_reco = (
        "vegano" in dieta.lower() or "vegetariano" in dieta.lower()
        or "vegano" in _condiciones_str_reco or "vegetariano" in _condiciones_str_reco
        or "Vegano" in _condiciones_msg_reco or "Vegetariano" in _condiciones_msg_reco
    )

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
    _ing_match = _re_reco.search(
        r'(?:con|de|que\s+tenga|a\s+base\s+de)\s+([a-záéíóúüñ]+(?:\s+[a-záéíóúüñ]+)?)'
        r'(?=\s+(?:para|en|hoy|ahora|al|por)\b|[.,?]|$)',
        _msg_low_reco,
    )
    pref_ingrediente_reco = ""
    if _ing_match:
        _ing_detectado = _ing_match.group(1).strip().rstrip('.,?')
        _GENERIC_IGNORE_WORDS = {
            "hoy", "comer", "ti", "mi", "algo", "uno", "plato", "poco",
            "dia", "dias", "tarde", "mañana", "noche", "antes", "despues",
            "durante", "luego", "meta", "caloria", "calorias", "kcal",
            "entreno", "ejercicio", "rutina", "dieta", "alimento", "comida",
            "ingrediente", "entrenar", "entrenando", "peso", "pesas",
            "grasa", "grasas", "proteina", "carbohidrato", "fibra", "macro", "macros",
            "desayuno", "almuerzo", "cena", "merienda"
        }
        _palabras_ing = set(_ing_detectado.split())
        _es_no_comida = False
        for _w in _palabras_ing:
            _w_norm = _normalizar_nombre(_w)
            if _w_norm in _GENERIC_IGNORE_WORDS:
                _es_no_comida = True
                break
            if any(_w_norm.startswith(k) or k in _w_norm for k in _TEMA_EJERCICIO_KW):
                _es_no_comida = True
                break
            if any(_w_norm.startswith(k) or k in _w_norm for k in _TEMA_NUTRICION_KW):
                _es_no_comida = True
                break

        if not _es_no_comida and len(_ing_detectado) > 2:
            pref_ingrediente_reco = (
                f"⚠️ El usuario pidió ESPECÍFICAMENTE algo con: **{_ing_detectado}**. "
                f"Esto tiene prioridad sobre la variedad: los 3 platos DEBEN incluir "
                f"'{_ing_detectado}' de alguna forma (como ingrediente principal o "
                f"visible en la preparación) — no lo menciones en uno solo y dejes "
                f"los otros 2 libres."
            )


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

    restriccion_dieta_reco = (
        "VEGANO/VEGETARIANO: PROHIBIDO carnes, pollo, pescado, mariscos, lácteos animales. "
        "Solo plantas, legumbres, granos, frutas, tofu, soja, hongos."
    ) if es_vegano_reco else ""

    _MACRO_RESTRICCIONES = {
        r'sin\s+carbohidrato|bajo\s+en\s+carbohidrato|sin\s+carb\b|low\s+carb|keto': (
            "carbohidratos",
            "🚨 RESTRICCIÓN DEL USUARIO: SIN CARBOHIDRATOS. "
            "Propón solo platos con máximo 15g de carbohidratos totales por plato. "
            "Aplica tu conocimiento nutricional — sabes cuáles alimentos son ricos en carbohidratos. Evítalos todos.",
        ),
        r'sin\s+grasa|bajo\s+en\s+grasa|sin\s+aceite': (
            "grasas",
            "🚨 RESTRICCIÓN DEL USUARIO: SIN GRASA / BAJO EN GRASA. "
            "Propón solo platos con máximo 5g de grasa total por plato. "
            "Aplica tu conocimiento nutricional — sabes cuáles alimentos son grasos. Evítalos todos.",
        ),
        r'sin\s+azucar|sin\s+az[uú]car|sin\s+dulce|bajo\s+en\s+az[uú]car': (
            "azúcar",
            "🚨 RESTRICCIÓN DEL USUARIO: SIN AZÚCAR. "
            "Propón solo opciones sin azúcares añadidos ni fuentes dulces evidentes. "
            "Aplica tu conocimiento — sabes qué lleva azúcar. Evítalo todo.",
        ),
    }
    for _patron_macro, (_nombre_macro, _restriccion_txt) in _MACRO_RESTRICCIONES.items():
        if _re_reco.search(_patron_macro, _msg_low_reco) and not exclusion_reco:
            exclusion_reco = _restriccion_txt
            if "carbohidrato" in _nombre_macro and es_vegano_reco:
                restriccion_dieta_reco = (
                    "🚨 VEGANO + SIN CARBOHIDRATOS: solo opciones vegetales con máximo 15g de carbs por plato. "
                    "Tofu, tempeh, hongos, verduras sin almidón, semillas, aguacate. "
                    "Sabes cuáles vegetales tienen almidón o carbs elevados — evítalos sin que yo tenga que listártelos."
                )
            break

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

    _masa_muscular_match = _balance_reco["es_masa_muscular"]
    _display_floor_mm = {
        "DESAYUNO": 450.0, "MERIENDA": 280.0, "CENA": 520.0,
    }.get(momento_reco, 500.0)
    _restante_display = max(restante, _display_floor_mm) if _masa_muscular_match else restante
    _MM_POR_MOMENTO = {
        "DESAYUNO": (
            "OBJETIVO MASA MUSCULAR: DESAYUNO dentro del rango del momento (250-450 kcal). "
            "Rico en proteína: huevos, pan con queso fresco, avena con leche, quinua con "
            "leche, yogur griego con granola. NO fuerces un plato de fondo ni superes 450 kcal."
        ),
        "MERIENDA": (
            "OBJETIVO MASA MUSCULAR: MERIENDA/snack dentro del rango del momento (80-280 kcal). "
            "Snack alto en proteína: yogur griego, pan con queso fresco, tostada con mantequilla "
            "de maní, huevo sancochado, vaso de leche con quinua. NO conviertas la merienda en "
            "un plato de fondo ni superes 280 kcal."
        ),
        "CENA": (
            "OBJETIVO MASA MUSCULAR: CENA ligera pero proteica dentro del rango del momento "
            "(200-520 kcal): pescado a la plancha, pollo a la plancha, tortilla de huevo, "
            "menestra con quinua. NO superes 520 kcal — es la comida nocturna."
        ),
    }
    _masa_muscular_txt = (
        _MM_POR_MOMENTO.get(
            momento_reco,
            "OBJETIVO MASA MUSCULAR: para ganar masa muscular se requiere un aporte calórico ALTO. "
            "Propón 3 platos completos de 400-700 kcal cada uno con ALTA proteína (≥25g por plato). "
            "Una ingesta calórica ligeramente superior al mantenimiento diario es CORRECTA y deseable "
            "para este objetivo — NO limites los platos al déficit restante del día. "
            "Usa fuentes de proteína magra: pollo a la plancha, pescado, res magra, huevos, "
            "menestras con quinua. Incluye carbohidratos de calidad (arroz, papa, quinua) como "
            "fuente de energía para el entrenamiento.",
        )
    ) if _masa_muscular_match else ""

    _balance_meta_txt = _balance_reco["bloque_balance"]
    advertencia_meta_natural = _balance_reco["advertencia_natural"]
    if _excedido_kcal_reco > 0 and not _masa_muscular_match:
        _balance_meta_txt += (
            "Aun así, si tiene hambre real, propón opciones livianas (no platos de fondo "
            "pesados) — no te niegues a recomendar, solo seas consciente del exceso.\n\n"
        )

    _DIETA_NO_MEDICA = {"vegano", "vegetariano", "vegan", "vegetarian"}
    _condiciones_sin_dieta = ", ".join(
        c.strip() for c in (condiciones or "").split(",")
        if c.strip().lower() not in _DIETA_NO_MEDICA
    )
    _condiciones_medicas_txt = ""
    if _condiciones_sin_dieta and _condiciones_sin_dieta.lower() != "ninguna":
        condiciones = _condiciones_sin_dieta
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
            _restricciones_raw = await _llamar_groq_con_excepciones(
                ia_engine, _prompt_med, max_tokens=200, temp=0.0
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
            _condiciones_medicas_txt = obtener_fallback_restricciones_medicas(ctx.condiciones_medicas)

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

    _knn_candidatos_txt = ""
    if _top_knn and not exclusion_reco:
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

    _restr_momento = (
        (_RESTRICCIONES_MOMENTO_VEGANO if es_vegano_reco else _RESTRICCIONES_MOMENTO_RECO)
        .get(momento_reco, "")
    )

    _contexto_dieta = restriccion_dieta_reco or (
        f"Objetivo: {objetivo}." if not es_vegano_reco else ""
    )
    _restricciones_extra = " | ".join(filter(None, [
        _condiciones_medicas_txt.replace("\n", " ").strip() if _condiciones_medicas_txt else "",
        exclusion_reco.replace("\n", " ").strip() if exclusion_reco else "",
        _masa_muscular_txt.replace("\n", " ").strip() if _masa_muscular_txt else "",
        objetivo_proteina_reco.replace("\n", " ").strip() if objetivo_proteina_reco else "",
        pref_ingrediente_reco.replace("\n", " ").strip() if pref_ingrediente_reco else "",
    ]))
    _prompt_reco_comida = (
        f"Eres nutricionista del Gimnasio World Light Lambayeque. Habla directamente a {perfil.first_name}, en español natural y breve.\n"
        f"{perfil.first_name} tiene {round(_restante_display)} kcal disponibles para {momento_reco.lower()} hoy.\n"
        + (f"\n⛔ REGLAS DEL MOMENTO ({momento_reco}): {_restr_momento}\n" if _restr_momento else "")
        + (f"Ejemplos de {momento_reco.lower()} en Perú: {_ref_platos}.\n" if _ref_platos else "")
        + (f"\nRESTRICCIONES OBLIGATORIAS (respétalas todas): {_contexto_dieta}\n" if _contexto_dieta else "")
        + (f"{_restricciones_extra}\n" if _restricciones_extra else "")
        + (f"No repetir: {_ya_sugeridos_txt.replace('PLATOS YA RECOMENDADOS (NO repetir):', '').strip()}\n" if _ya_sugeridos_txt else "")
        + (f"{_knn_candidatos_txt.strip()}\n" if _knn_candidatos_txt else "")
        + (f"{_balance_meta_txt.strip()}\n" if _balance_meta_txt else "")
        + f"\n{perfil.first_name} dice: \"{mensaje[:300]}\"\n\n"
        f"PRIMERO escribe esta línea exacta con los datos:\n"
        f"PLATOS: Nombre1 (~XXX kcal,P:Xg,C:Yg,G:Zg)|Nombre2 (~XXX kcal,P:Xg,C:Yg,G:Zg)|Nombre3 (~XXX kcal,P:Xg,C:Yg,G:Zg)\n"
        f"LUEGO en la siguiente línea escribe UNA oración natural y corta con los 3 nombres y sus kcal (ej: 'Para tu cena te propongo X (~300 kcal), Y (~400 kcal) y Z (~350 kcal).'). SIN introducciones largas."
    )

    _RE_PLATO_DATA = _re_reco.compile(
        r'([^()|]{3,80}?)\s*\(~?(\d+(?:\.\d+)?)\s*kcal[,;]?\s*'
        r'P\s*:?\s*(\d+(?:\.\d+)?)\s*g[,;]?\s*'
        r'C\s*:?\s*(\d+(?:\.\d+)?)\s*g[,;]?\s*'
        r'G\s*:?\s*(\d+(?:\.\d+)?)\s*g\)',
        _re_reco.IGNORECASE,
    )

    def _extraer_texto_y_platos(resp: str):
        if not resp:
            return "", []
        platos_line, texto_lines = "", []
        for line in resp.strip().splitlines():
            if line.strip().upper().startswith("PLATOS:"):
                platos_line = line.strip()
            else:
                texto_lines.append(line)
        texto = "\n".join(texto_lines).strip()
        platos = []
        if platos_line:
            for part in platos_line[len("PLATOS:"):].strip().split("|"):
                m = _RE_PLATO_DATA.search(part)
                if m:
                    platos.append((m.group(1).strip(), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
                else:
                    nombre = part.split("(")[0].strip().strip("-•* ")
                    if len(nombre) > 3:
                        platos.append((nombre, 0.0, 0.0, 0.0, 0.0))
        _parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
        if len(_parrafos) > 1 and platos:
            _con_platos = [p for p in _parrafos if any(n[0].lower()[:8] in p.lower() for n in platos)]
            if _con_platos:
                texto = _con_platos[-1]
        if platos and texto:
            _oraciones = _re_reco.split(r'(?<=[.!?])\s+', texto)
            _con_nombre = [s for s in _oraciones if any(n[0].lower()[:8] in s.lower() for n in platos)]
            if _con_nombre:
                texto = " ".join(_con_nombre)
        _nombres_en_texto = any(p[0].lower()[:8] in texto.lower() for p in platos) if platos else False
        if platos and (not texto or not _nombres_en_texto):
            partes = [
                f"{n} (~{k:.0f} kcal)" if k else n
                for n, k, *_ in platos
            ]
            sufijo = (", ".join(partes[:-1]) + " y " + partes[-1]) if len(partes) > 1 else partes[0]
            if not texto:
                texto = sufijo + "."
            elif texto.rstrip().endswith(":"):
                texto = texto.rstrip() + " " + sufijo + "."
            else:
                texto = texto.rstrip(".").rstrip() + ": " + sufijo + "."
        return texto, platos

    from app.services.recomendador_platos import _tokens_prohibidos, _detectar_dieta_en_mensaje
    _condiciones_dieta_check = list(getattr(perfil, "medical_conditions", None) or [])
    for _cond_msg in _detectar_dieta_en_mensaje(mensaje):
        if _cond_msg not in _condiciones_dieta_check:
            _condiciones_dieta_check.append(_cond_msg)
    _tokens_dieta_check = _tokens_prohibidos(_condiciones_dieta_check) | _tokens_exclusion_msg
    _CALIFICADORES_SEGUROS_DIETA = (
        "deslactosado", "deslactosada", "sin lactosa", "sin azúcar", "sin azucar",
        "sin gluten", "light", "diet",
    )

    def _linea_viola_dieta(linea: str) -> bool:
        ll = linea.lower()
        if any(c in ll for c in _CALIFICADORES_SEGUROS_DIETA):
            return False
        return any(t in ll for t in _tokens_dieta_check)

    _texto_usuario, _platos_data = "", []
    try:
        respuesta_llm_reco = await _llamar_groq_con_excepciones(
            ia_engine, _prompt_reco_comida, max_tokens=250, temp=0.6
        )
        _texto_usuario, _platos_data = _extraer_texto_y_platos(respuesta_llm_reco)

        if _tokens_dieta_check and any(_linea_viola_dieta(l) for l in _texto_usuario.split("\n")):
            _tokens_detectados = sorted({
                t for l in _texto_usuario.split("\n")
                for t in _tokens_dieta_check
                if t in l.lower() and not any(c in l.lower() for c in _CALIFICADORES_SEGUROS_DIETA)
            })
            logger.warning("[Reco] Token inadecuado (%s) — reintentando", _tokens_detectados)
            _prompt_retry = (
                f"Eres nutricionista. {perfil.first_name} tiene restricción ESTRICTA: "
                + (_contexto_dieta or "opciones saludables")
                + (f" | {exclusion_reco}" if exclusion_reco else "")
                + (f"\nNO uses: {', '.join(_tokens_detectados)}.\n" if _tokens_detectados else "\n")
                + (f"\n⛔ REGLAS DEL MOMENTO ({momento_reco}): {_restr_momento}\n" if _restr_momento else "")
                + (f"Ejemplos de {momento_reco.lower()} en Perú: {_ref_platos}.\n" if _ref_platos else "")
                + f"Sugiere 3 platos peruanos reales para {momento_reco.lower()} ({round(_restante_display)} kcal) "
                + f"respondiendo: \"{mensaje[:200]}\"\n"
                + f"1-2 oraciones naturales, luego:\n"
                + f"PLATOS: Nombre1 (~XXX kcal,P:Xg,C:Yg,G:Zg)|Nombre2 (~XXX kcal,P:Xg,C:Yg,G:Zg)|Nombre3 (~XXX kcal,P:Xg,C:Yg,G:Zg)"
            )
            respuesta_llm_reco = await _llamar_groq_con_excepciones(
                ia_engine, _prompt_retry, max_tokens=250, temp=0.3
            )
            _texto_usuario, _platos_data = _extraer_texto_y_platos(respuesta_llm_reco)

    except asyncio.TimeoutError as e:
        logger.error("[LLM Timeout in recomendacion comida]: %s", e)
        _texto_usuario = "Tuve un problema técnico. Intenta de nuevo o consulta con tu nutricionista."
    except ConnectionError as e:
        logger.error("[LLM ConnectionError in recomendacion comida]: %s", e)
        _texto_usuario = "Tuve un problema de conexión. Intenta de nuevo en un momento."
    except Exception as e:
        logger.exception("[General Error in recomendacion comida LLM flow]: %s", e)
        _texto_usuario = "Hubo un error al generar sugerencias. Intenta de nuevo."

    if _platos_data:
        for _np, _kp, _pp, _cp, _gp in _platos_data:
            cache_macros(_np, {"nombre": _np, "kcal": _kp, "prot_g": _pp, "carb_g": _cp, "grasa_g": _gp})
        _persistir_historial_recomendaciones(db, perfil, momento_reco, _platos_data)

    _prefijo = f"{advertencia_meta_natural} " if advertencia_meta_natural else ""
    return (f"{_prefijo}{_texto_usuario}".strip()
            or "No pude generar sugerencias en este momento. Intenta de nuevo.")


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
    _respuesta_retry = await _llamar_groq_con_excepciones(
        ia_engine, _prompt_retry, max_tokens=max_tokens_retry, temp=temp_retry
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
    ctx: Optional[UserContext] = None,
) -> str:
    """Respuesta conversacional corta vía LLM."""
    ctx = _asegurar_contexto(perfil, consumido, quemado, plan_macros, ctx)
    objetivo = ctx.objetivo_normalizado
    _es_vegano = any("vegano" in c.lower() for c in ctx.condiciones_medicas)
    _es_vegetariano = any("vegetariano" in c.lower() for c in ctx.condiciones_medicas)
    dieta = "Vegano" if _es_vegano else ("Vegetariano" if _es_vegetariano else "Normal")
    condiciones = ", ".join(ctx.condiciones_medicas) or "ninguna"
    _balance_chat = _calcular_balance_meta(consumido, meta, quemado, objetivo, mensaje)
    pct = _balance_chat["pct"]
    hist_txt = "\n".join(
        f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content'][:120]}"
        for m in historial[-4:]
    ) or "(inicio de conversación)"

    if _lesion_mencionada_sin_tipo(mensaje, historial):
        return (
            "¿Qué lesión tienes exactamente? ¿Es en la rodilla, espalda, hombro, "
            "codo u otra zona? Así te doy un consejo seguro y específico."
        )

    _tema_chat = _detectar_tema_chat(mensaje, historial)

    if _tema_chat == "general" and _es_peticion_ambigua(mensaje):
        return (
            "¿Sobre qué tema necesitas ayuda? Puedo ayudarte con nutrición, "
            "ejercicio o seguimiento de tu progreso."
        )

    if _tema_chat == "ejercicio":
        bloque_perfil = f"- Nombre: {perfil.first_name}"
        _bloque_balance_meta_txt = ""
    else:
        _bloque_balance_meta_txt = _balance_chat["bloque_balance"]
        _peso = getattr(perfil, "weight", None)
        _meta_prot = (plan_macros or {}).get("proteinas_g")
        bloque_perfil = (
            f"- Nombre: {perfil.first_name}\n"
            f"- {round(consumido)}/{round(meta)} kcal consumidas ({pct}%)  |  {round(quemado)} kcal quemadas hoy\n"
            f"- Dieta: {dieta}  |  Condiciones: {condiciones}  |  Objetivo: {objetivo}"
            + (f"  |  Peso: {_peso:.0f} kg" if _peso else "")
            + (f"  |  Meta proteína (plan): {round(_meta_prot)} g" if _meta_prot else "")
        )

    prompt = _PROMPT_CHAT.format(
        bloque_perfil=bloque_perfil,
        bloque_balance_meta=_bloque_balance_meta_txt,
        historial=hist_txt,
        mensaje=mensaje,
    )
    if _balance_chat.get("es_masa_muscular"):
        prompt += (
            "\n\n⚠️ INSTRUCCIÓN DE ALTA PRIORIDAD (OBJETIVO GANAR MASA): El usuario tiene el objetivo "
            "de ganar masa muscular / superávit / volumen. Si pregunta si debe/puede seguir comiendo o si ya pasó "
            "su meta calórica, debes felicitarlo o alentarlo a seguir comiendo. Explícale que estar "
            "en superávit (comer más de su meta) es lo correcto y necesario para hipertrofia/volumen/ganancia de músculo. "
            "PROHIBIDO decirle que evite comer, sugerirle comer menos, o recomendarle alimentos de muy bajo "
            "aporte calórico para no comer."
        )
    _m_lower_hora = mensaje.lower().strip()
    if any(k in _m_lower_hora for k in ("qué hora es", "que hora es", "qué hora son", "que hora son")):
        from app.core.utils import get_peru_now
        return f"Son las {get_peru_now().strftime('%H:%M')} (hora de Perú)."

    _m_norm_app = _normalizar_nombre(mensaje)
    _pregunta_uso_app = (
        ("como uso" in _m_norm_app or "como funciona" in _m_norm_app)
        and ("app" in _m_norm_app or "esto" in _m_norm_app or "aplicacion" in _m_norm_app)
    )
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

    import re as _re_puedo
    _RE_PUEDO_COMER = _re_puedo.compile(
        r'^(puedo|se\s+puede|puedo\s+yo|puede\s+uno)\s+'
        r'(comer|tomar|beber|ingerir|comerme|tomarme)\s+\S',
        _re_puedo.IGNORECASE,
    )
    if _RE_PUEDO_COMER.match(mensaje.strip()):
        _conds_raw   = getattr(perfil, "medical_conditions", None) or []
        _objetivo    = getattr(perfil, "goal", "mantener peso") or "mantener peso"
        _es_vegano   = any("vegano" in c.lower() for c in _conds_raw)
        _es_vegetariano = any("vegetariano" in c.lower() for c in _conds_raw)
        _tiene_diabetes = any("diabetes" in c.lower() for c in _conds_raw)
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
        try:
            _raw_perm = await _llamar_groq_con_excepciones(ia_engine, _prompt_perm, max_tokens=150, temp=0.5)
            _resultado_perm = _limpiar_markdown(_raw_perm)
        except Exception as e:
            logger.error("[LLM Error in chat permission check]: %s", e)
            return _obtener_fallback_chat_seguro(perfil, "nutricion")
        _nombre_escaped = _re_puedo.escape(perfil.first_name or "")
        if _nombre_escaped:
            _resultado_perm = _re_puedo.sub(
                rf'^{_nombre_escaped},\s*', '', _resultado_perm, flags=_re_puedo.IGNORECASE
            )
            if _resultado_perm:
                _resultado_perm = _resultado_perm[0].upper() + _resultado_perm[1:]
        return _resultado_perm

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
    _es_consulta_kcal = any(k in _m_lower for k in (
        "cuantas calorias tiene", "cuántas calorías tiene",
        "cuanto tiene de", "cuánto tiene de",
        "cuantas kcal tiene", "cuántas kcal tiene",
        "cuantos gramos tiene", "cuántos gramos tiene",
        "cuanta proteina tiene", "cuánta proteína tiene",
        "valor nutricional de", "macros de",
    ))

    if _es_consulta_kcal:
        alimento_query = mensaje
        try:
            raw_macros = await _llamar_groq_con_excepciones(
                ia_engine, _PROMPT_COMIDA.format(mensaje=alimento_query), max_tokens=400, temp=0.0, model="llama-3.3-70b-versatile"
            )
            d_macros = _parse_json(raw_macros)
        except Exception as e:
            logger.error("[LLM Error in chat kcal query]: %s", e)
            return _obtener_fallback_chat_seguro(perfil, "nutricion")
        if d_macros and d_macros.get("alimentos"):
            for item in d_macros["alimentos"]:
                p_i = float(item.get("prot_g", 0) or 0)
                c_i = float(item.get("carb_g", 0) or 0)
                g_i = float(item.get("grasa_g", 0) or 0)
                k_i = round(4*p_i + 4*c_i + 9*g_i, 1) or float(item.get("kcal", 0) or 0)
                if item.get("nombre") and k_i > 0:
                    cache_macros(item["nombre"], {
                        "nombre": item["nombre"], "kcal": k_i,
                        "prot_g": p_i, "carb_g": c_i, "grasa_g": g_i,
                        "porcion_g": float(item.get("porcion_g", 100) or 100),
                    })
            primer = d_macros["alimentos"][0]
            p_r = float(primer.get("prot_g", 0) or 0)
            c_r = float(primer.get("carb_g", 0) or 0)
            g_r = float(primer.get("grasa_g", 0) or 0)
            k_r = round(4*p_r + 4*c_r + 9*g_r, 1)
            grm = float(primer.get("porcion_g", 100) or 100)
            nombre_r = primer.get("nombre", "")
            _unidad_r = 'ml' if 'ml' in mensaje.lower() or 'jugo' in mensaje.lower() or 'leche' in mensaje.lower() else 'g'
            _partes_r = []
            for _val, _nom in ((p_r, "proteína"), (c_r, "carbohidratos"), (g_r, "grasa")):
                _partes_r.append(
                    f"casi nada de {_nom}" if _val < 0.5 else f"{_formato_num(_val)}g de {_nom}"
                )
            return (
                f"{nombre_r} ({grm:.0f}{_unidad_r}) tiene {k_r:.0f} kcal, "
                f"con {_partes_r[0]}, {_partes_r[1]} y {_partes_r[2]}."
            )

    try:
        _max_tok = 500 if (_es_receta or _es_tecnica) else 200
        raw = await _llamar_groq_con_excepciones(ia_engine, prompt, max_tokens=_max_tok, temp=0.7)
        resultado = _limpiar_markdown(raw)

        if _es_receta:
            import re as _re_fmt
            _idx_ing = resultado.lower().find("ingredientes:")
            if _idx_ing > 0:
                resultado = resultado[_idx_ing:]
            resultado = _re_fmt.sub(r'\s*(Ingredientes:)', r'\n\nIngredientes:', resultado)
            resultado = _re_fmt.sub(r'\s*(Preparaci[oó]n:)', r'\n\nPreparación:', resultado)
            resultado = _re_fmt.sub(r'\.?\s*(\d+\.)\s+', r'\n\1 ', resultado)
            resultado = resultado.strip()

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

        from app.services.rutina_service import (
            _LESIONES_SUSTITUCION, _detectar_lesiones, filtrar_lesiones_activas,
        )
        _condiciones_lista_chat = list(getattr(perfil, "medical_conditions", None) or [])
        _texto_hist_chat = " ".join(str(h.get("content", "")) for h in (historial or []))
        _lesiones_candidatas_chat = _detectar_lesiones(
            _condiciones_lista_chat + [mensaje or "", _texto_hist_chat]
        )
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
                resultado = await _llamar_groq_con_excepciones(ia_engine, _prompt_retry_chat, max_tokens=200, temp=0.3)
                resultado = _limpiar_markdown(resultado)

                if any(r in _normalizar_nombre(resultado) for r in _riesgosos_chat):
                    logger.warning("[Chat] Reintento también riesgoso — usando fallback seguro")
                    _nombres_unicos_chat = list(dict.fromkeys(_alternativas_chat))[:2]
                    _alt_txt = " o ".join(_nombres_unicos_chat) if _nombres_unicos_chat else "estiramientos suaves"
                    resultado = (
                        f"Por tu lesión, mejor evita esfuerzos que la sobrecarguen. "
                        f"Prueba con {_alt_txt} mientras te recuperas, y consulta con un profesional de salud."
                    )
    except asyncio.TimeoutError as e:
        logger.error("[LLM Timeout in respuesta_chat_llm]: %s", e)
        resultado = _obtener_fallback_chat_seguro(perfil, _tema_chat)
    except ConnectionError as e:
        logger.error("[LLM ConnectionError in respuesta_chat_llm]: %s", e)
        resultado = _obtener_fallback_chat_seguro(perfil, _tema_chat)
    except Exception as e:
        logger.exception("[General Error in respuesta_chat_llm]: %s", e)
        resultado = _obtener_fallback_chat_seguro(perfil, _tema_chat)

    if not _es_receta and not _es_tecnica:
        resultado = _recortar_respuesta_chat(resultado, mensaje)

    resultado = _naturalizar_macros(resultado)

    _es_masa_muscular = bool(_balance_chat.get("es_masa_muscular"))
    if _es_masa_muscular:
        _msg_low = (mensaje or "").lower()
        if any(k in _msg_low for k in ("comer", "como", "meta", "caloria", "caloría", "algo")):
            _res_low = (resultado or "").lower()
            if not any(w in _res_low for w in ("ganar", "músculo", "musculo", "masa", "superávit", "superavit")):
                _es_advertencia = any(
                    w in _res_low
                    for w in ("excediste", "pasaste", "excedido", "evita", "modera", "cuidado", "alerta")
                )
                _nota_masa = (
                    " Recuerda que tu objetivo es ganar masa muscular, así que estar en superávit calórico "
                    "es necesario y correcto."
                )
                if _es_advertencia:
                    resultado = (
                        "Como tu objetivo es ganar masa muscular, es correcto y necesario estar en superávit. "
                        "No te preocupes por haber pasado la meta calórica; sigue alimentándote bien para lograr tu ganancia de músculo."
                    )
                else:
                    resultado = resultado.rstrip() + _nota_masa

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


import time as _time
import unicodedata as _ud2
import re as _re2

_macro_cache: dict = {}
_CACHE_TTL = 7200


_SINONIMOS_ALIMENTOS = {
    "quinoa": "quinua", "kinua": "quinua", "kino":  "quinua",
    "quino":  "quinua", "kinoa": "quinua", "quinuoa":"quinua",
    "quinuo": "quinua", "quínoa":"quinua", "quínua": "quinua",
    "kinwa":  "quinua", "kinwua":"quinua",
    "palta": "aguacate", "aguacate": "palta",
    "choclo": "maiz",  "maiz": "choclo",
    "camote": "batata", "batata": "camote",
    "fulbito": "futbol", "pichanga": "futbol", "pichanguear": "futbol",
    "fulbo": "futbol", "cletear": "bicicleta", "cleteada": "bicicleta",
    "bici": "bicicleta", "fierros": "pesas",
}


def _normalizar_nombre(nombre: str) -> str:
    """Normaliza nombre: quita tildes, minúsculas, aplica sinónimos."""
    n = nombre.lower().strip()
    n = "".join(c for c in _ud2.normalize("NFD", n) if _ud2.category(c) != "Mn")
    n = _re2.sub(r"\s+", " ", n)
    def _sinonimo(t: str) -> str:
        if t in _SINONIMOS_ALIMENTOS:
            return _SINONIMOS_ALIMENTOS[t]
        if t.endswith("s") and len(t) > 4 and t[:-1] in _SINONIMOS_ALIMENTOS:
            return _SINONIMOS_ALIMENTOS[t[:-1]]
        return t
    tokens = n.split()
    tokens = [_sinonimo(t) for t in tokens]
    return " ".join(tokens)


_STOPWORDS_BASE_TEXTUAL = frozenset({
    "de", "la", "el", "los", "las", "con", "y", "en", "del", "al", "un", "una",
})


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


_CONECTORES_PLATO_COMPUESTO = frozenset({"con", "y", "de", "a", "la", "el", "los", "las"})


def _filtrar_componentes_de_plato_compuesto(alimentos: list[dict]) -> list[dict]:
    """Si el LLM extrae tanto un plato compuesto ("Arroz con lentejas") como
    sus propios ingredientes por separado ("arroz", "lentejas"), descarta los
    ingredientes — ya están contados dentro del plato completo. Sin esto, un
    plato reconocido como "X con Y" puede registrarse 3 veces (X, Y, y X con Y)
    e inflar las kcal del día. Determinista (sin costo de tokens): un ítem se
    descarta si TODAS sus palabras clave están contenidas en otro ítem que
    tiene más palabras clave que él."""
    if len(alimentos) < 2:
        return alimentos

    def _palabras_clave(nombre: str) -> set[str]:
        return {
            w for w in _normalizar_nombre(nombre or "").split()
            if w not in _CONECTORES_PLATO_COMPUESTO
        }

    claves = [_palabras_clave(a.get("nombre", "")) for a in alimentos]
    descartar_idx: set[int] = set()
    for i, palabras_i in enumerate(claves):
        if not palabras_i:
            continue
        for j, palabras_j in enumerate(claves):
            if i == j or j in descartar_idx:
                continue
            if palabras_i.issubset(palabras_j) and len(palabras_j) > len(palabras_i):
                descartar_idx.add(i)
                break

    if not descartar_idx:
        return alimentos
    return [a for idx, a in enumerate(alimentos) if idx not in descartar_idx]


_CONTENEDORES_GENERICOS = frozenset({
    "batido", "licuado", "jugo", "sopa", "ensalada", "smoothie",
    "preparado", "mezcla",
})


_RX_MULTI_MOMENTO = re.compile(
    r"\bdesayun|\balmuerz|\balmorc|\bcen[ée]|\bcena\b|\bmerienda|\bsnack",
    re.IGNORECASE,
)


def _es_mensaje_multi_comida(mensaje: str) -> bool:
    """True si el mensaje menciona 2+ momentos del día distintos (desayuno,
    almuerzo, cena, merienda) — un resumen del día completo en un solo
    mensaje, no una sola comida."""
    momentos = set(_RX_MULTI_MOMENTO.findall((mensaje or "").lower()))
    grupos = set()
    for m in momentos:
        if m.startswith("almuerz") or m.startswith("almorc"):
            grupos.add("almuerzo")
        elif m.startswith("desayun"):
            grupos.add("desayuno")
        elif m.startswith("cen"):
            grupos.add("cena")
        else:
            grupos.add("merienda")
    return len(grupos) >= 2


def _filtrar_contenedor_generico_con_ingredientes(alimentos: list[dict], mensaje: str = "") -> list[dict]:
    """Si el LLM extrae un contenedor genérico ("Batido de avena") Y además
    ≥2 ingredientes independientes en la misma extracción ("Leche",
    "Plátano", "Miel"), el contenedor es redundante — sus propias kcal se
    solapan con las de los ingredientes ya contados por separado. Se
    descarta el contenedor, se conservan los ingredientes reales.

    Solo se descarta si hay un solapamiento real de ingredientes (es decir,
    si alguno de los otros alimentos extraídos coincide con las palabras
    descriptivas del contenedor). Esto evita descartar bebidas independientes
    (ej: "jugo de piña") o acompañamientos (ej: "ensalada mixta") cuando se
    consumen junto a un plato de fondo (ej: "chuleta con arroz").
    """
    def _palabras_nombre(nombre: str) -> list[str]:
        return _normalizar_nombre(nombre or "").split()

    potenciales_contenedores = []
    for i, a in enumerate(alimentos):
        p = _palabras_nombre(a.get("nombre", ""))
        if p and p[0] in _CONTENEDORES_GENERICOS and len(p) > 1:
            potenciales_contenedores.append((i, a, p))

    if not potenciales_contenedores:
        return alimentos

    descartar_idx = set()
    for idx_c, a_c, palabras_c in potenciales_contenedores:
        keywords_c = {
            w for w in palabras_c[1:]
            if w not in _STOPWORDS_BASE_TEXTUAL and len(w) > 2
        }

        tiene_ingrediente_separado = False
        if keywords_c:
            for i, a in enumerate(alimentos):
                if i == idx_c or i in descartar_idx:
                    continue
                palabras_other = set(_palabras_nombre(a.get("nombre", "")))
                if keywords_c.intersection(palabras_other):
                    tiene_ingrediente_separado = True
                    break

        is_described_with_ingredients = False
        msg_lower = (mensaje or "").lower()
        container_word = palabras_c[0]
        if container_word in msg_lower:
            start_idx = msg_lower.find(container_word)
            sub_msg = msg_lower[start_idx + len(container_word):]
            sub_msg_norm = _normalizar_nombre(sub_msg)
            
            matching_other_foods = 0
            for i, a in enumerate(alimentos):
                if i == idx_c or i in descartar_idx:
                    continue
                other_words = {
                    w for w in _palabras_nombre(a.get("nombre", ""))
                    if w not in _STOPWORDS_BASE_TEXTUAL and len(w) > 2 and not w.isdigit()
                }
                if not other_words:
                    continue
                if any(w in sub_msg_norm for w in other_words):
                    matching_other_foods += 1
            
            if matching_other_foods >= 2:
                is_described_with_ingredients = True

        if tiene_ingrediente_separado or is_described_with_ingredients:
            descartar_idx.add(idx_c)

    if not descartar_idx:
        return alimentos

    if len(descartar_idx) == len(alimentos):
        return alimentos

    return [a for i, a in enumerate(alimentos) if i not in descartar_idx]


def _fusionar_alimentos_redundantes(alimentos: list[dict]) -> list[dict]:
    """Si dos ítems son el MISMO alimento mencionado dos veces — uno con
    nombre genérico y otro con un calificador agregado (ej. "Papitas" y
    "Papitas fritas") — el LLM lo duplicó pese a la Regla 16 del prompt
    (encontrado en pruebas reales: ocurre incluso con la instrucción
    explícita). Se conserva solo el nombre MÁS específico, evitando contar
    las calorías dos veces.

    No es una lista de palabras fija: la señal es ESTRUCTURAL — las palabras
    del nombre corto deben ser exactamente las palabras INICIALES del nombre
    largo (mismo orden, desde el principio). Por eso "Papitas" → "Papitas
    fritas" se fusiona, pero "Pollo" → "Caldo de pollo" NO — "pollo" no es la
    primera palabra de ese nombre, es un ingrediente DENTRO de otro plato
    distinto, no el mismo alimento repetido."""
    if len(alimentos) < 2:
        return alimentos

    def _palabras(nombre: str) -> list[str]:
        return _normalizar_nombre(nombre or "").split()

    listas_palabras = [_palabras(a.get("nombre", "")) for a in alimentos]
    descartar = set()
    for i, pi in enumerate(listas_palabras):
        if i in descartar or not pi:
            continue
        for j, pj in enumerate(listas_palabras):
            if i == j or j in descartar or len(pi) >= len(pj):
                continue
            if pj[: len(pi)] == pi:
                descartar.add(i)
                break
    return [a for idx, a in enumerate(alimentos) if idx not in descartar]


_PALABRAS_MOMENTO_DIA = frozenset({
    "desayuno", "almuerzo", "cena", "merienda", "snack", "comida",
    "entreno", "postentreno", "post",
})


def _es_solo_palabra_momento_dia(nombre: str) -> bool:
    """True si TODAS las palabras clave del nombre son de momento del día
    (ej. "Almuerzo", "Post entreno") — no si el momento aparece junto a un
    alimento real (ej. "Pollo al almuerzo" se queda, porque "pollo" no es
    palabra de momento)."""
    palabras = {
        w for w in _normalizar_nombre(nombre or "").split()
        if w not in _CONECTORES_PLATO_COMPUESTO
    }
    return bool(palabras) and palabras.issubset(_PALABRAS_MOMENTO_DIA)


_ITEMS_BAJOS_EN_CALORIAS_CONOCIDOS = frozenset({
    "agua", "cafe", "te", "infusion", "gaseosa light", "gaseosa zero",
    "agua con gas", "agua mineral", "agua de mesa", "soda",
})


def _macros_sospechosamente_nulos(item: dict) -> bool:
    """True si el ítem no tiene NINGÚN macro positivo (kcal/prot/carb/grasa
    todos en 0) Y no es una bebida real conocida por ser ~0 kcal. Es la señal
    más barata de que el propio LLM no tuvo datos reales para estimar este
    "alimento" — un alimento real casi nunca da exactamente 0 en los 4
    valores a la vez."""
    try:
        kcal  = float(item.get("kcal", 0) or 0)
        prot  = float(item.get("prot_g", 0) or 0)
        carb  = float(item.get("carb_g", 0) or 0)
        grasa = float(item.get("grasa_g", 0) or 0)
    except (TypeError, ValueError):
        return False
    if kcal > 0 or prot > 0 or carb > 0 or grasa > 0:
        return False
    nombre_norm = _normalizar_nombre(item.get("nombre", ""))
    return not any(b in nombre_norm for b in _ITEMS_BAJOS_EN_CALORIAS_CONOCIDOS)


def _existe_en_bd_alimentos(db, nombre: str) -> bool:
    """Confirma (no descarta) que el nombre corresponde a un alimento real
    consultando la tabla `alimentos` (~700+ registros INS/CENAN/OpenFoodFacts)
    y sus alias — la misma fuente que ya usa FoodSourceResolver._buscar_bd_local()
    para el flujo de platos dinámicos, reutilizada aquí en vez de un catálogo
    manual nuevo. Solo se usa como señal de APOYO (ver _alimento_es_alucinacion):
    muchos alimentos reales e internacionales ("sushi", marcas) no están en
    esta tabla acotada a Perú, así que su ausencia NUNCA basta sola para
    rechazar — solo refuerza el rechazo cuando los macros YA son sospechosos."""
    if db is None:
        return True
    nombre_norm = _normalizar_nombre(nombre or "")
    palabras = [
        p for p in nombre_norm.split()
        if len(p) >= 3 and p not in _CONECTORES_PLATO_COMPUESTO
    ]
    if not palabras:
        return True
    try:
        condiciones = " OR ".join(
            f"unaccent(lower(nombre_normalizado)) ~* "
            f"('(^| )' || unaccent(lower(:p{i})) || '( |$)')"
            for i in range(len(palabras))
        )
        params = {f"p{i}": p for i, p in enumerate(palabras)}
        fila = db.execute(
            text(f"SELECT 1 FROM alimentos WHERE {condiciones} LIMIT 1"),
            params,
        ).fetchone()
        if fila:
            return True
        from app.models.alimento_alias import AlimentoAlias
        alias = db.query(AlimentoAlias).filter(AlimentoAlias.alias.in_(palabras)).first()
        return alias is not None
    except Exception as exc:
        logger.warning("[Registro] Chequeo BD de alimento falló (no bloqueante): %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        return True


_RX_CORTE_POLLO = re.compile(
    r"\b(?:un\s+|una\s+)?(cuarto|octavo)\s+de\s+pollo\b"
    r"|\bmedio\s+pollo\b"
    r"|\bpollo\s+entero\b",
    re.IGNORECASE,
)
_PESO_CORTE_POLLO_G = {"octavo": 125, "cuarto": 250, "medio": 500, "entero": 1000}


def _aplicar_corte_pollo_brasa(alimentos: list, mensaje: str) -> None:
    """Mutación in-place: corrige porcion_g/cantidad/macros del ítem 'pollo'
    cuando el mensaje usa "cuarto/octavo/medio/entero de pollo" — fuerza el
    peso real del corte de menú (250g/125g/500g/1000g) en vez de la fracción
    matemática que el LLM a veces le asigna a una porción arbitraria."""
    m = _RX_CORTE_POLLO.search(mensaje or "")
    if not m:
        return
    texto_match = m.group(0).lower()
    palabra_corte = next((p for p in _PESO_CORTE_POLLO_G if p in texto_match), None)
    if not palabra_corte:
        return
    peso_objetivo = _PESO_CORTE_POLLO_G[palabra_corte]
    for item in alimentos:
        if "pollo" not in (item.get("nombre") or "").lower():
            continue
        try:
            porcion_actual = float(item.get("porcion_g", 100) or 100)
        except (TypeError, ValueError):
            continue
        if porcion_actual <= 0:
            continue
        factor = peso_objetivo / porcion_actual
        item["porcion_g"] = peso_objetivo
        item["cantidad"] = 1
        for campo in ("kcal", "prot_g", "carb_g", "grasa_g"):
            if item.get(campo) is not None:
                try:
                    item[campo] = round(float(item[campo]) * factor, 1)
                except (TypeError, ValueError):
                    pass
        item["_corte_pollo_aplicado"] = True
        item["_porcion_explicita_usuario"] = True
        logger.info(
            "[Registro] Corte de pollo a la brasa ('%s') -> porcion_g=%sg fijo",
            texto_match, peso_objetivo,
        )


def _alimento_es_alucinacion(item: dict, db) -> bool:
    """Rechazo conservador en 3 capas — las 3 deben coincidir para rechazar,
    así el riesgo de falso positivo (bloquear un alimento real) es mínimo:
    1. Sus propios macros son todos 0 (el LLM no tuvo datos reales).
    2. No es una bebida real conocida por ser ~0 kcal.
    3. Ninguna de sus palabras existe en la BD local de alimentos/alias.
    "umas" falla las 3. "café" (aunque salga con 0 kcal) se salva en la 2.
    "sushi" (aunque no esté en la BD local) se salva en la 1, porque el LLM
    sí le asigna macros reales."""
    if not _macros_sospechosamente_nulos(item):
        return False
    return not _existe_en_bd_alimentos(db, item.get("nombre", ""))


_SUFIJOS_VERBALES = ("aste", "iste", "ar", "er", "ir", "e", "i", "o", "a")


def _raiz_verbal(palabra: str) -> str:
    for suf in _SUFIJOS_VERBALES:
        if palabra.endswith(suf) and len(palabra) - len(suf) >= 3:
            return palabra[: -len(suf)]
    return palabra


def _extraccion_tiene_base_textual(
    nombre_extraido: str, mensaje_original: str, es_ejercicio: bool = False,
) -> bool:
    """Red de seguridad determinista (sin costo de tokens) contra alucinaciones
    del LLM: verifica que el nombre extraído tenga al menos una palabra
    significativa presente en el mensaje real del usuario. La Regla 0 del
    prompt ya le pide al LLM no inventar nada cuando no hay alimento/ejercicio
    real, pero esa instrucción no es garantía (validado en pruebas: el mismo
    bug reapareció con mensajes distintos) — esto la respalda con código.

    es_ejercicio=True activa un fallback adicional de raíz verbal (ver abajo)
    — solo tiene sentido para ejercicios, cuyo "nombre oficial" suele ser un
    infinitivo ("Correr") mientras el usuario lo dice conjugado ("corrí").
    Los alimentos son sustantivos, no verbos, así que ese fallback no se
    activa para la ruta de comida — evita falsos positivos entre nombres de
    alimentos que comparten una terminación común (ej. "Palta"/"Pasta")."""
    _nombre_norm = _normalizar_nombre(nombre_extraido or "")
    palabras = [
        p for p in _nombre_norm.split()
        if len(p) > 3 and p not in _STOPWORDS_BASE_TEXTUAL
    ]
    if not palabras:
        return True
    if all(p in _PALABRAS_NO_ALIMENTO_GENERICAS for p in palabras):
        return False
    msg_norm = _normalizar_nombre(mensaje_original or "")
    if any(p in msg_norm for p in palabras):
        return True
    import difflib as _difflib_base
    palabras_msg = [w for w in msg_norm.split() if len(w) > 3]
    for p in palabras:
        for w in palabras_msg:
            if _difflib_base.SequenceMatcher(None, p, w).ratio() >= 0.8:
                return True
    if es_ejercicio:
        for p in palabras:
            raiz_p = _raiz_verbal(p)
            if len(raiz_p) < 3:
                continue
            for w in palabras_msg:
                if _raiz_verbal(w) == raiz_p:
                    return True
    return False


_STOPWORDS_COMPLETITUD = frozenset({
    "hola", "buenas", "buenos", "comi", "come", "comer", "tome", "tomo",
    "bebi", "cene", "almorce", "desayune", "hoy", "ayer", "con", "para",
    "por", "mis", "tus", "sus", "una", "unos", "unas", "taza", "vaso",
    "plato", "porcion", "racion", "rebanada", "tajada", "lonja", "rodaja",
    "trozo", "pedazo", "cucharada", "cucharadita", "puñado", "copa",
    "botella", "lata", "jarra", "gramos", "litros", "cantidad", "tambien",
    "agrega", "agregalo", "agregale", "agregar", "anota", "anotalo",
    "registra", "registralo", "incluye", "incluyelo", "guarda", "guardalo",
    "ponme", "apunta", "apuntalo", "registro", "sumale", "suma",
    "acabo", "chapar", "eche", "echarme", "echarse",
    "platazo",
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
    def _singular(p: str) -> str:
        return p[:-1] if p.endswith("s") and len(p) > 4 else p
    _extraidas_singular = {_singular(w) for w in palabras_extraidas}
    _PUNTUACION_BORDE = ",.;:!?()¡¿\"'"
    palabras_msg = [
        p for p in (
            t.strip(_PUNTUACION_BORDE) for t in _normalizar_nombre(mensaje or "").split()
        )
        if len(p) >= 4
        and p not in _STOPWORDS_COMPLETITUD
        and p not in _STOPWORDS_BASE_TEXTUAL
        and p not in _PALABRAS_NO_ALIMENTO_GENERICAS
        and not any(c.isdigit() for c in p)
    ]
    def _ya_cubierta_como_substring(p: str) -> bool:
        return any(len(w) >= 4 and w in p for w in palabras_extraidas)
    import difflib as _difflib_completitud
    def _ya_cubierta_difusa(p: str) -> bool:
        return any(
            _difflib_completitud.SequenceMatcher(None, p, w).ratio() >= 0.7
            for w in palabras_extraidas if len(w) >= 4
        )
    _candidatas = [
        p for p in palabras_msg
        if _singular(p) not in _extraidas_singular
        and not _ya_cubierta_como_substring(p)
        and not _ya_cubierta_difusa(p)
    ]
    if _candidatas and _candidatas[-1] == "causa":
        _palabras_msg_full = [
            w.rstrip(_PUNTUACION_BORDE) for w in _normalizar_nombre(mensaje or "").split()
        ]
        if _palabras_msg_full and _palabras_msg_full[-1] == "causa":
            _idx_causa = len(_palabras_msg_full) - 1
            _palabra_previa = _palabras_msg_full[_idx_causa - 1] if _idx_causa > 0 else ""
            if _palabra_previa not in ("un", "una", "el", "la"):
                _candidatas = _candidatas[:-1]
    return _candidatas


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
    1. Coincidencia exacta limpia (quitando verbos de acción y artículos)
    2. Fuzzy matching limpio (umbral 0.82) sobre las formas limpias"""
    from difflib import SequenceMatcher
    
    _PREFIX_VERBS_STOPWORDS = {
        "comi", "tome", "cene", "almorce", "almorze", "desayune", "para", "el", "la", "un", "una", "de",
        "hoy", "ayer", "registra", "anota", "apunta", "con", "y", "mas"
    }
    
    clean_msg = " ".join([w for w in _normalizar_nombre(mensaje).split() if w not in _PREFIX_VERBS_STOPWORDS])
    if not clean_msg:
        return None
        
    ahora = _time.time()
    mejor_ratio = 0.0
    mejor_entry = None
    mejor_key = None

    for key, entry in list(_macro_cache.items()):
        if (ahora - entry.get("_ts", 0)) >= _CACHE_TTL:
            continue
        
        clean_key = " ".join([w for w in _normalizar_nombre(key).split() if w not in _PREFIX_VERBS_STOPWORDS])
        if not clean_key:
            continue
            
        if clean_msg == clean_key:
            logger.info("[MacroCache] Hit exacto limpio: '%s'", key)
            return {k: v for k, v in entry.items() if k != "_ts"}
            
        ratio = SequenceMatcher(None, clean_key, clean_msg).ratio()
        if ratio > mejor_ratio:
            mejor_ratio = ratio
            mejor_entry = entry
            mejor_key = key

    if mejor_ratio >= 0.82 and mejor_entry:
        logger.info("[MacroCache] Hit fuzzy limpio (ratio=%.2f, key='%s')", mejor_ratio, mejor_key)
        return {k: v for k, v in mejor_entry.items() if k != "_ts"}
    return None


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

_LACTEOS_LACTOSA = frozenset({"leche", "queso", "yogur", "mantequilla", "crema"})
_CALIFICADORES_SIN_LACTOSA = (
    "deslactosado", "deslactosada", "sin lactosa", "vegetal",
    "almendra", "soya", "soja", "avena", "coco",
)


def _detectar_conflicto_dieta(
    nombres: list, diet_type: str, condiciones_medicas: list | None = None,
) -> str | None:
    """Detecta si algún alimento registrado no corresponde a la dieta
    (vegano/vegetariano) o a una condición médica (intolerancia a la
    lactosa) del usuario — son chequeos independientes, pueden darse los dos
    a la vez o ninguno."""
    if not nombres:
        return None

    conflictos_dieta: list[str] = []
    dieta = (diet_type or "").lower().strip()
    prohibidos = _ANIMAL_VEGANO if "vegano" in dieta else (
        _ANIMAL_VEGETARIANO if "vegetariano" in dieta else set()
    )
    if prohibidos:
        for nombre in nombres:
            n_lower = nombre.lower()
            if any(p in n_lower for p in prohibidos):
                conflictos_dieta.append(nombre)

    conflictos_lactosa: list[str] = []
    condiciones_str = " ".join(condiciones_medicas or []).lower()
    if "lactosa" in condiciones_str:
        for nombre in nombres:
            n_lower = nombre.lower()
            if any(lact in n_lower for lact in _LACTEOS_LACTOSA) and not any(
                c in n_lower for c in _CALIFICADORES_SIN_LACTOSA
            ):
                conflictos_lactosa.append(nombre)

    if not conflictos_dieta and not conflictos_lactosa:
        return None

    avisos = []
    if conflictos_dieta:
        tipo = "vegana" if "vegano" in dieta else "vegetariana"
        items = ", ".join(list(dict.fromkeys(conflictos_dieta))[:2])
        avisos.append(f"⚠️ {items} no es parte de tu dieta {tipo}.")
    if conflictos_lactosa:
        items = ", ".join(list(dict.fromkeys(conflictos_lactosa))[:2])
        avisos.append(f"⚠️ {items} tiene lactosa — registraste intolerancia a la lactosa.")
    return " ".join(avisos) + " Registrado igual para mantener tu historial."


def _limpiar_markdown(texto: str) -> str:
    """Elimina markdown y patrones de intro/cierre del LLM."""
    import re as _re_md
    t = texto
    t = _re_md.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', t)
    t = _re_md.sub(r'^#{1,6}\s+.*$', '', t, flags=_re_md.MULTILINE)
    t = _re_md.sub(r'^-{3,}$', '', t, flags=_re_md.MULTILINE)

    lineas = t.split('\n')
    _INTRO_PATS = _re_md.compile(
        r'^[A-Za-záéíóúÁÉÍÓÚñÑ]+,\s*(me\s+alegra|qué\s+buena|es\s+un\s+placer|'
        r'para\s+hacer\s+\w+.*necesitas|para\s+preparar|para\s+realizar\s+una)',
        _re_md.IGNORECASE
    )
    if lineas and _INTRO_PATS.match(lineas[0].strip()):
        lineas = lineas[1:]
    t = '\n'.join(lineas)

    t = _re_md.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def _parse_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    try:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        cleaned = re.sub(r'//[^\n\r"]*', '', cleaned)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        def _evaluar_cadena_numerica(m: "re.Match") -> str:
            partes = re.split(r'\s*([*/])\s*', m.group(0))
            resultado = float(partes[0])
            for i in range(1, len(partes), 2):
                op, num = partes[i], float(partes[i + 1])
                resultado = resultado * num if op == '*' else resultado / num
            return str(round(resultado, 2))

        cleaned = re.sub(
            r'\d+(?:\.\d+)?(?:\s*[*/]\s*\d+(?:\.\d+)?)+',
            _evaluar_cadena_numerica,
            cleaned,
        )
        cleaned = re.sub(r'(\d+(?:\.\d+)?)\s*(?:g|ml|kcal|kg|mg|cc)(?=\s*[,}\]])', r'\1', cleaned)
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
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
    logger.warning("[_parse_json] JSON no extraíble — raw: %r", raw[:300])
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
