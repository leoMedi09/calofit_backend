import re
from typing import Dict, List, Optional

def parsear_respuesta_para_frontend(texto_principal: str, mensaje_usuario: str = None) -> Dict:
    """
    Motor de parsing ultra-robusto (v11.0 - Protocolo Fallo Cero).
    Prioriza etiquetas blindadas [CALOFIT_XXX] y usa fallback elástico si no existen.
    """
    resultado = {
        "intent": "CHAT",
        "texto_conversacional": "",
        "secciones": [],
        "advertencia_nutricional": None
    }

    if not texto_principal: return resultado

    # --- FASE 1: DETECCIÓN DE INTENCIÓN (VÍA ETIQUETA) ---
    intent_match = re.search(r'\[CALOFIT_INTENT:\s*(\w+)\]', texto_principal, re.IGNORECASE)
    if intent_match:
        resultado["intent"] = intent_match.group(1).upper()
        # Limpiar la etiqueta del texto para no mostrarla al usuario
        texto_principal = texto_principal.replace(intent_match.group(0), "").strip()

    # --- FASE 2: EXTRACCIÓN POR ETIQUETAS BLINDADAS (PROTOCOLO 3.5 - MULTI-SECCIÓN) ---
    # Detectar headers sin importar mayúsculas/minúsculas
    if re.search(r'\[CALOFIT_HEADER\]', texto_principal, re.IGNORECASE):
        # Dividir el texto en potenciales bloques de sección (cada bloque empieza con intent o header)
        bloques_raw = re.split(r'(\[CALOFIT_INTENT:.*?\]|\[CALOFIT_HEADER\])', texto_principal, flags=re.IGNORECASE)
        
        # Reconstruir bloques lógicos
        bloques_reales = []
        i = 1 
        while i < len(bloques_raw):
            etiqueta = bloques_raw[i]
            contenido = bloques_raw[i+1] if (i+1) < len(bloques_raw) else ""
            bloques_reales.append(etiqueta + contenido)
            i += 2

        for bloque in bloques_reales:
            header = re.search(r'\[CALOFIT_HEADER\](.*?)\[/CALOFIT_HEADER\]', bloque, re.DOTALL | re.IGNORECASE)
            stats = re.search(r'\[CALOFIT_STATS\](.*?)\[/CALOFIT_STATS\]', bloque, re.DOTALL | re.IGNORECASE)
            lista = re.search(r'\[CALOFIT_LIST\](.*?)\[/CALOFIT_LIST\]', bloque, re.DOTALL | re.IGNORECASE)
            action = re.search(r'\[CALOFIT_ACTION\](.*?)\[/CALOFIT_ACTION\]', bloque, re.DOTALL | re.IGNORECASE)
            footer = re.search(r'\[CALOFIT_FOOTER\](.*?)\[/CALOFIT_FOOTER\]', bloque, re.DOTALL | re.IGNORECASE)

            if header or lista:
                # 🎯 DETECCIÓN DE TIPO MEJORADA (v12.1 - INTENT POR BLOQUE + DEBUG MACROS)
                bloque_low = bloque.lower()
                tipo = "comida"  # Default

                # 1. Prioridad: Intent dentro del propio bloque (multi-opcion tiene un intent por bloque)
                bloque_intent_match = re.search(r'\[CALOFIT_INTENT:\s*(\w+)\]', bloque, re.IGNORECASE)
                bloque_intent = bloque_intent_match.group(1).upper() if bloque_intent_match else resultado["intent"]

                if bloque_intent in ["ITEM_WORKOUT", "WORKOUT", "EXERCISE"]:
                    tipo = "ejercicio"
                elif bloque_intent in ["ITEM_RECIPE", "RECIPE", "FOOD", "MEAL"]:
                    tipo = "comida"
                else:
                    # 2. Fallback: Keywords en el bloque
                    kw_ejercicio = ["series", "repeticiones", "reps", "sets", "plancha", "sentadillas",
                                    "flexiones", "abdominales", "cardio", "calentamiento", "rutina",
                                    "workout", "ejercicio", "burpees", "trote"]
                    kw_comida = ["ingredientes", "preparación", "preparacion", "cocina", "gramos", "cucharada",
                                 "recipe", "comida", "plato", "receta"]

                    ejercicio_score = sum(1 for kw in kw_ejercicio if kw in bloque_low)
                    comida_score = sum(1 for kw in kw_comida if kw in bloque_low)

                    if ejercicio_score > comida_score:
                        tipo = "ejercicio"
                    elif comida_score > 0:
                        tipo = "comida"
                
                # Limpiar items - Regex mejorado para NO borrar cantidades (ej: "100g")
                if lista:
                    items_raw = lista.group(1).strip().split('\n')
                else:
                    # Fallback: Buscar líneas con bullets en el bloque si no hay etiqueta
                    items_raw = re.findall(r'^\s*[-\*•]\s+(.+)$', bloque, re.MULTILINE)

                # Solo borra bullets (*, -, •) o números seguidos de punto/paréntesis (1., 1))
                items = [re.sub(r'^(\s*[-\*•]\s?|\s*\d+[\.\)]\s?)', '', i).strip() for i in items_raw if i.strip()]
                # Filtrar líneas que solo digan "ingredientes:" o similar
                items = [i for i in items if not re.match(r'^(ingredientes|ejercicios|lista)[:\.]?$', i, re.IGNORECASE)]

                # Limpiar pasos
                if action:
                    pasos_raw = action.group(1).strip().split('\n')
                else:
                    # Fallback: Buscar líneas numeradas (1., 2.)
                    pasos_raw = re.findall(r'^\s*\d+[\.\)]\s+(.+)$', bloque, re.MULTILINE)
                    
                # Igual para pasos, manteniendo el texto limpio
                pasos = [re.sub(r'^(\s*[-\*•]\s?|\s*\d+[\.\)]\s?)', '', p).strip() for p in pasos_raw if p.strip()]
                # Filtrar líneas que solo digan "preparación:" o similar
                pasos = [p for p in pasos if not re.match(r'^(preparaci[oó]n|instrucciones|pasos|tecnica)[:\.]?$', p, re.IGNORECASE)]

                msg_stats = stats.group(1).strip() if stats else ""
                # Limpiar texto extra como "(Ajustado a tu meta)" o emojis del stats
                msg_stats_clean = re.sub(r'\(Ajustado.*?\)', '', msg_stats).strip()
                if msg_stats_clean.endswith('\U0001f373'):  # emoji sarten
                    msg_stats_clean = msg_stats_clean[:-1].strip()

                nombre_raw = header.group(1).strip() if header else "Sugerencia CaloFit"
                nombre_clean = re.sub(r'^(Opción|Option|Plato|Rutina)\s*\d+[:\.]?\s*', '', nombre_raw, flags=re.IGNORECASE).strip()

                print(f"📊 [Parser] Sección '{nombre_clean}' | tipo={tipo} | macros='{msg_stats_clean}'")

                seccion = {
                    "tipo": tipo,
                    "nombre": nombre_clean,
                    "justificacion": "",
                    "ingredientes": items if tipo == "comida" else [],
                    "ejercicios": items if tipo == "ejercicio" else [],
                    "preparacion": pasos if tipo == "comida" else [],
                    "tecnica": pasos if tipo == "ejercicio" else [],
                    "instrucciones": pasos if tipo == "ejercicio" else [],
                    "macros": msg_stats_clean if tipo == "comida" else "",
                    "gasto_calorico_estimado": msg_stats_clean if tipo == "ejercicio" else "",
                    "nota": footer.group(1).strip() if footer else ""
                }
                
        # De-duplicación inteligente (evitar misma sección repetida)
                if not any(s["nombre"] == seccion["nombre"] and s["tipo"] == seccion["tipo"] for s in resultado["secciones"]):
                    resultado["secciones"].append(seccion)

        # --- FASE 3: RECONSTRUCCIÓN DE COMPVERSACIÓN (SIN RESIDUOS DE RECETAS) ---
        # Estrategia: "Split and Select". Solo mantenemos texto que NO pertenece a un bloque HEADER.
        # bloques_raw[0] es el texto antes del primer tag.
        # bloques_raw[1], [3]... son los tags.
        # bloques_raw[2], [4]... son los contenidos.
        
        texto_limpio_parts = [bloques_raw[0]]
        k = 1
        while k < len(bloques_raw):
            tag = bloques_raw[k]
            content = bloques_raw[k+1] if (k+1) < len(bloques_raw) else ""
            
            # Si el tag es INTENT, el contenido es conversación.
            # Si el tag es HEADER, el contenido es una Card y NO debe ir al chat.
            if "INTENT" in tag.upper():
                texto_limpio_parts.append(content)
            
            k += 2
            
        texto_limpio = "".join(texto_limpio_parts)
        
        # --- FASE 4: FORMATEO VISUAL (LISTAS BONITAS) ---
        # Detectar listas pegadas (ej: "incluyen: * Tofu") y forzar saltos de línea
        # Regex: Espacio/Punto + [bullet/numero] + espacio -> Newline + bullet
        texto_limpio = re.sub(r'([:;.])\s*([-\*•]|\d+\.)\s+', r'\1\n\2 ', texto_limpio) # Case: "text: * Item" -> "text:\n* Item"
        texto_limpio = re.sub(r'\s+([-\*•])\s+', r'\n\1 ', texto_limpio) # Case: "Item 1 * Item 2" -> "Item 1\n* Item 2"
        # Evitar romper numeros en medio de texto, solo si parece una lista (num + punto)
        texto_limpio = re.sub(r'\s+(\d+\.)\s+', r'\n\1 ', texto_limpio) 

        # Eliminar etiquetas residuales o cabeceras alucinadas (ej: "**CHAT**", "**ITEM_RECIPE**")
        texto_limpio = re.sub(r'^\s*\*\*?(CHAT|ITEM_RECIPE|ITEM_WORKOUT|ASISTENTE|RESPUESTA|INTENT|PLAN_DIET|PLAN_WORKOUT)\*\*?\s*', '', texto_limpio, flags=re.IGNORECASE)
        
        # 🧹 LIMPIEZA FINAL: Eliminar nombres de platos/ejercicios que quedaron en texto plano
        # Si detectamos los nombres de las secciones en el texto conversacional, los borramos
        for seccion in resultado["secciones"]:
            nombre_plato = seccion["nombre"]
            # Eliminar el nombre si aparece literal en el texto (suele estar en mayúsculas o con formato)
            # Casos: "TACACHO DE HUEVOS", "Tacacho de Huevos", etc.
            texto_limpio = re.sub(r'\b' + re.escape(nombre_plato) + r'\b', '', texto_limpio, flags=re.IGNORECASE)
            # También eliminar versiones en mayúsculas completas
            texto_limpio = re.sub(r'\b' + re.escape(nombre_plato.upper()) + r'\b', '', texto_limpio)
        
        # Limpiar espacios múltiples y saltos de línea excesivos generados por las eliminaciones
        texto_limpio = re.sub(r'\n\s*\n\s*\n+', '\n\n', texto_limpio)
        texto_limpio = re.sub(r'  +', ' ', texto_limpio)
        
        resultado["texto_conversacional"] = texto_limpio.strip()
        return resultado

    # --- FASE 3: FALLBACK A PARSER ELÁSTICO (Formato Antiguo) ---
    # (Mantener compatibilidad con respuestas que no sigan el nuevo protocolo)
    t = texto_principal.replace('***', '').strip()
    lineas = [l.strip() for l in t.split('\n') if l.strip()]
    
    start_markers = ["plato:", "rutina:", "receta:", "nombre:", "ejercicio:", "comida:"]
    field_markers = ["justificacion:", "ingredientes:", "ejercicios:", "preparacion:", "tecnica:", "pasos:", "aporte:", "stats:", "nota:"]
    
    current_section = None
    last_key = None
    intro_lines = []

    for l in lineas:
        l_low = l.lower()
        is_start = any(l_low.startswith(m) for m in start_markers)
        
        if is_start:
            if current_section: resultado["secciones"].append(current_section)
            tipo = "ejercicio" if "rutina" in l_low or "ejercicio" in l_low else "comida"
            nombre = l.split(':', 1)[1].strip() if ':' in l else l
            current_section = {"tipo": tipo, "nombre": nombre, "justificacion": "", "ingredientes": [], "preparacion": [], "macros": "", "nota": ""}
            last_key = "nombre"
            continue

        if not current_section:
            intro_lines.append(l)
            continue

        # Procesar campos dentro de sección
        if "justificacion" in l_low: last_key = "justificacion"
        elif "ingredientes" in l_low or "ejercicios" in l_low: last_key = "ingredientes"
        elif "preparacion" in l_low or "tecnica" in l_low or "pasos" in l_low: last_key = "preparacion"
        elif "aporte" in l_low or "macros" in l_low or "calorias" in l_low: last_key = "macros"
        elif "nota" in l_low or "recuerda" in l_low: last_key = "nota"
        else:
            if last_key in ["ingredientes", "preparacion"]:
                item = re.sub(r'^([-\*\+\#•]|\d+[\.\)\s])\s*', '', l).strip()
                if item: current_section[last_key].append(item)
            elif last_key:
                current_section[last_key] = (current_section[last_key] + " " + l).strip()

    if current_section:
        if current_section["tipo"] == "ejercicio":
            current_section["ejercicios"] = current_section.pop("ingredientes")
            current_section["tecnica"] = current_section.pop("preparacion")
            current_section["gasto_calorico_estimado"] = current_section.pop("macros")
        resultado["secciones"].append(current_section)

    texto_limpio = "\n".join(intro_lines).strip()
    
    # --- FASE 4: FORMATEO VISUAL TAMBIÉN PARA FALLBACK ---
    texto_limpio = re.sub(r'([:;.])\s*([-\*•]|\d+\.)\s+', r'\1\n\2 ', texto_limpio)
    texto_limpio = re.sub(r'\s+([-\*•])\s+', r'\n\1 ', texto_limpio)
    texto_limpio = re.sub(r'\s+(\d+\.)\s+', r'\n\1 ', texto_limpio) 
    
    resultado["texto_conversacional"] = texto_limpio
    return resultado
