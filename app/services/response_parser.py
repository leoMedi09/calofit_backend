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
    intent_match = re.search(r'\[CALOFIT_INTENT:\s*(\w+)\]', texto_principal)
    if intent_match:
        resultado["intent"] = intent_match.group(1).upper()
        # Limpiar la etiqueta del texto para no mostrarla al usuario
        texto_principal = texto_principal.replace(intent_match.group(0), "").strip()

    # --- FASE 2: EXTRACCIÓN POR ETIQUETAS BLINDADAS (PROTOCOLO 3.5 - MULTI-SECCIÓN) ---
    if "[CALOFIT_HEADER]" in texto_principal:
        # Dividir el texto en potenciales bloques de sección (cada bloque empieza con intent o header)
        bloques_raw = re.split(r'(\[CALOFIT_INTENT:.*?\]|\[CALOFIT_HEADER\])', texto_principal)
        
        # Reconstruir bloques lógicos
        bloques_reales = []
        i = 1 
        while i < len(bloques_raw):
            etiqueta = bloques_raw[i]
            contenido = bloques_raw[i+1] if (i+1) < len(bloques_raw) else ""
            bloques_reales.append(etiqueta + contenido)
            i += 2

        for bloque in bloques_reales:
            header = re.search(r'\[CALOFIT_HEADER\](.*?)\[/CALOFIT_HEADER\]', bloque, re.DOTALL)
            stats = re.search(r'\[CALOFIT_STATS\](.*?)\[/CALOFIT_STATS\]', bloque, re.DOTALL)
            lista = re.search(r'\[CALOFIT_LIST\](.*?)\[/CALOFIT_LIST\]', bloque, re.DOTALL)
            action = re.search(r'\[CALOFIT_ACTION\](.*?)\[/CALOFIT_ACTION\]', bloque, re.DOTALL)
            footer = re.search(r'\[CALOFIT_FOOTER\](.*?)\[/CALOFIT_FOOTER\]', bloque, re.DOTALL)

            if header or lista:
                # Determinar tipo por contenido del bloque o por el intent global
                bloque_low = bloque.lower()
                tipo = "comida"
                if "workout" in bloque_low or "ejercicio" in bloque_low or "repeticiones" in bloque_low:
                    tipo = "ejercicio"
                elif "diet" in bloque_low or "recipe" in bloque_low or "comida" in bloque_low:
                    tipo = "comida"
                
                # Limpiar items
                items_raw = lista.group(1).strip().split('\n') if lista else []
                items = [re.sub(r'^[-\*•\d\.\s]+', '', i).strip() for i in items_raw if i.strip()]

                # Limpiar pasos
                pasos_raw = action.group(1).strip().split('\n') if action else []
                pasos = [re.sub(r'^\d+[\.\)\s]+', '', p).strip() for p in pasos_raw if p.strip()]

                seccion = {
                    "tipo": tipo,
                    "nombre": header.group(1).strip() if header else "Sugerencia CaloFit",
                    "justificacion": "", 
                    "ingredientes": items if tipo == "comida" else [],
                    "ejercicios": items if tipo == "ejercicio" else [],
                    "preparacion": pasos if tipo == "comida" else [],
                    "tecnica": pasos if tipo == "ejercicio" else [],
                    "macros": stats.group(1).strip() if stats else None,
                    "gasto_calorico_estimado": stats.group(1).strip() if stats and tipo == "ejercicio" else None,
                    "nota": footer.group(1).strip() if footer else ""
                }
                
                # De-duplicación inteligente (evitar misma sección repetida)
                if not any(s["nombre"] == seccion["nombre"] and s["tipo"] == seccion["tipo"] for s in resultado["secciones"]):
                    resultado["secciones"].append(seccion)

        # El texto conversacional es todo lo que está FUERA de las etiquetas blindadas
        texto_limpio = re.sub(r'\[CALOFIT_.*?\](.*?)\[/CALOFIT_.*?\]', '', texto_principal, flags=re.DOTALL)
        texto_limpio = re.sub(r'\[CALOFIT_INTENT:.*?\]', '', texto_limpio)
        resultado["texto_conversacional"] = texto_limpio.strip()
        return resultado

    # --- FASE 3: FALLBACK A PARSER ELÁSTICO (Formato Antiguo) ---
    # (Mantener compatibilidad con respuestas que no sigan el nuevo protocolo)
    t = texto_principal.replace('***', '').replace('**', '').strip()
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

    resultado["texto_conversacional"] = " ".join(intro_lines).strip()
    return resultado
