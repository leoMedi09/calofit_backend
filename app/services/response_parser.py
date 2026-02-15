import re
from typing import Dict, List, Optional

def parsear_respuesta_para_frontend(texto_principal: str) -> Dict:
    """
    Motor de parsing ultra-robusto (v5.0) para separar IA-Chat de Ficha Técnica.
    Detecta secciones incluso si falta la etiqueta 'Plato:'.
    """
    resultado = {
        "texto_conversacional": "",
        "secciones": [],
        "advertencia_nutricional": None
    }

    if not texto_principal: return resultado

    # 1. Limpieza profunda y estandarización (v6.4 - Anti-Markdown)
    t = texto_principal.replace('**', '').replace('__', '').replace('###', '').replace('####', '').replace('#', '').strip()
    lineas = [l.strip() for l in t.split('\n') if l.strip()]
    
    # 2. Marcadores elásticos
    start_markers = ["plato:", "rutina:", "receta:", "nombre:", "plato recomendado:", "receta recomendada:"]
    field_markers = [
        "justificacion", "justificación", "calorias y macros", "calorías y macros", 
        "gasto calórico", "gasto calorico", "ingredientes", "ejercicios", 
        "preparacion", "preparación", "tecnica", "técnica", "pasos", 
        "aporte nutricional", "valor nutricional", "recuerda", "nota",
        "calculo de calorías", "cálculo de calorías", "calculo", "cálculo"
    ]
    
    intro_lines = []
    current_section = None
    last_key = None
    
    for l in lineas:
        l_low = l.lower()
        
        # --- DETECCIÓN DE INICIO DE SECCIÓN (v5.7) ---
        is_start = any(l_low.startswith(m) for m in start_markers)
        
        if not current_section:
            # Marcadores de que NO es un título (si parece ingrediente o paso de lista)
            no_es_titulo = l.startswith(('-', '*', '•')) or re.match(r'^\d+\.?\s+', l)
            
            # Si una línea es corta (<55 chars) y no parece conversación ni lista, podría ser el nombre del plato
            palabras_ruido_titulo = ["hola", "disculpo", "error", "aquí", "presento", "siento", "amigo", "amiga", "lamento"]
            if not is_start and not no_es_titulo and len(l) < 55 and not l.endswith('.') and ':' not in l and not any(word in l_low for word in palabras_ruido_titulo):
                current_section = {
                    "tipo": "comida",
                    "nombre": l,
                    "justificacion": "",
                    "ingredientes": [],
                    "preparacion": [],
                    "macros": "",
                    "nota": ""
                }
                last_key = "nombre"
                continue
            
            # Si se detectó una etiqueta crítica (Ingredientes, Preparación, Macros, etc.)
            es_tag_critica = any(l_low.startswith(f + ":") for f in field_markers)
            if is_start or es_tag_critica:
                current_section = {
                    "tipo": "comida",
                    "nombre": "Sugerencia CaloFit",
                    "justificacion": "",
                    "ingredientes": [],
                    "preparacion": [],
                    "macros": "",
                    "nota": ""
                }
                if "rutina" in l_low or "ejercicio" in l_low or "gasto" in l_low:
                    current_section["tipo"] = "ejercicio"
                # NO HACEMOS CONTINUE: Dejamos que el bloque de abajo procese la línea para extraer el valor (ej: Macros)
            else:
                intro_lines.append(l)
                continue

        processed = False
        
        # --- DETECCIÓN DE CAMPOS FLEXIBLE (v7.5 - Más estricta) ---
        # Solo detectamos si la palabra clave está al inicio de la línea o es el encabezado
        if any(l_low.startswith(m) for m in start_markers):
            current_section["nombre"] = l.split(':', 1)[1].strip() if ':' in l else l
            last_key = "nombre"
            processed = True
        elif l_low.startswith("justificacion") or l_low.startswith("justificación"):
            current_section["justificacion"] = l.split(':', 1)[1].strip() if ':' in l else l
            last_key = "justificacion"
            processed = True
        elif any(l_low.startswith(fm) for fm in ["calorías y macros", "calorias y macros", "gasto", "aporte nutricional", "valor nutricional"]):
            current_section["macros"] = l.split(':', 1)[1].strip() if ':' in l else l
            last_key = "macros"
            processed = True
        elif l_low.startswith("recuerda") or l_low.startswith("nota"):
            current_section["nota"] = l.split(':', 1)[1].strip() if ':' in l else l
            last_key = "nota"
            processed = True
        elif l_low.startswith("ingredientes") or l_low.startswith("ejercicios"):
            last_key = "ingredientes"
            processed = True
        elif l_low.startswith("advertencia nutricional"):
            resultado["advertencia_nutricional"] = l.split(':', 1)[1].strip() if ':' in l else l
            processed = True
        elif any(l_low.startswith(fm) for fm in ["preparacion", "preparación", "tecnica", "técnica", "pasos"]):
            last_key = "preparacion"
            processed = True

        if processed: continue

        if last_key:
            if last_key in ["ingredientes", "preparacion"]:
                # Ignorar encabezados markdown ### en las listas
                if l.strip().startswith("#"): continue

                # Si una línea de lista contiene "Recuerda" o "Nota", la movemos a la nota automáticamente
                if "recuerda" in l_low or "nota" in l_low:
                    current_section["nota"] = re.sub(r'^[-\*•\d\.\s\)]+(recuerda|nota):\s*', '', l, flags=re.IGNORECASE).strip()
                else:
                    item = re.sub(r'^([-\*•]|\d+[\.\)\s])\s*', '', l).strip()
                    if item: current_section[last_key].append(item)
            elif last_key == "nombre":
                pass 
            elif last_key == "macros":
                current_section["macros"] = l.strip()
            else:
                current_section[last_key] = (current_section[last_key] + " " + l).strip()

    # Consolidación final y Filtro Anti-Fantasma (v8.0)
    if current_section:
        # Limpieza final de la nota
        if not current_section.get("nota"):
            for i, p in enumerate(current_section["preparacion"]):
                if "recuerda" in p.lower() or "nota" in p.lower():
                    current_section["nota"] = p
                    current_section["preparacion"].pop(i)
                    break
        
        # VALIDACIÓN: Una ficha técnica real DEBE tener Ingredientes (o ejercicios) O Preparación.
        # Si no tiene listas, es solo texto conversacional que pareció un título.
        es_ficha_valida = len(current_section["ingredientes"]) > 0 or len(current_section["preparacion"]) > 0
        
        if es_ficha_valida:
            resultado["secciones"].append(current_section)
        else:
            # Reintegrar al chat ("Disolver" la sección)
            texto_recuperado = []
            if current_section["nombre"] != "Sugerencia CaloFit": texto_recuperado.append(current_section["nombre"])
            if current_section["justificacion"]: texto_recuperado.append(current_section["justificacion"])
            if current_section["macros"]: texto_recuperado.append(current_section["macros"])
            if current_section["nota"]: texto_recuperado.append(current_section["nota"])
            
            bloque_texto = "\n".join(texto_recuperado)
            intro_lines.append(bloque_texto)
    
    resultado["texto_conversacional"] = " ".join(intro_lines).strip()
    if not resultado["texto_conversacional"] and resultado["secciones"]:
        resultado["texto_conversacional"] = "¡Aquí tienes tu recomendación!"

    return resultado
