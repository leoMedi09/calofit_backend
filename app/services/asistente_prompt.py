"""
Construcción y post-proceso del prompt del asistente.

Funciones exportadas:
  construir_prompt_cliente()       — prompt de contexto para Llama-3
  enriquecer_prompt_con_bd()       — inyecta ejercicios/platos reales desde Postgres
  clasificar_intencion_respuesta() — decide tarjeta vs texto plano
  limpiar_tags_calofit()           — elimina residuos de tags CALOFIT
  detectar_intencion_principal()   — tema visual para Flutter (RECIPE, POWER, etc.)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.cache import get_user_recent_meals
from app.models.preferencias import PreferenciaAlimento
from app.services.asistente_modos import bloque_prompt_modo_funcion
from app.services.ml_service import ml_perfil, ml_recomendador
from app.services.response_parser import sanear_texto_conversacional_recipe


# ── Prompt builder ────────────────────────────────────────────────────────────

def construir_prompt_cliente(
    perfil,
    edad: int,
    plan_hoy_data: dict,
    calorias_meta: float,
    consumo_real: float,
    quemadas_real: float,
    adherencia_pct: float,
    progreso_pct: float,
    mensaje_fuzzy: str,
    es_saludo: bool = False,
    db: Optional[Session] = None,
    modo_funcion: Optional[str] = None,
    mensaje_usuario: str = "",
) -> str:
    """Construye el prompt del Coach Personal del Cliente."""
    from datetime import datetime

    # ── Condiciones médicas segmentadas ──
    alergias, preferencias_dieta, condiciones = [], [], []
    for cond in (perfil.medical_conditions or []):
        cond_l = cond.lower()
        if any(p in cond_l for p in ["alérgico", "alergia", "intolerancia"]):
            alergias.append(cond)
        elif any(p in cond_l for p in ["vegano", "vegetariano", "pescetariano"]):
            preferencias_dieta.append(cond)
        else:
            condiciones.append(cond)

    texto_alergias    = ", ".join(alergias)          or "Ninguna"
    texto_dieta       = ", ".join(preferencias_dieta) or "Omnívoro"
    texto_condiciones = ", ".join(condiciones)        or "Ninguna"
    restantes         = max(0.0, calorias_meta - consumo_real + quemadas_real)  # igual que la UI de la app
    foco              = perfil.ai_strategic_focus or "Bienestar General"
    alimentos_pro     = perfil.forbidden_foods or []
    alimentos_rec_raw = perfil.recommended_foods or []

    # Filtrar alimentos_rec: eliminar ítems incompatibles con la dieta del usuario.
    # Si el nutri puso "pescado" en recomendados pero el cliente es Vegano/Vegetariano,
    # ese ítem NO debe llegar al prompt (causaría contradicción con Regla 10).
    _conds_dieta_check = [c.lower() for c in (perfil.medical_conditions or [])]
    _tokens_dieta_prohib: set[str] = set()
    if any("vegano" in c for c in _conds_dieta_check):
        _tokens_dieta_prohib.update({
            "pollo", "pechuga", "gallina", "pato", "pavo", "cabrito",
            "cerdo", "chancho", "res", "carne", "bistec", "lomo",
            "pescado", "salmon", "atun", "trucha", "caballa", "mariscos",
            "leche", "queso", "yogur", "mantequilla", "huevo",
        })
    elif any("vegetariano" in c for c in _conds_dieta_check):
        _tokens_dieta_prohib.update({
            "pollo", "pechuga", "gallina", "pato", "pavo", "cabrito",
            "cerdo", "chancho", "res", "carne", "bistec", "lomo",
            "pescado", "salmon", "atun", "trucha", "caballa", "mariscos",
        })
    alimentos_rec = [
        a for a in alimentos_rec_raw
        if not any(t in a.lower() for t in _tokens_dieta_prohib)
    ]

    # ── Hora peruana y sugerencia por momento ──
    try:
        from app.core.utils import get_peru_now
        now_pe     = get_peru_now()
        hora_actual = now_pe.hour
        hora_txt    = now_pe.strftime("%H:%M")
    except Exception:
        from datetime import datetime as _dt
        now_pe      = _dt.now()
        hora_actual = now_pe.hour
        hora_txt    = now_pe.strftime("%H:%M")

    _MOMENTOS = {
        range(5, 10):  ("DESAYUNO",     "Sugiere desayuno peruano ligero (pan, tamalito, quinua, fruta). Evita almuerzos/cenas pesadas."),
        range(10, 12): ("SNACK MAÑANA", "Snack liviano: fruta, yogurt, canchita, pan pequeño con proteína."),
        range(12, 15): ("ALMUERZO",     "Momento más contundente: seco, guiso, arroz, cebiche, tallarines en porción razonable."),
        range(15, 18): ("SNACK TARDE",  "Merienda: fruta, yogurt, sándwich pequeño, choclo con queso controlado."),
        range(18, 21): ("CENA",         "Cena más ligera que el almuerzo: ensaladas, pescado plancha, cremas livianas, causa moderada."),
    }
    momento_comida, sugerencia_horaria = "NOCTURNO", (
        "Horario nocturno. Prefiere snacks pequeños y digestivos. PROHIBIDO almuerzos copiosos de noche."
    )
    for rng, (m, s) in _MOMENTOS.items():
        if hora_actual in rng:
            momento_comida, sugerencia_horaria = m, s
            break

    bloque_hora = (
        f"\nMOMENTO DEL DÍA (hora Perú): {momento_comida} — reloj local {hora_txt}. "
        f"OBLIGATORIO para MODO RECIPE: las 2–3 opciones deben encajar con este momento. "
        f"{sugerencia_horaria}"
    )

    # ── Rango kcal orientativo ──
    r = max(0.0, restantes)
    bloque_rango_kcal = ""
    if r > 15:
        _rangos = {
            "NOCTURNO":      (min(100, r * 0.15), min(320, r * 0.45)),
            "CENA":          (min(200, r * 0.2),  min(520, r * 0.55)),
            "ALMUERZO":      (min(380, r * 0.25), min(900, r * 0.75)),
            "DESAYUNO":      (min(220, r * 0.2),  min(480, r * 0.42)),
            "SNACK MAÑANA":  (min(80,  r * 0.1),  min(260, r * 0.32)),
            "SNACK TARDE":   (min(80,  r * 0.1),  min(260, r * 0.32)),
        }
        lo, hi = _rangos.get(momento_comida, (150, min(500, r * 0.5)))
        lo_i, hi_i = max(50, int(lo)), max(int(lo) + 30, int(hi))
        bloque_rango_kcal = (
            f"\nRANGO KCAL ORIENTATIVO: cada opción ≈{lo_i}–{hi_i} kcal. "
            f"Quedan {r:.0f} kcal; reparte sin exceder el total diario."
        )

    # ── Anti-repetir platos recientes ──
    bloque_anti_repetir = ""
    if db and (modo_funcion or "").strip().lower() == "recomendar_nutricion":
        try:
            rec = get_user_recent_meals(perfil.id) or []
            nombres = list(dict.fromkeys(
                m.get("nombre", "").strip() for m in rec[-12:] if m.get("nombre")
            ))[:8]
            if nombres:
                bloque_anti_repetir = (
                    f"\nVARIACIÓN: No repitas: {', '.join(nombres)}. Elige otras opciones o cambia la proteína."
                )
        except Exception:
            pass

    # ── Favoritos del usuario ──
    bloque_favoritos = ""
    if db:
        try:
            favs = (
                db.query(PreferenciaAlimento)
                .filter(PreferenciaAlimento.client_id == perfil.id)
                .order_by(PreferenciaAlimento.frecuencia.desc()).limit(5).all()
            )
            if favs:
                lista = ", ".join(f"{p.alimento} ({p.frecuencia}x)" for p in favs)
                bloque_favoritos = f"\nCOMIDAS FAVORITAS: {lista}. Prioriza estos ingredientes en sugerencias generales."
        except Exception:
            pass

    # ── Saludo proactivo ──
    bloque_saludo = ""
    if es_saludo:
        pct_dia = min(100, (consumo_real / calorias_meta * 100)) if calorias_meta > 0 else 0
        bloque_saludo = (
            f"\n\nINSTRUCCIÓN SALUDO: Responde con calidez e incluye mini-resumen: "
            f"'Hoy llevas {consumo_real:.0f} de {calorias_meta:.0f} kcal ({pct_dia:.0f}%), "
            f"te faltan {restantes:.0f} kcal. Has quemado {quemadas_real:.0f} kcal.' "
            f"Luego pregunta qué necesita el cliente."
        )

    # ── Perfil ML (Random Forest) ──
    bloque_perfil_ml = ""
    try:
        _act_map = {"Sedentario": 1, "Ligero": 2, "Moderado": 3, "Intenso": 5, "Muy intenso": 6}
        act_lvl  = getattr(perfil, "activity_level", "Moderado") or "Moderado"
        perfil_ml, conf_ml = ml_perfil.predecir_perfil({
            "age":           edad,
            "gender":        "M" if getattr(perfil, "gender", "M") == "M" else "F",
            "weight":        float(perfil.weight or 70),
            "height":        float(perfil.height or 170),
            "workout_freq":  _act_map.get(act_lvl, 3),
            "session_hours": float(getattr(perfil, "session_duration", None) or 1.0),
            "calories":      round(400 + adherencia_pct * 10),
            "fat_pct":       round(max(10, 35 - adherencia_pct * 0.3), 1),
            "water":         round(1.5 + _act_map.get(act_lvl, 3) * 0.2, 1),
            "avg_bpm": 140, "resting_bpm": 65,
            "workout_type":  getattr(perfil, "workout_type", "") or "",
        })
        bloque_perfil_ml = (
            f"\n\nPERFIL ML ({perfil_ml}, {conf_ml}% confianza): "
            f"{ml_perfil.get_tono_asistente(perfil_ml)}"
        )
    except Exception:
        pass

    # ── Recomendaciones Completas (Reemplaza al KNN crudo) ──
    bloque_reco_ml = ""
    if restantes > 100 and modo_funcion == "recomendar_nutricion":
        try:
            from app.services.recomendador_platos import RecomendadorPlatosConfiables
            from app.services.nutrition.plate.plate_builder import PlatoBuilder
            from app.services.nutrition.food.resolver.cache_manager import CacheManager
            from app.services.nutrition.food.resolver.source_resolver import FoodSourceResolver

            # Inicializar motor de recomendación con capacidad de generar nuevos (fallback IA)
            cache = CacheManager(db)
            resolver = FoodSourceResolver(db, cache)
            pb = PlatoBuilder(db, resolver, cache)
            rec = RecomendadorPlatosConfiables(db, plate_builder=pb)

            pct_c      = (consumo_real / calorias_meta) if calorias_meta > 0 else 0
            prot_meta  = plan_hoy_data.get("proteinas_g", 0)
            carbo_meta = plan_hoy_data.get("carbohidratos_g", 0)
            grasa_meta = plan_hoy_data.get("grasas_g", 0)
            
            excluir = [
                m.get("nombre") for m in (get_user_recent_meals(perfil.id) or [])
                if isinstance(m.get("nombre"), str) and m.get("nombre", "").strip()
            ]
            
            # Escalar el objetivo al tamaño de la comida actual (hi_i) en lugar de usar todo el restante del día
            escala = (hi_i / restantes) if restantes > 0 else 1.0
            
            d_prot = max(0, prot_meta - prot_meta * pct_c) * escala
            d_carb = max(0, carbo_meta - carbo_meta * pct_c) * escala
            d_gras = max(0, grasa_meta - grasa_meta * pct_c) * escala

            msg_low = mensaje_usuario.lower() if mensaje_usuario else (mensaje_fuzzy.lower() if mensaje_fuzzy else "")
            
            if any(w in msg_low for w in ["proteina", "proteína", "musculo", "músculo", "proteico"]):
                d_prot = max(35.0, d_prot * 2.0)
            if any(w in msg_low for w in ["carbohidrato", "carbohidratos", "carbo", "carbos", "energia", "energía"]):
                d_carb = max(50.0, d_carb * 2.0)
            if any(w in msg_low for w in ["grasa", "grasas", "lipid", "lípid", "keto", "cetogenico", "cetogénico"]):
                d_gras = max(20.0, d_gras * 2.0)
                
            # Detección de ingrediente específico desde el mensaje del usuario.
            # Orden: términos más específicos/largos primero para evitar falsos positivos.
            # Todas las variantes (con/sin tilde, sinónimos regionales peruanos)
            # se normalizan a un valor canónico usado como clave de búsqueda en BD.
            _ING_MAP = {
                # ── Mariscos (Lambayeque costero) ────────────────────────────
                "langostino": "mariscos",
                "langosta":  "mariscos",
                "mariscos":  "mariscos",
                "camarón":   "mariscos",
                "camaron":   "mariscos",
                "calamar":   "mariscos",
                "mejillon":  "mariscos",
                "mejillón":  "mariscos",
                "almeja":    "mariscos",
                "pulpo":     "mariscos",
                "choro":     "mariscos",
                "concha":    "mariscos",
                "cangrejo":  "mariscos",
                # ── Pescados ────────────────────────────────────────────────
                "anchoveta": "pescado",
                "caballa":   "pescado",
                "cachema":   "pescado",
                "corvina":   "pescado",
                "merluza":   "pescado",
                "tilapia":   "pescado",
                "sardina":   "pescado",
                "jurel":     "pescado",
                "tollo":     "pescado",
                "mero":      "pescado",
                "cabrilla":  "pescado",
                "lisa":      "pescado",
                "salmón":    "salmon",
                "salmon":    "salmon",
                "trucha":    "trucha",
                "atún":      "atun",
                "atun":      "atun",
                "pescado":   "pescado",
                # ── Carnes rojas ─────────────────────────────────────────────
                "cabrito":   "cabrito",
                "cordero":   "cordero",
                "cuy":       "cuy",
                "conejo":    "conejo",
                "pato":      "pato",
                "pavo":      "pavo",
                "chancho":   "cerdo",
                "cerdo":     "cerdo",
                "ternera":   "res",
                "bistec":    "res",
                "lomo":      "res",
                "res":       "res",
                "carne":     "carne",
                # ── Aves ────────────────────────────────────────────────────
                "pechuga":   "pollo",
                "pollo":     "pollo",
                # ── Huevo y lácteos ──────────────────────────────────────────
                "huevo":     "huevo",
                "yogurt":    "yogur",
                "yogur":     "yogur",
                "queso":     "queso",
                "leche":     "leche",
                "mantequilla": "lacteos",
                # ── Legumbres (menestras peruanas) ───────────────────────────
                "garbanzo":  "garbanzo",
                "pallares":  "pallares",
                "menestra":  "menestra",
                "arveja":    "arveja",
                "haba":      "haba",
                "frijol":    "frejol",
                "frejol":    "frejol",
                "lentejón":  "lenteja",
                "lenteja":   "lenteja",
                # ── Cereales y granos ────────────────────────────────────────
                "spaghetti": "pasta",
                "tallarín":  "pasta",
                "tallarin":  "pasta",
                "fideos":    "pasta",
                "pasta":     "pasta",
                "cebada":    "cebada",
                "trigo":     "trigo",
                "quinoa":    "quinua",
                "quinua":    "quinua",
                "avena":     "avena",
                "arroz":     "arroz",
                # ── Tubérculos y raíces ──────────────────────────────────────
                "boniato":   "camote",
                "camote":    "camote",
                "yuca":      "yuca",
                "papa":      "papa",
                # ── Otros vegetales ──────────────────────────────────────────
                "choclo":    "choclo",
                "maíz":      "choclo",
                "maiz":      "choclo",
                "brócoli":   "verdura",
                "brocoli":   "verdura",
                "espinaca":  "verdura",
                "zanahoria": "verdura",
                "pepino":    "verdura",
                "lechuga":   "verdura",
                "tomate":    "verdura",
                "acelga":    "verdura",
                "ensalada":  "ensalada",
                "vegetal":   "verdura",
                "verdura":   "verdura",
                # ── Frutas ───────────────────────────────────────────────────
                "aguacate":  "palta",
                "palta":     "palta",
                "plátano":   "platano",
                "platano":   "platano",
                "lúcuma":    "lucuma",
                "lucuma":    "lucuma",
                "maracuyá":  "fruta",
                "maracuya":  "fruta",
                "papaya":    "fruta",
                "mango":     "mango",
                "sandía":    "fruta",
                "sandia":    "fruta",
                "melón":     "fruta",
                "melon":     "fruta",
                "piña":      "fruta",
                "pina":      "fruta",
                "fresa":     "fruta",
                "manzana":   "fruta",
                "naranja":   "fruta",
                "uva":       "fruta",
                "pera":      "fruta",
                # ── Frutos secos y semillas ──────────────────────────────────
                "almendra":  "fruto_seco",
                "pecana":    "fruto_seco",
                "nuez":      "fruto_seco",
                "maní":      "mani",
                "mani":      "mani",
                "linaza":    "semilla",
                "chía":      "semilla",
                "chia":      "semilla",
                # ── Proteína vegetal ─────────────────────────────────────────
                "soya":      "soya",
                "tofu":      "tofu",
                # ── Pan y derivados ──────────────────────────────────────────
                "tostada":   "pan",
                "pan":       "pan",
            }
            ingrediente_clave = None
            for kw, valor in _ING_MAP.items():
                if kw in msg_low:
                    ingrediente_clave = valor
                    break
                
            platos = rec.recomendar(
                client_id=perfil.id,
                deficit_kcal=hi_i,
                deficit_proteina=d_prot,
                deficit_carb=d_carb,
                deficit_grasas=d_gras,
                momento_dia=momento_comida.lower(),
                n=3,
                excluir_nombres=excluir,
                ingrediente_clave=ingrediente_clave,
                condiciones_dieta=list(perfil.medical_conditions or []),
            )

            if platos:
                lista_platos = ""
                for i, p in enumerate(platos):
                    m = p['macros']
                    ings = p.get('ingredientes_str', '')
                    lista_platos += (
                        # Bug 2 fix: usar '→ Nombre' en vez de '{i+1}. Nombre'
                        # para evitar que el LLM use los números como títulos genéricos
                        f"→ {p['nombre']}\n"
                        f"   Macros: Kcal: {m['calorias']:.0f}, P: {m['proteinas_g']:.1f}g, C: {m['carbohidratos_g']:.1f}g, G: {m['grasas_g']:.1f}g\n"
                        f"   Ingredientes exactos: {ings}\n"
                    )
                bloque_reco_ml = (
                    f"\n\nSUGERENCIAS DE PLATOS CONFIABLES (OBLIGATORIO SUGERIR ESTOS):\n{lista_platos}"
                    "Debes sugerir EXACTAMENTE estos platos en tu respuesta. "
                    "Usa los nombres provistos como títulos de las opciones, respeta sus macros, y copia "
                    "los ingredientes exactos proporcionados. "
                    "NOTA PROFESIONAL: Si el cliente pide explícitamente algo que contradiga su meta (ej. 'alto en grasa' o 'puros carbohidratos' "
                    "mientras busca perder peso), actúa como un coach profesional: edúcalo amablemente sobre por qué no es lo ideal "
                    "para su objetivo actual, y luego preséntale estas 3 opciones como alternativas saludables y balanceadas que SÍ "
                    "le ayudarán a llegar a su meta."
                )

            # ── KNN complementario — alimentos individuales por similitud coseno ──
            if ml_recomendador is not None:
                try:
                    _knn_sugs = ml_recomendador.obtener_recomendaciones(
                        calorias_faltantes=hi_i,
                        prote_faltante=d_prot,
                        carbo_faltante=d_carb,
                        grasa_faltante=d_gras,
                        n_recomendaciones=6,   # extra para compensar filtro dietético
                        excluir_nombres=excluir,
                    )
                    # Filtrar por restricciones dietéticas del usuario
                    if perfil.medical_conditions:
                        try:
                            from app.services.recomendador_platos import _tokens_prohibidos
                            _knn_tok = _tokens_prohibidos(list(perfil.medical_conditions))
                            if _knn_tok:
                                _knn_sugs = [
                                    k for k in (_knn_sugs or [])
                                    if not any(
                                        t in (k.get("alimento", "") or "").lower()
                                        for t in _knn_tok
                                    )
                                ][:3]
                        except Exception:
                            pass
                    if _knn_sugs:
                        _knn_lineas = "\n".join(
                            f"- {k['alimento']}: {k['calorias_100g']:.0f} kcal/100g "
                            f"(P:{k['proteina_100g']:.1f}g C:{k.get('carbohidratos_100g', 0):.1f}g "
                            f"G:{k['grasas_100g']:.1f}g) — similitud {k['similitud']}%"
                            for k in _knn_sugs
                        )
                        bloque_reco_ml += (
                            "\n\nALIMENTOS COMPLEMENTARIOS KNN (snacks o acompañantes sugeridos):\n"
                            + _knn_lineas
                            + "\nMenciónales brevemente si el cliente necesita complementar su ingesta."
                        )
                except Exception:
                    pass
        except Exception as e:
            print(f"[Prompt] Error RecomendadorPlatosConfiables: {e}")

    # ── Justificación personalizada: objetivo + condiciones clínicas ─────────
    # Le dice al LLM que explique POR QUÉ cada sugerencia es adecuada para este usuario,
    # vinculando el objetivo (ganar masa, perder peso…) y las condiciones (diabetes, vegano…).
    _objetivo_raw = (getattr(perfil, "goal", "Mantenimiento") or "Mantenimiento").strip()
    _OBJ_FRASES = {
        "ganar masa":    "ganar masa muscular (superávit calórico + alta proteína)",
        "ganar":         "ganar masa muscular (superávit calórico + alta proteína)",
        "perder peso":   "perder peso (déficit calórico + proteína para conservar músculo)",
        "perder":        "perder peso (déficit calórico + proteína para conservar músculo)",
        "mantenimiento": "mantener el peso actual (balance calórico equilibrado)",
        "mantener peso": "mantener el peso actual (balance calórico equilibrado)",
        "mantener":      "mantener el peso actual (balance calórico equilibrado)",
    }
    _obj_txt = _OBJ_FRASES.get(_objetivo_raw.lower(), f"objetivo: {_objetivo_raw}")

    _cond_justif: list[str] = []
    for _c in list(perfil.medical_conditions or []):
        _cl = _c.lower()
        if "diabetes" in _cl:
            _cond_justif.append("diabetes (bajo índice glucémico, sin azúcares refinados)")
        elif "vegano" in _cl:
            _cond_justif.append("alimentación 100% vegana (sin ningún producto animal)")
        elif "vegetariano" in _cl:
            _cond_justif.append("vegetarianismo (sin carnes ni pescados, lácteos/huevos permitidos)")
        elif "intolerancia" in _cl:
            _cond_justif.append("intolerancia a la lactosa (sin lácteos)")
        elif "celí" in _cl or "celiaco" in _cl:
            _cond_justif.append("celiaquía / sin gluten")
        elif "hipertensión" in _cl or "hipertension" in _cl:
            _cond_justif.append("hipertensión (bajo en sodio, evitar embutidos y frituras)")
        elif "renal" in _cl:
            _cond_justif.append("enfermedad renal (controlar proteína total y potasio)")

    _hay_condiciones = bool(_cond_justif)
    _cond_txt = "; ".join(_cond_justif) if _hay_condiciones else ""

    bloque_justificacion = (
        f"\n\nOBJETIVO ACTIVO: {_obj_txt}."
        + (f"\nCONDICIONES CLÍNICAS: {_cond_txt}." if _hay_condiciones else "")
        + "\n\nREGLA DE JUSTIFICACIÓN (aplicar en MODO RECIPE):"
        "\n• ANTES de las tarjetas escribe 1–2 frases explicando por qué las opciones"
        " encajan con el objetivo y las condiciones del usuario."
        "\n  Ejemplo objetivo solo:    'Como tu meta es ganar masa, prioricé opciones"
        " ricas en proteína y con superávit calórico moderado:'"
        "\n  Ejemplo con condición:    'Considerando que buscas perder peso y tienes"
        " diabetes, elegí platos altos en proteína, sin azúcar y de bajo índice glucémico:'"
        + (
            "\n• Después del nombre de cada tarjeta añade 1 línea corta (fuera de [CALOFIT_LIST])"
            " con el beneficio principal para su condición."
            "\n  Ejemplo: '→ Apto para diabéticos: sin azúcares refinados, proteína de legumbres.'"
            if _hay_condiciones else ""
        )
    )

    # ── Bloque meta cumplida — bloquea RECIPE cuando el usuario ya llegó a su meta ──
    bloque_meta_cumplida = ""
    if restantes <= 0 and calorias_meta > 0:
        bloque_meta_cumplida = (
            f"\n\n🚫 META CALÓRICA DEL DÍA CUMPLIDA (REGLA CRÍTICA — MÁXIMA PRIORIDAD):"
            f"\nEl usuario ya alcanzó su meta: {consumo_real:.0f} kcal consumidas "
            f"/ {calorias_meta:.0f} kcal meta. Le quedan 0 kcal disponibles."
            "\n• PROHIBIDO ABSOLUTO: sugerir comidas, platos o recetas adicionales. NO usar MODO RECIPE."
            "\n• OBLIGATORIO: informar amablemente que la meta del día está cubierta."
            "\n• Si el usuario insiste en comer, sugiere SOLO agua, infusión o una fruta muy pequeña (≤80 kcal)."
            "\n• Usa MODO PROGRESS o INFO. NUNCA RECIPE en esta situación."
        )

    # ── Refuerzo proteína alta para ganar masa + dieta vegana/vegetariana ────
    _es_ganar_masa = _objetivo_raw.lower() in ("ganar masa", "ganar", "ganar músculo", "ganar musculo")
    _es_dieta_plant = any(
        p in (texto_dieta or "").lower()
        for p in ("vegano", "vegetariano", "plant")
    )
    if _es_ganar_masa and _es_dieta_plant:
        bloque_justificacion += (
            "\n• PROTEÍNA VEGETAL ALTA (obligatorio para ganar masa sin carne):"
            " cada opción debe aportar ≥20 g de proteína."
            " Prioriza: tofu (~18 g/100 g), seitán (~25 g/100 g), tempeh (~19 g/100 g),"
            " edamame (~11 g/100 g), combinación quinua + lentejas. EVITA 'ensaladas ligeras'"
            " o sopas con <15 g proteína — son insuficientes para superávit muscular."
        )

    # ── Regla 10 — Restricciones dietéticas del usuario ──────────────────────
    _bloque_regla10 = ""
    _reglas_dieta_r10: list[str] = []
    _conds_all = list(perfil.medical_conditions or [])
    if "Vegano" in _conds_all:
        _reglas_dieta_r10.append(
            "VEGANO — PROHIBIDO absolutamente en CADA INGREDIENTE de CADA PLATO: "
            "pollo, pechuga, carne, res, cerdo, pato, pavo, cabrito, pescado, camarón, mariscos, "
            "huevo, leche, queso, yogur, mantequilla y cualquier derivado animal. "
            "ÚNICAMENTE ingredientes 100% vegetales. "
            "Fuentes de proteína PERMITIDAS: lentejas, garbanzos, frijoles, pallares, quinua, "
            "tofu, tempeh, seitán, edamame, maní, semillas de chía, almendras. "
            "Si un plato tiene cualquier ingrediente animal → descártalo y elige otro."
        )
    elif "Vegetariano" in _conds_all:
        _reglas_dieta_r10.append(
            "VEGETARIANO — PROHIBIDO en cualquier ingrediente: carne de res, cerdo, pollo, pavo, "
            "pato, cabrito, pescado, camarón y mariscos. "
            "Huevos y lácteos SÍ permitidos. "
            "Si un plato contiene carne o pescado → descártalo completamente."
        )
    if any("intolerancia" in c.lower() for c in _conds_all):
        _reglas_dieta_r10.append(
            "INTOLERANCIA A LA LACTOSA — PROHIBIDO: leche, queso, yogurt, "
            "mantequilla, crema de leche, quesillo."
        )
    if any("celí" in c.lower() or "celiaco" in c.lower() for c in _conds_all):
        _reglas_dieta_r10.append(
            "CELÍACO — PROHIBIDO: trigo, avena, cebada, centeno, pan de trigo, "
            "pasta, fideos, galletas con gluten."
        )
    if any("diabetes" in c.lower() for c in _conds_all):
        _reglas_dieta_r10.append(
            "DIABETES — PROHIBIDO: azúcar refinada, miel, mermelada, gaseosas, "
            "chocolates, helados, pasteles, bebidas azucaradas."
        )
    if _reglas_dieta_r10:
        _bloque_regla10 = (
            f"\n10. RESTRICCIÓN DIETÉTICA OBLIGATORIA (perfil: {texto_dieta}):\n"
            + "\n".join(f"   • {r}" for r in _reglas_dieta_r10)
            + "\n   ⚠️ INCUMPLIR ESTA REGLA ES ERROR GRAVE. Verifica que CADA opción "
            "cumpla TODAS las restricciones antes de responder."
            + "\n   ✅ JUSTIFICACIÓN OBLIGATORIA: Por cada plato recomendado añade 1 línea "
            "entre paréntesis explicando por qué es apto para el perfil del usuario."
            + "\n      Ejemplo: '(100% vegetal · proteína de quinua · sin derivados animales)'"
        )

    # ── Bloque Nutricionista (máxima prioridad) ──
    nota = getattr(perfil, "nutri_weekly_note", None)
    # Resumen de restricción dietética para el bloque del nutricionista
    _resumen_dieta = ""
    if "Vegano" in _conds_all:
        _resumen_dieta = (
            "\n🚫 DIETA VEGANA ACTIVA — OVERRIDE TOTAL: ningún plato puede contener "
            "pollo, carne, pescado, camarón, huevo, leche ni derivados animales. "
            "Esta restricción ANULA cualquier alimento animal en la lista PRIORIZAR.\n"
        )
    elif "Vegetariano" in _conds_all:
        _resumen_dieta = (
            "\n🚫 DIETA VEGETARIANA ACTIVA — OVERRIDE: ningún plato puede contener "
            "carne, pollo ni pescado. Esta restricción ANULA esos ítems en PRIORIZAR.\n"
        )
    bloque_nutri = (
        f"\n⚠️ INSTRUCCIONES OBLIGATORIAS DEL NUTRICIONISTA (NO NEGOCIABLES):\n"
        f"• FOCO: {foco}\n"
        f"• PRIORIZAR: {', '.join(alimentos_rec) if alimentos_rec else 'Sin restricción especial'}\n"
        f"• PROHIBIDOS: {', '.join(alimentos_pro) if alimentos_pro else 'Ninguno'} — NUNCA los sugieras.\n"
        + (f"• META SEMANAL: \"{nota}\"\n" if nota else "")
        + _resumen_dieta
        + "━" * 60 + "\n"
    )

    return (
        bloque_nutri
        + f"Eres el coach personal de {perfil.first_name}. "
        f"PERFIL: {perfil.weight}kg, {perfil.height}cm, {edad} años, {perfil.gender}. "
        f"Entrenamiento preferido: {getattr(perfil, 'workout_type', None) or 'No especificado'} · "
        f"Sesión habitual: {int((getattr(perfil, 'session_duration', 1.0) or 1.0) * 60)} min. "
        f"ALERGIAS: {texto_alergias}. DIETA: {texto_dieta}. CONDICIONES: {texto_condiciones}."
        f"{bloque_hora}{bloque_rango_kcal}{bloque_anti_repetir}{bloque_favoritos}"
        f"{bloque_perfil_ml}{bloque_reco_ml}"
        f"{bloque_justificacion}"
        f"{bloque_meta_cumplida}"
        f"\nSTATUS DEL DÍA: Meta: {calorias_meta} kcal | Consumido: {consumo_real} kcal | "
        f"Restante: {restantes:.0f} kcal | Adherencia: {adherencia_pct:.0f}% | {mensaje_fuzzy}."
        "\n\nREGLAS DE INTENCIÓN Y FORMATO (OBLIGATORIO):"
        "\n0. BREVEDAD OBLIGATORIA en modo INFO/CHAT/PROGRESS:"
        "\n   • Máximo 2-3 oraciones o una lista corta de máximo 4 ítems."
        "\n   • PROHIBIDO: recetas completas, instrucciones de preparación paso a paso, listas de ingredientes."
        "\n   • Si recomendas un plato en texto libre: solo nombre + calorías aproximadas. Nada más."
        "\n   • El usuario pedirá más detalle si lo necesita."
        "\n1. Progreso/dudas → MODO PROGRESS o INFO: solo texto. ¡PROHIBIDO [CALOFIT_HEADER] salvo que pidan comida!"
        "\n2. Piden opciones/recetas → MODO RECIPE: SIEMPRE 2–3 opciones con protocolo CALOFIT completo."
        "\n3. Indica intención: [CALOFIT_INTENT:CATEGORIA] (INFO, RECIPE, PROGRESS, POWER, LOG, ALERT)."
        "\n4. NO uses [CALOFIT_QUESTION_TYPE]."
        "\n5. MODO RECIPE: HEADER→LIST→ACTION→STATS por cada opción. [CALOFIT_LIST] obligatorio (≥3 ingredientes con gramos y kcal)."
        "\n6. USA EL NOMBRE REAL DEL PLATO en [CALOFIT_HEADER]: NUNCA nombres genéricos."
        "\n   ❌ INCORRECTO: [CALOFIT_HEADER:Sugerencia 1] o [CALOFIT_HEADER:Opción 2]"
        "\n   ✅ CORRECTO:   [CALOFIT_HEADER:Arroz Con Pollo] o [CALOFIT_HEADER:Lomo Saltado]"
        "\n   El nombre en CALOFIT_HEADER DEBE SER EXACTAMENTE el nombre del plato de la lista SUGERENCIAS."
        "\n7. PROHIBIDOS los alimentos del Nutricionista aunque el cliente los pida."
        "\n8. COCINA PERUANA: arroz con pollo, cebiche, lomo saltado, ají de gallina, causa, tallarines, etc."
        "\n9. IDENTIDAD LAMBAYEQUE — OBLIGATORIO en sugerencias marinas:"
        "\n   • Prioridad de especies: Caballa → Lisa → Mero → Tollo (en ese orden)."
        "\n   • PROHIBIDO sugerir Atún o Salmón salvo que el usuario los pida por nombre."
        "\n   • El Ceviche NUNCA lleva palta, tomate ni zanahoria."
        f"{_bloque_regla10}"
        f"{bloque_saludo}"
        f"{bloque_prompt_modo_funcion(modo_funcion)}"
    )


async def enriquecer_prompt_con_bd(
    prompt_final: str, mensaje: str, modo_funcion: str, db: Optional[Session]
) -> str:
    """Inyecta en el prompt ejercicios y platos con macros exactos desde Postgres."""
    if db is None:
        return prompt_final

    # DB-hint para ejercicios por grupo muscular
    if modo_funcion == "recomendar_ejercicio":
        try:
            from sqlalchemy import text as _sql
            low    = (mensaje or "").lower()
            _MAP   = {
                "biceps": ["%biceps%", "%bíceps%"], "triceps": ["%triceps%", "%tríceps%"],
                "pecho": ["%pecho%", "%pector%"], "espalda": ["%espalda%", "%dorsal%"],
                "pierna": ["%pierna%", "%cuadr%", "%femoral%", "%glute%", "%pantorrilla%", "%gemelo%"],
                "gluteos": ["%glute%", "%glúte%"], "hombro": ["%hombro%", "%delto%"],
                "core": ["%core%", "%abd%"],
            }
            _KEYS = {
                "biceps": r"biceps|bíceps", "triceps": r"triceps|tríceps",
                "pecho": r"pecho|pector", "espalda": r"espalda|dorsal",
                "pierna": r"pierna|cuadriceps|cuádriceps|femoral|pantorrilla|gemelo",
                "gluteos": r"gluteo|glúteo", "hombro": r"hombros?|deltoid",
                "core": r"abdomen|abdominal|core",
            }
            
            # 1. Encontrar TODOS los objetivos mencionados
            objetivos_detectados = [k for k, p in _KEYS.items() if re.search(p, low)]
            
            if objetivos_detectados:
                pats_completos = []
                for obj in objetivos_detectados:
                    pats_completos.extend(_MAP[obj])
                
                # Excluir ejercicios culturales/deportivos que no corresponden a gym:
                # danzas folclóricas, deportes de equipo, actividades al aire libre.
                _IDS_NO_GYM = (
                    'huayno_zapateo', 'pichanga_futbol', 'voley_calle',
                    'trote_ligero', 'subir_escaleras_cerro', 'pilates',
                )
                _excl_ids = ", ".join(f"'{eid}'" for eid in _IDS_NO_GYM)
                rows = db.execute(
                    _sql(
                        "SELECT id, nombre, musculo_principal, met FROM ejercicios "
                        "WHERE (" + " OR ".join(f"musculo_principal ILIKE :p{i}" for i in range(len(pats_completos))) + ") "
                        f"  AND id NOT IN ({_excl_ids}) "
                        "  AND tipo NOT IN ('Strongman') "
                        "ORDER BY met DESC NULLS LAST, nombre ASC LIMIT 15"
                    ),
                    {f"p{i}": pats_completos[i] for i in range(len(pats_completos))},
                ).fetchall()
                
                if rows:
                    listado = "\n".join(
                        f"- {r[1]} (id={r[0]}, musculo={r[2]}, MET={r[3]})" 
                        for r in rows
                    )
                    prompt_final += (
                        f"\n\n🏋️ EJERCICIOS EN TU BASE DE DATOS (CATÁLOGO OFICIAL) 🏋️\n"
                        f"Objetivos detectados: {', '.join(objetivos_detectados)}\n"
                        f"Catálogo Oficial:\n{listado}\n\n"
                        f"INSTRUCCIÓN PARA ENTRENAMIENTO (HÍBRIDO DB + LLM):\n"
                        f"1. Tienes total libertad para diseñar la mejor rutina posible usando tu conocimiento experto.\n"
                        f"2. PRIORIZA los ejercicios listados arriba si encajan bien con el objetivo.\n"
                        f"3. Si conoces ejercicios mejores que NO están en esta lista, "
                        f"TIENES PERMISO para incluirlos e inventar las instrucciones técnicas necesarias.\n"
                        f"4. NUNCA escribas los códigos `id=...`, `musculo=...` ni `MET=...` en tu respuesta al usuario. Usa SOLO el nombre del ejercicio limpio.\n"
                        f"5. Asume que el usuario entrena en un GIMNASIO equipado. PRIORIZA el uso de pesas, barras, mancuernas y máquinas.\n"
                        f"   PROHIBIDO ABSOLUTO en contexto de gym: danzas folclóricas (Huayno, Marinera), deportes de equipo (Pichanga, Vóley),\n"
                        f"   actividades al aire libre ('subir cerros', 'trotar en el parque'). Estos NO son ejercicios de gym.\n"
                        f"6. SIEMPRE cruza tu rutina con las CONDICIONES médicas del usuario. "
                        f"Descarta CUALQUIER ejercicio (del catálogo o inventado por ti) que le haría daño.\n"
                    )
        except Exception as e:
            print(f"Error extrayendo ejercicios: {e}")

    # DB-hint para platos mencionados en el mensaje
    if modo_funcion in ("recomendar_nutricion", "responder_consulta", "otro"):
        try:
            from app.services.asistente_nutricion import (
                _norm_nombre_plato as _nn,
                _buscar_plato_bd_por_nombre as _buscar_bd,
                _extraer_platos_del_mensaje as _extraer_platos,
            )
            hints = []
            for cp in _extraer_platos(mensaje):
                pd = _buscar_bd(db, _nn(cp))
                if pd and float(pd.get("calorias") or 0) > 0:
                    hints.append(
                        f"- {cp.title()}: {pd['calorias']:.0f} kcal | "
                        f"P: {pd['proteinas_g']:.1f}g | C: {pd['carbohidratos_g']:.1f}g | G: {pd['grasas_g']:.1f}g"
                    )
            if hints:
                prompt_final += (
                    "\n\nPLATOS CON MACROS EXACTOS (usa EXACTAMENTE estos valores):\n"
                    + "\n".join(hints)
                    + "\nIMPORTANTE: usa el nombre exacto como título de la sección.\n"
                )
        except Exception:
            pass

    return prompt_final


# ── Post-procesado de respuesta ───────────────────────────────────────────────

def clasificar_intencion_respuesta(respuesta_estructurada: dict, mensaje: str) -> None:
    """Clasifica si la respuesta debe mostrarse como tarjeta (card) o texto plano."""
    msg_low    = mensaje.lower()
    texto_ai   = respuesta_estructurada.get("texto_conversacional", "").lower()
    intent_ai  = str(respuesta_estructurada.get("intent") or "INFO").upper().strip()

    m = re.search(r'\[\s*CALOFIT_INTENT\s*:\s*(.*?)\s*\]', texto_ai)
    if m and intent_ai not in ("RECIPE", "POWER", "LOG"):
        intent_ai = m.group(1).upper().strip()

    tipo_pregunta = str(respuesta_estructurada.get("modo_funcion") or "otro").upper().strip()
    respuesta_estructurada["intent_ai"]    = intent_ai
    respuesta_estructurada["tipo_pregunta"] = tipo_pregunta

    secciones_comida = [s for s in respuesta_estructurada.get("secciones", []) if s.get("tipo") == "comida"]
    _verbos_log = ("comi", "comí", "almorcé", "almorce", "desayuné", "desayune",
                   "cené", "cene", "tomé", "tome", "bebí", "bebi")
    _es_log_verb = any(v in msg_low for v in _verbos_log)

    es_info_directa = (
        intent_ai in ["INFO", "PROGRESS", "NORMAL"] or
        (
            intent_ai != "LOG" and not _es_log_verb and
            len(secciones_comida) == 1 and
            not any(k in msg_low for k in [
                "opcion", "opciones", "receta", "menú", "menu", "cena",
                "almuerzo", "desayuno", "suger", "dame", "recomienda",
            ])
        )
    )

    secciones_conservar = []
    for sec in respuesta_estructurada.get("secciones", []):
        tipo = sec.get("tipo")
        if tipo == "comida":
            if intent_ai in ["INFO", "PROGRESS"] and not any(
                k in msg_low for k in
                ["como", "comer", "opcion", "opciones", "receta", "menú", "menu",
                 "cena", "almuerzo", "desayuno", "suger", "dame", "recomienda", "plan"]
            ):
                continue
            tiene_pasos = bool(sec.get("pasos") or sec.get("preparacion"))
            if not tiene_pasos and es_info_directa:
                titulo     = re.sub(r'\[/?[A-Z_]+.*$', '', sec.get("nombre", "Alimento")).strip()
                lista      = "\n".join(f"• {ing}" for ing in sec.get("ingredientes", []))
                stats_raw  = sec.get("macros", "")
                stats      = stats_raw.replace("P:", "🥚 P:").replace("C:", "🍞 C:").replace("G:", "🥑 G:")
                texto_extra = f"\n\n🍏 **{titulo}**\n{lista}"
                if stats.strip():
                    texto_extra += f"\n\n📊 {stats}"
                actual = respuesta_estructurada.get("texto_conversacional", "")
                respuesta_estructurada["texto_conversacional"] = (actual + texto_extra).strip()
                continue
            secciones_conservar.append(sec)
        else:
            secciones_conservar.append(sec)

    respuesta_estructurada["secciones"] = secciones_conservar


def limpiar_tags_calofit(respuesta_estructurada: dict) -> None:
    """Elimina residuos de tags CALOFIT del texto y secciones."""
    # Con corchetes: [CALOFIT_INTENT:LOG] / [/CALOFIT_HEADER]
    _re = re.compile(r'\[/?CALOFIT_[A-Z_:]*.*?\]', re.IGNORECASE)
    # Sin corchetes: cuando el LLM filtra el texto dentro de un HEADER
    _re_bare = re.compile(
        r'\bCALOFIT_(?:INTENT|HEADER|LIST|ACTION|STATS|QUESTION_TYPE)'
        r'(?:\s*[:/]\s*\w+)?\b',
        re.IGNORECASE,
    )

    def _limpiar(t: str) -> str:
        return _re_bare.sub("", _re.sub("", t)).strip()

    texto = respuesta_estructurada.get("texto_conversacional", "")
    respuesta_estructurada["texto_conversacional"] = sanear_texto_conversacional_recipe(
        _limpiar(texto)
    )
    for s in respuesta_estructurada.get("secciones", []):
        for k in ["nombre", "macros", "gasto_calorico_estimado", "nota"]:
            if s.get(k):
                s[k] = _limpiar(str(s[k]))
        for k in ["ingredientes", "ejercicios", "preparacion", "tecnica", "instrucciones"]:
            if s.get(k) and isinstance(s[k], list):
                s[k] = [_limpiar(str(item)) for item in s[k]]


def detectar_intencion_principal(respuesta_estructurada: dict, mensaje: str) -> str:
    """Devuelve el tema visual para Flutter: RECIPE, POWER, PROGRESS, SUCCESS, DANGER, INFO."""
    secciones  = respuesta_estructurada.get("secciones", [])
    intent_ai  = respuesta_estructurada.get("intent_ai", "INFO")
    msg_low    = mensaje.lower()
    modo_fn    = (respuesta_estructurada.get("modo_funcion") or "").strip().lower()
    tipo_p     = (respuesta_estructurada.get("tipo_pregunta") or "").upper()
    texto_full = (respuesta_estructurada.get("texto_conversacional", "") + mensaje).lower()
    tipos      = [s.get("tipo") for s in secciones]

    if any(s.get("tipo") == "alerta" for s in secciones):
        return "DANGER"
    if intent_ai == "PROGRESS" or "balance" in msg_low or "progreso" in msg_low:
        return "PROGRESS"
    if "anotado" in texto_full or "registrado" in texto_full or intent_ai == "LOG":
        return "SUCCESS"
    if modo_fn == "recomendar_nutricion" or "RECOMENDAR_NUTRICION" in tipo_p:
        return "RECIPE"
    if "ejercicio" in tipos or any(k in texto_full for k in ["entren", "ejercicio", "rutina"]) or intent_ai == "POWER":
        return "POWER"
    if "comida" in tipos and "ejercicio" not in tipos:
        return "RECIPE"
    if any(
        k in texto_full for k in
        ["receta", "sugerencia", "opcion", "menú", "puedo comer", "qué comer", "cena",
         "almuerzo", "desayuno", "plato", "según mi plan"]
    ) or intent_ai == "RECIPE":
        return "RECIPE"
    if any(k in msg_low for k in ["cuántas", "qué tiene", "qué es", "dime sobre"]):
        return "INFO"
    return intent_ai if intent_ai in ("INFO", "RECIPE", "POWER", "PROGRESS", "SUCCESS", "DANGER") else "INFO"


# ── Rescue NLP (LOG sin tarjeta) ─────────────────────────────────────────────

async def rescue_nlp_log(
    resp_est: dict, mensaje: str, perfil, ia_engine, db
) -> None:
    """
    Si el LLM declaró LOG pero no generó ninguna tarjeta de comida,
    ejecuta NLPFoodExtractor y añade la sección resultante.
    """
    if str(resp_est.get("intent") or "").upper() != "LOG":
        return
    # Saltar solo si alguna sección realmente proviene del mensaje del usuario
    # (marcada con _origen_usuario en procesar_secciones_comida).
    # Sugerencias proactivas del LLM (ej. "Palta ensalada" cuando el usuario
    # dijo "comí pan con pollo") no tienen _origen_usuario → rescue sigue.
    if any(
        s.get("tipo") == "comida" and s.get("_origen_usuario")
        for s in (resp_est.get("secciones") or [])
    ):
        return
    try:
        import uuid
        from app.services.asistente_nutricion import (
            _limpiar_nombre_plato_bd,
            add_user_recent_meal,
            set_consulta_cached,
        )
        from app.services.nlp_food_extractor import NLPFoodExtractor

        res = await NLPFoodExtractor(ia_engine, db).extraer(mensaje)
        if not (res and res.calorias_total > 0):
            return
        nombre  = (_limpiar_nombre_plato_bd(res.nombres[0] if res.nombres else "Comida")).title()
        cid     = str(uuid.uuid4())
        payload = {
            "calorias":        round(res.calorias_total, 1),
            "proteinas_g":     round(res.proteinas_total, 1),
            "carbohidratos_g": round(res.carbohidratos_total, 1),
            "grasas_g":        round(res.grasas_total, 1),
            "nombre": nombre, "ingredientes": [],
        }
        set_consulta_cached(cid, payload)
        try:
            add_user_recent_meal(perfil.id, payload)
        except Exception:
            pass
        mcn = (
            f"Cal: {payload['calorias']}kcal | P: {payload['proteinas_g']}g | "
            f"C: {payload['carbohidratos_g']}g | G: {payload['grasas_g']}g"
        )
        resp_est.setdefault("secciones", []).append({
            "tipo": "comida", "nombre": nombre,
            "macros": mcn, "macros_cache": mcn,
            "ingredientes": getattr(res, "ingredientes", []),
            "preparacion": [], "consulta_id": cid,
        })
        print(f"[NLP-Rescue] LOG sin card → rescatado: {nombre} {payload['calorias']} kcal")
    except Exception as e:
        print(f"[NLP-Rescue] Error: {e}")


# ── Helpers de respuesta estructurada ────────────────────────────────────────

def respuesta_info_faltante(perfil, modo_funcion: str, falt) -> dict:
    """Devuelve el payload de 'necesito más información' para el frontend."""
    return {
        "asistente": "CaloFit IA", "usuario": perfil.first_name,
        "intencion": "INFO", "tipo_pregunta": (modo_funcion or "otro").upper(),
        "alerta_salud": False,
        "data_cientifica": {"progreso_diario": {}, "macros": {}},
        "respuesta_ia": "",
        "respuesta_estructurada": {
            "intent": "INFO", "modo_funcion": modo_funcion,
            "tipo_pregunta": (modo_funcion or "otro").upper(),
            "texto_conversacional": f"{falt.question}\n\nEjemplos: {', '.join(falt.suggested_options)}.",
            "secciones": [], "needs_more_info": True,
            "missing_fields": falt.missing_fields,
        },
    }


def respuesta_fallo_llm(
    perfil, consumo_real: float, calorias_meta: float,
    quemadas_real: float, respuesta_ia: str, modo_funcion: str,
) -> dict:
    """Devuelve el payload de error cuando el LLM falla o está offline."""
    return {
        "asistente": "CaloFit IA", "usuario": perfil.first_name,
        "intencion": "INFO", "tipo_pregunta": "INFO", "alerta_salud": False,
        "data_cientifica": {
            "progreso_diario": {
                "consumido": round(consumo_real, 1),
                "meta":      round(calorias_meta, 1),
                "restante":  round(max(0.0, calorias_meta - consumo_real + quemadas_real), 1),
                "quemado":   round(quemadas_real, 1),
            },
            "macros": {},
        },
        "respuesta_ia": respuesta_ia,
        "respuesta_estructurada": {
            "intent": "INFO", "modo_funcion": modo_funcion,
            "tipo_pregunta": "INFO",
            "texto_conversacional": respuesta_ia,
            "secciones": [],
        },
    }
