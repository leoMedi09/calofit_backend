"""
Rutas del Asistente del Cliente — "CaloFit Coach Personal"

Endpoints:
  POST /consultar          → Consulta de chat (nutrición, recetas, rutinas, progreso)
  POST /log-inteligente    → Registro de comida/ejercicio por texto o voz
  POST /confirmar-registro → Confirmar registro desde tarjeta interactiva (consulta_id)

Toda la lógica de negocio está delegada a:
  app.services.asistente.asistente_service.AsistenteService
"""

import traceback
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.services.asistente.asistente_service import asistente_service
from app.models.historial import SugerenciaGuardada
from app.models.client import Client

router = APIRouter()


class ChatRequest(BaseModel):
    mensaje: str
    historial: list = None
    contexto_manual: str = None
    override_ia: str = None
    consulta_id: str = None


class RegistroManualAlimentoRequest(BaseModel):
    nombre: str
    calorias: float
    proteinas_g: float = 0
    carbohidratos_g: float = 0
    grasas_g: float = 0
    porcion_g: float = 0  # gramos por porción (si lo sabes). si 0, se asume 100g.
    categoria: str = "manual"
    unidad: str | None = None  # "botella", "vaso", "taza", "porción"
    gramos_por_unidad: float | None = None  # si unidad existe, cuantos gramos equivale 1 unidad


class CalcularEjercicioRequest(BaseModel):
    nombre: str
    series: int = 3
    reps:   int = 10
    peso_kg: float = 0.0


class RegistroRutinaManualRequest(BaseModel):
    ejercicios: list


class ConfirmarRegistroRequest(BaseModel):
    consulta_id: str


class GuardarSugerenciaRequest(BaseModel):
    tipo: str           # 'comida' o 'ejercicio'
    nombre: str
    ingredientes: list = []
    preparacion: list = []
    macros: str = ""
    nota: str = ""


# ── helpers de memoria conversacional ──────────────────────────────────────

def _cargar_historial_bd(client_id: int, db: Session, limite: int = 8) -> list:
    """
    Devuelve los últimos `limite` mensajes de chat_historial como lista [{role, content}],
    SOLO si el último mensaje es de las últimas 2 horas (misma sesión activa).
    Si el usuario no ha hablado en más de 2 horas, devuelve [] para evitar
    que el LLM mezcle temas de sesiones anteriores.
    """
    from sqlalchemy import text as _t
    from datetime import datetime, timezone, timedelta

    # Verificar cuándo fue el último mensaje
    last_row = db.execute(_t(
        "SELECT created_at FROM chat_historial "
        "WHERE client_id = :cid ORDER BY created_at DESC LIMIT 1"
    ), {"cid": client_id}).fetchone()

    if not last_row:
        return []

    # Usar UTC para comparar (la BD guarda en UTC)
    last_ts = last_row.created_at
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    now_utc = datetime.now(timezone.utc)
    if (now_utc - last_ts) > timedelta(hours=2):
        return []  # Sesión expirada — no contaminar con contexto viejo

    rows = db.execute(_t(
        "SELECT rol, contenido FROM chat_historial "
        "WHERE client_id = :cid ORDER BY created_at DESC LIMIT :n"
    ), {"cid": client_id, "n": limite}).fetchall()
    return [{"role": r.rol, "content": r.contenido} for r in reversed(rows)]


def _guardar_turno_bd(client_id: int, mensaje_user: str, texto_asistente: str, db: Session):
    """Persiste un turno completo (user + assistant) en chat_historial."""
    from sqlalchemy import text as _t
    try:
        db.execute(_t(
            "INSERT INTO chat_historial (client_id, rol, contenido) VALUES (:cid, :rol, :cont)"
        ), {"cid": client_id, "rol": "user", "cont": mensaje_user[:2000]})
        db.execute(_t(
            "INSERT INTO chat_historial (client_id, rol, contenido) VALUES (:cid, :rol, :cont)"
        ), {"cid": client_id, "rol": "assistant", "cont": texto_asistente[:2000]})
        # Mantener solo los últimos 100 mensajes por usuario (50 turnos)
        db.execute(_t(
            "DELETE FROM chat_historial WHERE client_id = :cid AND id NOT IN ("
            "  SELECT id FROM chat_historial WHERE client_id = :cid "
            "  ORDER BY created_at DESC LIMIT 100)"
        ), {"cid": client_id})
        db.commit()
    except Exception:
        db.rollback()


@router.post("/consultar")
async def consultar_asistente(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Chat principal con memoria conversacional persistida en BD."""
    try:
        cliente = db.query(Client).filter(Client.email == current_user.email).first()

        # Solo historial de la sesión actual (últimos 6 mensajes enviados por Flutter).
        # El historial BD está desactivado — causaba que el LLM dijera "como recordarás..." y similares.
        historial_sesion = request.historial or []
        historial_combinado = historial_sesion

        resultado = await asistente_service.consultar(
            mensaje=request.mensaje,
            db=db,
            current_user=current_user,
            historial=historial_combinado,
            contexto_manual=request.contexto_manual,
            override_ia=request.override_ia,
            consulta_id=request.consulta_id,
        )

        # Persistir el nuevo turno en BD (no guardar turnos bloqueados por guardia)
        if cliente and not resultado.get("_blocked"):
            texto_resp = (
                resultado.get("respuesta_estructurada", {}).get("texto_conversacional", "")
                or resultado.get("respuesta", "")
            )
            if texto_resp:
                _guardar_turno_bd(cliente.id, request.mensaje, texto_resp, db)

        return resultado
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ ERROR EN /consultar: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/historial")
async def obtener_historial_chat(
    limite: int = 30,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Devuelve los últimos mensajes del chat para restaurar conversación entre sesiones."""
    from sqlalchemy import text as _t
    cliente = db.query(Client).filter(Client.email == current_user.email).first()
    if not cliente:
        return []
    rows = db.execute(_t(
        "SELECT rol, contenido, created_at FROM chat_historial "
        "WHERE client_id = :cid ORDER BY created_at DESC LIMIT :n"
    ), {"cid": cliente.id, "n": min(limite, 60)}).fetchall()
    return [
        {
            "role": r.rol,
            "content": r.contenido,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in reversed(rows)
    ]


# /log-inteligente (NLP de comida/ejercicio por voz, arquitectura vieja) se
# eliminó: el frontend migró a "todo va por /consultar" (ver comentario en
# chat_screen.dart) y registrarPorVoz() quedó sin ningún caller. Confirmado
# sin referencias en todo lib/ antes de borrar.


@router.post("/log-manual")
async def registro_manual_alimento(
    body: RegistroManualAlimentoRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Registro manual (cuando el usuario tiene la etiqueta a mano):
    - Guarda/actualiza el alimento en BD con categoría
    - Opcionalmente registra la unidad (botella/vaso/etc) en alimento_unidades
    - Registra el consumo al progreso del día (como /log-inteligente)
    """
    try:
        return await asistente_service.registrar_manual_alimento(
            body=body.model_dump(),
            db=db,
            current_user=current_user,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"ERROR EN /log-manual: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class AlimentoDirectoItem(BaseModel):
    nombre: str
    gramos: float
    kcal: float
    proteinas_g: float
    carbohidratos_g: float
    grasas_g: float


class RegistroDirectoRequest(BaseModel):
    alimentos: list[AlimentoDirectoItem]
    texto_original: str = ""


@router.post("/registrar-directo")
async def registrar_macros_directos(
    body: RegistroDirectoRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Registro directo con macros pre-calculados desde el Registro Inteligente.
    NO re-estima — usa exactamente los valores calculados en el preview.
    Garantiza consistencia entre lo que el usuario vio y lo que se guarda.
    """
    from app.models.client import Client
    from app.models.historial import ProgresoCalorias
    from app.models.comida_registro import ComidaRegistro
    from app.core.utils import get_peru_date
    from sqlalchemy import func

    perfil = db.query(Client).filter(Client.email.ilike(current_user.email)).first()
    if not perfil:
        raise HTTPException(status_code=404, detail="Perfil no encontrado")

    hoy = get_peru_date()
    kcal_total  = round(sum(a.kcal          for a in body.alimentos), 1)
    prot_total  = round(sum(a.proteinas_g   for a in body.alimentos), 1)
    carb_total  = round(sum(a.carbohidratos_g for a in body.alimentos), 1)
    grasa_total = round(sum(a.grasas_g      for a in body.alimentos), 1)

    try:
        # 1. Actualizar progreso_calorias
        prog = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id,
            ProgresoCalorias.fecha == hoy,
        ).first()
        if prog:
            prog.calorias_consumidas      = int((prog.calorias_consumidas or 0)      + kcal_total)
            prog.proteinas_consumidas     = round((prog.proteinas_consumidas or 0)   + prot_total, 1)
            prog.carbohidratos_consumidos = round((prog.carbohidratos_consumidos or 0) + carb_total, 1)
            prog.grasas_consumidas        = round((prog.grasas_consumidas or 0)      + grasa_total, 1)
        # Si no hay progreso aún no pasa nada — el balance reflejará cuando haga consulta

        # 2. Insertar cada ingrediente en comida_registros
        for item in body.alimentos:
            db.add(ComidaRegistro(
                client_id=perfil.id,
                fecha=hoy,
                nombre_alimento=item.nombre,
                kcal=item.kcal,
                proteina_g=item.proteinas_g,
                carbohidratos_g=item.carbohidratos_g,
                grasas_g=item.grasas_g,
                tipo_resolucion="manual_exacto",
                confianza=1.0,
                texto_original=body.texto_original[:490] if body.texto_original else "",
            ))

        db.commit()

        return {
            "success": True,
            "mensaje": f"✅ Registré {len(body.alimentos)} ingrediente(s) — {round(kcal_total)} kcal totales.",
            "datos": {
                "nombre": " + ".join(a.nombre for a in body.alimentos[:3]) + (f" y {len(body.alimentos)-3} más" if len(body.alimentos) > 3 else ""),
                "alimentos_lista": [a.nombre for a in body.alimentos],
                "calorias": kcal_total,
                "proteinas_g": prot_total,
                "carbohidratos_g": carb_total,
                "grasas_g": grasa_total,
            },
            "balance_actualizado": {
                "consumido": round(float(prog.calorias_consumidas if prog else kcal_total), 1),
                "quemado": round(float(prog.calorias_quemadas if prog else 0), 1),
            },
        }
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calcular-ejercicio")
async def calcular_ejercicio(
    body: CalcularEjercicioRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        return await asistente_service.calcular_ejercicio_manual(
            nombre=body.nombre,
            series=body.series,
            reps=body.reps,
            peso_kg=body.peso_kg,
            db=db,
            current_user=current_user,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"ERROR EN /calcular-ejercicio: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/log-rutina-manual")
async def log_rutina_manual(
    body: RegistroRutinaManualRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    try:
        return await asistente_service.registrar_rutina_manual(
            ejercicios=body.ejercicios,
            db=db,
            current_user=current_user,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"ERROR EN /log-rutina-manual: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/confirmar-registro")
async def confirmar_registro_con_consulta_id(
    body: ConfirmarRegistroRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Registra comida usando valores exactos de una tarjeta del chat (consulta_id).
    Evita inconsistencia: mismo valor que vio el usuario, sin recalcular.
    """
    try:
        resultado = await asistente_service.confirmar_registro(
            consulta_id=body.consulta_id,
            db=db,
            current_user=current_user,
        )
        return resultado
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ ERROR EN /confirmar-registro: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ═══ PART C: Guardar Sugerencias (Recetario Personal) ═══

@router.post("/guardar-sugerencia")
async def guardar_sugerencia(
    body: GuardarSugerenciaRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Guarda una receta/rutina sugerida por la IA para prepararla después."""
    import re as _re
    from app.models.preferencias import PreferenciaAlimento

    try:
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise HTTPException(status_code=404, detail="Perfil no encontrado")

        nueva = SugerenciaGuardada(
            client_id=perfil.id,
            tipo=body.tipo,
            nombre=body.nombre,
            ingredientes=body.ingredientes,
            preparacion=body.preparacion,
            macros=body.macros,
            nota=body.nota,
        )
        db.add(nueva)
        db.commit()
        db.refresh(nueva)

        # ── Sincronizar con Favoritos cuando es una comida ────────────────
        if body.tipo == "comida" and body.nombre:
            nombre_lower = body.nombre.lower().strip()

            # Parsear macros del string "Cal: 252.9kcal | P: 26.4g | C: 13.5g | G: 13.3g"
            def _parse_macro(pattern: str, text: str) -> float:
                m = _re.search(pattern, text or "")
                return float(m.group(1)) if m else 0.0

            _cal  = _parse_macro(r"Cal:\s*([\d.]+)", body.macros)
            _prot = _parse_macro(r"P:\s*([\d.]+)",   body.macros)
            _carb = _parse_macro(r"C:\s*([\d.]+)",   body.macros)
            _gras = _parse_macro(r"G:\s*([\d.]+)",   body.macros)

            pref = db.query(PreferenciaAlimento).filter(
                PreferenciaAlimento.client_id == perfil.id,
                PreferenciaAlimento.alimento   == nombre_lower,
            ).first()

            if pref:
                pref.es_favorito    = 1
                pref.frecuencia     = (pref.frecuencia or 0) + 1
                pref.calorias       = _cal  if _cal  > 0 else (pref.calorias or 0)
                pref.proteinas      = _prot if _prot > 0 else (pref.proteinas or 0)
                pref.carbohidratos  = _carb if _carb > 0 else (pref.carbohidratos or 0)
                pref.grasas         = _gras if _gras > 0 else (pref.grasas or 0)
            else:
                pref = PreferenciaAlimento(
                    client_id      = perfil.id,
                    alimento       = nombre_lower,
                    es_favorito    = 1,
                    frecuencia     = 1,
                    puntuacion     = 1.0,
                    calorias       = _cal,
                    proteinas      = _prot,
                    carbohidratos  = _carb,
                    grasas         = _gras,
                )
                db.add(pref)

            db.commit()

        return {"mensaje": f"🔖 '{body.nombre}' guardado en tu recetario", "id": nueva.id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mis-sugerencias")
async def listar_sugerencias(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Lista las sugerencias guardadas del usuario."""
    try:
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise HTTPException(status_code=404, detail="Perfil no encontrado")

        items = db.query(SugerenciaGuardada).filter(
            SugerenciaGuardada.client_id == perfil.id
        ).order_by(SugerenciaGuardada.fecha_guardado.desc()).all()

        return [{
            "id": s.id,
            "tipo": s.tipo,
            "nombre": s.nombre,
            "ingredientes": s.ingredientes or [],
            "preparacion": s.preparacion or [],
            "macros": s.macros or "",
            "nota": s.nota or "",
            "completada": s.completada,
            "fecha_guardado": str(s.fecha_guardado),
        } for s in items]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/sugerencia/{sugerencia_id}/completar")
async def completar_sugerencia(
    sugerencia_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Marca una sugerencia como completada (ya la preparó/hizo)."""
    try:
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        item = db.query(SugerenciaGuardada).filter(
            SugerenciaGuardada.id == sugerencia_id,
            SugerenciaGuardada.client_id == perfil.id,
        ).first()
        if not item:
            raise HTTPException(status_code=404, detail="Sugerencia no encontrada")
        item.completada = True
        db.commit()
        return {"mensaje": f"✅ '{item.nombre}' marcada como completada"}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sugerencia/{sugerencia_id}")
async def eliminar_sugerencia(
    sugerencia_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Elimina una sugerencia guardada."""
    try:
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        item = db.query(SugerenciaGuardada).filter(
            SugerenciaGuardada.id == sugerencia_id,
            SugerenciaGuardada.client_id == perfil.id,
        ).first()
        if not item:
            raise HTTPException(status_code=404, detail="Sugerencia no encontrada")
        db.delete(item)
        db.commit()
        return {"mensaje": f"🗑️ Sugerencia eliminada"}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
