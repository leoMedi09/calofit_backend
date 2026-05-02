"""
Rutas del Asistente del Cliente — "CaloFit Coach Personal"

Endpoints:
  POST /consultar          → Consulta de chat (nutrición, recetas, rutinas, progreso)
  POST /log-inteligente    → Registro de comida/ejercicio por texto o voz
  POST /confirmar-registro → Confirmar registro desde tarjeta interactiva (consulta_id)

Toda la lógica de negocio está delegada a:
  app.services.asistente_service.AsistenteService
"""

import traceback
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.services.asistente_service import asistente_service
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


class ConfirmarRegistroRequest(BaseModel):
    consulta_id: str


class GuardarSugerenciaRequest(BaseModel):
    tipo: str           # 'comida' o 'ejercicio'
    nombre: str
    ingredientes: list = []
    preparacion: list = []
    macros: str = ""
    nota: str = ""


@router.post("/consultar")
async def consultar_asistente(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Chat principal del Asistente CaloFit para clientes.
    Responde consultas de nutrición, recetas, rutinas y progreso diario.
    """
    try:
        resultado = await asistente_service.consultar(
            mensaje=request.mensaje,
            db=db,
            current_user=current_user,
            historial=request.historial,
            contexto_manual=request.contexto_manual,
            override_ia=request.override_ia,
            consulta_id=request.consulta_id,
        )
        return resultado
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ ERROR EN /consultar: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/log-inteligente")
async def registro_inteligente_nlp(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """
    Registrar comida o ejercicio por texto/voz (NLP).
    Si envías consulta_id desde una card, se usan los mismos valores mostrados.
    """
    try:
        resultado = await asistente_service.registrar_por_nlp(
            mensaje=request.mensaje,
            db=db,
            current_user=current_user,
            consulta_id=request.consulta_id,
        )
        return resultado
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ ERROR EN /log-inteligente: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
