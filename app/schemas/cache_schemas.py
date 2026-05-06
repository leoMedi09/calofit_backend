"""
Schemas Pydantic para caché de alimentos, platos, rutinas y alimentos sin resolver.
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ─── AppCacheAlimentos ────────────────────────────────────────────────────────

class AppCacheAlimentosCreate(BaseModel):
    food_normalized: str
    user_id:         Optional[int] = None
    alimento_id:     Optional[int] = None
    source:          Optional[str] = None
    raw_response:    Optional[str] = None
    expires_at:      Optional[datetime] = None


class AppCacheAlimentosResponse(AppCacheAlimentosCreate):
    id:         int
    hit_count:  int = 1
    created_at: datetime

    class Config:
        from_attributes = True


# ─── AppCachePlatos ───────────────────────────────────────────────────────────

class AppCachePlatosCreate(BaseModel):
    plato_normalized: str
    user_id:          Optional[int] = None
    plato_id:         Optional[int] = None
    source:           Optional[str] = None
    expires_at:       Optional[datetime] = None


class AppCachePlatosResponse(AppCachePlatosCreate):
    id:         int
    hit_count:  int = 1
    created_at: datetime

    class Config:
        from_attributes = True


# ─── AppCacheRutinas ──────────────────────────────────────────────────────────

class AppCacheRutinasCreate(BaseModel):
    cache_key:      str
    user_id:        Optional[int] = None
    perfil_tipo:    Optional[str] = None
    zonas_objetivo: Optional[str] = None
    tiempo_min:     Optional[int] = None
    rutina_json:    str
    expires_at:     Optional[datetime] = None


class AppCacheRutinasResponse(AppCacheRutinasCreate):
    id:         int
    hit_count:  int = 1
    created_at: datetime

    class Config:
        from_attributes = True


# ─── AlimentoSinResolver ─────────────────────────────────────────────────────

class AlimentoSinResolverCreate(BaseModel):
    nombre_original:    str
    nombre_normalizado: Optional[str] = None
    user_id:            Optional[int] = None
    reporter_id:        Optional[int] = None
    mensaje_contexto:   Optional[str] = None


class AlimentoSinResolverValidar(BaseModel):
    estado:           str = "validado"
    notas:            Optional[str] = None
    fecha_resolucion: Optional[datetime] = None


class AlimentoSinResolverResponse(AlimentoSinResolverCreate):
    id:               int
    intentos:         int = 1
    estado:           str
    notas:            Optional[str] = None
    fecha_reporte:    datetime
    fecha_resolucion: Optional[datetime] = None

    class Config:
        from_attributes = True
