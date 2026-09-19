"""
Metas nutricionales y requerimientos por usuario (TMB, GET personalizados).
Fórmula Mifflin-St Jeor para TMB; GET = TMB * factor_actividad.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class MetaUsuario(Base):
    __tablename__ = "metas_usuario"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)

    genero = Column(String(1), nullable=False)
    edad = Column(Integer, nullable=False)
    peso_kg = Column(Float, nullable=False)
    talla_cm = Column(Float, nullable=False)
    nivel_actividad = Column(String(32), nullable=False)
    objetivo = Column(String(64), nullable=False)

    tmb = Column(Float, nullable=False)
    get = Column(Float, nullable=False)
    calorias_objetivo = Column(Float, nullable=False)
    proteinas_g = Column(Float, nullable=False)
    carbohidratos_g = Column(Float, nullable=False)
    grasas_g = Column(Float, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
