"""
Modelo SQLAlchemy alineado con Postgres (BD_Calofit).

NOTA: Estas columnas están expresadas por 100g (no por porción).
"""
from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.core.database import Base


class Alimento(Base):
    __tablename__ = "alimentos"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(255), nullable=False, unique=True, index=True)
    nombre_normalizado = Column(String(255), nullable=False, index=True)

    calorias_100g = Column(Float, nullable=False)
    proteina_100g = Column(Float, nullable=False)
    carbohidratos_100g = Column(Float, nullable=False)
    grasas_100g = Column(Float, nullable=False)

    fibra_100g = Column(Float, nullable=True)
    azucar_100g = Column(Float, nullable=True)

    categoria = Column(String(100), nullable=True)
    fuente = Column(String(255), nullable=True)
    id_externo = Column(String(100), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
