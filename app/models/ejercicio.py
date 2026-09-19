"""
Modelo de ejercicios en PostgreSQL.
Incluye MET para cálculo de calorías quemadas (fórmula METs).
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.sql import func
from app.core.database import Base


class Ejercicio(Base):
    __tablename__ = "ejercicios"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(255), nullable=False, index=True)
    nombre_normalizado = Column(String(255), nullable=False, index=True)
    alias = Column(Text, nullable=True)
    descripcion = Column(Text, nullable=True)

    met = Column(Float, nullable=False)
    grupo_muscular = Column(String(128), nullable=True)
    origen = Column(String(64), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
