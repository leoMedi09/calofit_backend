"""
Modelos SQLAlchemy para rutinas de ejercicio.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text,
    DateTime, ForeignKey, Index,
)
from sqlalchemy.orm import relationship
from app.core.database import Base


class Rutina(Base):
    """
    Catálogo de rutinas de entrenamiento (generadas por LLM o manuales).
    """
    __tablename__ = "rutinas"

    id             = Column(Integer, primary_key=True, index=True)
    nombre         = Column(String(255), nullable=False)
    descripcion    = Column(Text, nullable=True)
    perfil_tipo    = Column(String(16), nullable=True, index=True)   # A|B|C
    nivel          = Column(String(32), nullable=True)               # Principiante|Intermedio|Avanzado
    grupo_muscular = Column(String(128), nullable=True, index=True)
    tiempo_min     = Column(Integer, nullable=True)
    series_config  = Column(Text, nullable=True)   # JSON con configuración de series por perfil
    origen         = Column(String(32), default="llm", nullable=False)  # manual|llm
    created_at     = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at     = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    ejercicios = relationship(
        "RutinaEjercicio",
        back_populates="rutina",
        cascade="all, delete-orphan",
    )
    sesiones = relationship(
        "WorkoutSession",
        back_populates="rutina",
    )

    def __repr__(self):
        return f"<Rutina(nombre={self.nombre!r}, perfil={self.perfil_tipo!r}, grupo={self.grupo_muscular!r})>"


class RutinaEjercicio(Base):
    """
    Tabla pivote Rutina ↔ Ejercicio.
    Define series, reps, descanso y orden de cada ejercicio en la rutina.
    """
    __tablename__ = "rutinas_ejercicios"

    id             = Column(Integer, primary_key=True, index=True)
    rutina_id      = Column(Integer, ForeignKey("rutinas.id",   ondelete="CASCADE"),  nullable=False, index=True)
    ejercicio_id   = Column(Integer, ForeignKey("ejercicios.id", ondelete="CASCADE"), nullable=False, index=True)
    orden          = Column(Integer, default=1, nullable=False)
    series         = Column(Integer, default=3, nullable=False)
    reps           = Column(Integer, default=12, nullable=False)
    descanso_s     = Column(Integer, default=60, nullable=False)
    peso_sugerido_kg = Column(Float, nullable=True)
    notas          = Column(Text, nullable=True)

    # Relationships
    rutina    = relationship("Rutina",    back_populates="ejercicios")
    ejercicio = relationship("Ejercicio", foreign_keys=[ejercicio_id])

    __table_args__ = (
        Index("idx_rut_ej_orden", "rutina_id", "orden"),
    )

    def __repr__(self):
        return (
            f"<RutinaEjercicio(rutina_id={self.rutina_id}, "
            f"ejercicio_id={self.ejercicio_id!r}, orden={self.orden})>"
        )
