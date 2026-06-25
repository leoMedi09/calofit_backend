"""
Modelos SQLAlchemy para sesiones de entrenamiento estructuradas.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text,
    DateTime, Date, ForeignKey, Index,
)
from sqlalchemy.orm import relationship
from app.core.database import Base


class WorkoutSession(Base):
    """
    Sesión de entrenamiento completa de un cliente.
    Puede estar asociada a una Rutina o ser libre.
    """
    __tablename__ = "workout_sessions"

    id                   = Column(Integer, primary_key=True, index=True)
    client_id            = Column(Integer, ForeignKey("clients.id", ondelete="CASCADE"),  nullable=False, index=True)
    rutina_id            = Column(Integer, ForeignKey("rutinas.id", ondelete="SET NULL"),  nullable=True,  index=True)
    nombre_rutina        = Column(String(255), nullable=True)
    fecha                = Column(Date, nullable=False, index=True)
    duracion_min         = Column(Integer, nullable=True)
    calorias_quemadas    = Column(Float, nullable=True)
    intensity            = Column(String(16), nullable=True)  # Alta|Media|Baja
    notas                = Column(Text, nullable=True)
    created_at           = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    # Relationships
    client    = relationship("Client", foreign_keys=[client_id])
    rutina    = relationship("Rutina",  back_populates="sesiones", foreign_keys=[rutina_id])
    ejercicios = relationship(
        "WorkoutSessionEjercicio",
        back_populates="sesion",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_ws_client_fecha", "client_id", "fecha"),
    )

    def __repr__(self):
        return (
            f"<WorkoutSession(client_id={self.client_id}, "
            f"fecha={self.fecha}, duracion={self.duracion_min}min)>"
        )


class WorkoutSessionEjercicio(Base):
    """
    Ejercicio ejecutado dentro de una sesión.
    Registra lo que el usuario realmente realizó (series, reps, peso).
    """
    __tablename__ = "workout_session_ejercicios"

    id                  = Column(Integer, primary_key=True, index=True)
    session_id          = Column(Integer, ForeignKey("workout_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    ejercicio_id        = Column(Integer, ForeignKey("ejercicios.id",   ondelete="CASCADE"), nullable=False, index=True)
    orden               = Column(Integer, default=1, nullable=False)
    series_completadas  = Column(Integer, nullable=True)
    reps_completadas    = Column(Integer, nullable=True)
    peso_kg             = Column(Float, nullable=True)
    duracion_s          = Column(Integer, nullable=True)
    calorias_quemadas   = Column(Float, nullable=True)  # MET×peso×3.5/200×(duracion_s/60)
    notas               = Column(Text, nullable=True)

    # Relationships
    sesion    = relationship("WorkoutSession", back_populates="ejercicios")
    ejercicio = relationship("Ejercicio", foreign_keys=[ejercicio_id])

    __table_args__ = (
        Index("idx_wse_session_orden", "session_id", "orden"),
    )

    def __repr__(self):
        return (
            f"<WorkoutSessionEjercicio(session_id={self.session_id}, "
            f"ejercicio_id={self.ejercicio_id!r}, kcal={self.calorias_quemadas})>"
        )


class WorkoutLog(Base):
    """
    Registro histórico simple de un ejercicio realizado (NLP sync / manual sync).
    """
    __tablename__ = "workout_logs"

    id                   = Column(Integer, primary_key=True, index=True)
    client_id            = Column(Integer, ForeignKey("clients.id", ondelete="CASCADE"), nullable=False, index=True)
    ejercicio            = Column(String(255), nullable=False)
    series               = Column(Integer, nullable=False)
    reps                 = Column(Integer, nullable=False)
    peso_kg              = Column(Float, nullable=True)
    created_at           = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    calorias_quemadas    = Column(Float, nullable=True)
    session_duration_min = Column(Float, nullable=True)
    intensity            = Column(String(50), nullable=True)

    # Relationships
    client = relationship("Client", foreign_keys=[client_id])

    def __repr__(self):
        return (
            f"<WorkoutLog(client_id={self.client_id}, "
            f"ejercicio={self.ejercicio!r}, series={self.series}, reps={self.reps})>"
        )

