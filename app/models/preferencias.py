from sqlalchemy import Column, Integer, String, Float, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class PreferenciaAlimento(Base):
    """
    Almacena las preferencias de alimentos de cada cliente
    basadas en su historial de consumo.
    """
    __tablename__ = "preferencias_alimentos"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    alimento = Column(String(200), nullable=False)
    frecuencia = Column(Integer, default=1)
    puntuacion = Column(Float, default=1.0)
    ultima_vez = Column(TIMESTAMP, server_default=func.now())
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relación con cliente
    cliente = relationship("Client", back_populates="preferencias_alimentos")


class PreferenciaEjercicio(Base):
    """
    Almacena las preferencias de ejercicios de cada cliente
    basadas en su historial de entrenamiento.
    """
    __tablename__ = "preferencias_ejercicios"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    ejercicio = Column(String(200), nullable=False)
    frecuencia = Column(Integer, default=1)
    puntuacion = Column(Float, default=1.0)
    ultima_vez = Column(TIMESTAMP, server_default=func.now())
    created_at = Column(TIMESTAMP, server_default=func.now())

    # Relación con cliente
    cliente = relationship("Client", back_populates="preferencias_ejercicios")
