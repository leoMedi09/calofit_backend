"""
Registro auditado de cada evento de ingesta.

`comida_registros` es la fuente de verdad por evento.
`progreso_calorias` es derivado: recalcular_progreso_diario() lo mantiene sincronizado.
"""

from __future__ import annotations

from sqlalchemy import Column, Date, Float, ForeignKey, Integer, String, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class ComidaRegistro(Base):
    __tablename__ = "comida_registros"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False, index=True)
    fecha = Column(Date, nullable=False, index=True)

    nombre_alimento = Column(String(255), nullable=False)
    plato_id = Column(Integer, ForeignKey("platos.id"), nullable=True)
    alimento_id = Column(Integer, ForeignKey("alimentos.id"), nullable=True)
    gramos = Column(Float, nullable=True)

    kcal = Column(Float, nullable=False, default=0.0)
    proteina_g = Column(Float, nullable=False, default=0.0)
    carbohidratos_g = Column(Float, nullable=False, default=0.0)
    grasas_g = Column(Float, nullable=False, default=0.0)

    tipo_resolucion = Column(String(50), nullable=False, default="bd_alimento")
    confianza = Column(Float, nullable=False, default=1.0)
    texto_original = Column(String(500), nullable=True)
    momento = Column(String(20), nullable=True)

    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

    cliente = relationship("Client", back_populates="comida_registros")
    plato = relationship("Plato")
    alimento = relationship("Alimento")
