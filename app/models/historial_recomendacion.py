from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class HistorialRecomendacion(Base):
    """
    Historial de platos recomendados por usuario.

    Relaciona: cliente → plato (catálogo) + snapshot de macros al momento
    de la recomendación + si el usuario lo consumió.

    El campo plato_id puede ser NULL si el plato fue libre (no está en el catálogo).
    Los campos calorias/proteinas_g/etc. son un snapshot inmutable del momento
    de la recomendación (aunque los valores del plato se actualicen después).
    """

    __tablename__ = "historial_recomendaciones"

    id              = Column(Integer, primary_key=True, index=True)
    client_id       = Column(Integer, nullable=False, index=True)
    plato_id        = Column(Integer, ForeignKey("platos.id", ondelete="SET NULL"), nullable=True, index=True)

    nombre_plato    = Column(String(255))       # snapshot del nombre al recomendar
    calorias        = Column(Float)
    proteinas_g     = Column(Float)
    carbohidratos_g = Column(Float)
    grasas_g        = Column(Float)

    momento_dia     = Column(String(30))        # 'desayuno','almuerzo','cena','snack'
    fue_consumido   = Column(Boolean, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    plato = relationship("Plato", foreign_keys=[plato_id])
