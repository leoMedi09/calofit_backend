from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Float, Integer, String

from app.core.database import Base


class AlimentoUnidad(Base):
    __tablename__ = "alimento_unidades"

    id = Column(Integer, primary_key=True, index=True)
    alimento_id = Column(Integer, ForeignKey("alimentos.id", ondelete="CASCADE"), nullable=False)
    nombre = Column(String(100), nullable=False)
    gramos = Column(Float, nullable=False)

