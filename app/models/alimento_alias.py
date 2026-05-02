from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Integer, String

from app.core.database import Base


class AlimentoAlias(Base):
    __tablename__ = "alimento_alias"

    id = Column(Integer, primary_key=True, index=True)
    alimento_id = Column(Integer, ForeignKey("alimentos.id", ondelete="CASCADE"), nullable=False)
    alias = Column(String(255), nullable=False)
    alias_normalizado = Column(String(255), nullable=False, index=True)

