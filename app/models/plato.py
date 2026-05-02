from __future__ import annotations

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer,
    JSON, String, Text, CheckConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class Plato(Base):
    """
    Catálogo compartido de platos/recetas.

    Los macros NO se almacenan aquí — se calculan siempre en tiempo real
    sumando los valores de plato_ingredientes × alimentos:
        kcal = Σ (alimento.calorias_100g × ingrediente.gramos / 100)

    Esto garantiza consistencia absoluta: una sola fuente de verdad (alimentos).
    """

    __tablename__ = "platos"

    id                 = Column(Integer, primary_key=True, index=True)
    nombre             = Column(String(255), nullable=False)
    nombre_normalizado = Column(String(255), nullable=False, unique=True, index=True)

    tipo_plato   = Column(String(50), default="cualquiera")  # desayuno/almuerzo/cena/snack/cualquiera
    preparacion  = Column(JSON, nullable=True)
    nota         = Column(Text, nullable=True)
    origen       = Column(String(50), default="manual")      # 'manual', 'llm', 'usuario'

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    ingredientes = relationship(
        "PlatoIngrediente",
        back_populates="plato",
        cascade="all, delete-orphan",
        order_by="PlatoIngrediente.orden",
    )

    def calcular_macros(self) -> dict:
        """Devuelve macros calculados desde los ingredientes reales (no estimaciones LLM)."""
        cal = prot = carb = gras = 0.0
        for ing in self.ingredientes:
            if ing.alimento:
                f = ing.gramos / 100.0
                cal  += ing.alimento.calorias_100g     * f
                prot += ing.alimento.proteina_100g     * f
                carb += ing.alimento.carbohidratos_100g * f
                gras += ing.alimento.grasas_100g       * f
        return {
            "calorias":         round(cal,  1),
            "proteinas_g":      round(prot, 1),
            "carbohidratos_g":  round(carb, 1),
            "grasas_g":         round(gras, 1),
        }


class PlatoIngrediente(Base):
    """
    Ingrediente de un plato vinculado a un alimento real.
    Columna central: gramos — cantidad del ingrediente en la receta estándar.
    """

    __tablename__ = "plato_ingredientes"
    __table_args__ = (
        CheckConstraint("gramos > 0", name="ck_plato_ing_gramos_positivo"),
    )

    id          = Column(Integer, primary_key=True, index=True)
    plato_id    = Column(Integer, ForeignKey("platos.id",    ondelete="CASCADE"),   nullable=False, index=True)
    alimento_id = Column(Integer, ForeignKey("alimentos.id", ondelete="RESTRICT"),  nullable=False)
    gramos      = Column(Float,   nullable=False)
    orden       = Column(Integer, nullable=False, default=0)
    notas       = Column(String(255), nullable=True)

    plato    = relationship("Plato",    back_populates="ingredientes")
    alimento = relationship("Alimento")
