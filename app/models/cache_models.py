"""
Modelos SQLAlchemy para tablas de caché y alimentos no resueltos.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text,
    DateTime, ForeignKey, CheckConstraint, Index,
)
from sqlalchemy.orm import relationship
from app.core.database import Base


class AppCacheAlimentos(Base):
    """
    Caché de lookups exitosos USDA/FatSecret/BD para evitar consultar APIs repetidamente.
    Expira después de 60 días.
    """
    __tablename__ = "app_cache_alimentos"

    id             = Column(Integer, primary_key=True, index=True)
    food_normalized = Column(String(255), nullable=False, index=True)
    user_id        = Column(Integer, ForeignKey("clients.id", ondelete="SET NULL"), nullable=True, index=True)
    alimento_id    = Column(Integer, ForeignKey("alimentos.id", ondelete="CASCADE"), nullable=True)
    source         = Column(String(64), nullable=True)   # BD|USDA|FatSecret|Groq
    raw_response   = Column(Text, nullable=True)
    hit_count      = Column(Integer, default=1, nullable=False)
    expires_at     = Column(DateTime(timezone=True), nullable=True)
    created_at     = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    # Relationships
    alimento = relationship("Alimento", foreign_keys=[alimento_id])
    client   = relationship("Client",   foreign_keys=[user_id])

    __table_args__ = (
        Index("idx_cache_food",    "food_normalized"),
        Index("idx_cache_expires", "expires_at"),
    )

    def __repr__(self):
        return f"<AppCacheAlimentos(food={self.food_normalized!r}, source={self.source!r})>"


class AppCachePlatos(Base):
    """
    Caché de platos resueltos y validados con sus macros en tiempo real.
    """
    __tablename__ = "app_cache_platos"

    id               = Column(Integer, primary_key=True, index=True)
    plato_normalized = Column(String(255), nullable=False, index=True)
    user_id          = Column(Integer, ForeignKey("clients.id", ondelete="SET NULL"), nullable=True, index=True)
    plato_id         = Column(Integer, ForeignKey("platos.id", ondelete="CASCADE"), nullable=True)
    source           = Column(String(64), nullable=True)
    hit_count        = Column(Integer, default=1, nullable=False)
    expires_at       = Column(DateTime(timezone=True), nullable=True)
    created_at       = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    # Relationships
    plato  = relationship("Plato",   foreign_keys=[plato_id])
    client = relationship("Client",  foreign_keys=[user_id])

    __table_args__ = (
        Index("idx_cache_plato_norm",    "plato_normalized"),
        Index("idx_cache_plato_expires", "expires_at"),
    )

    def __repr__(self):
        return f"<AppCachePlatos(plato={self.plato_normalized!r}, plato_id={self.plato_id})>"


class AppCacheRutinas(Base):
    """
    Caché de rutinas generadas por el LLM para perfil/zonas/tiempo específicos.
    """
    __tablename__ = "app_cache_rutinas"

    id            = Column(Integer, primary_key=True, index=True)
    cache_key     = Column(String(512), nullable=False, unique=True, index=True)
    user_id       = Column(Integer, ForeignKey("clients.id", ondelete="CASCADE"), nullable=True, index=True)
    perfil_tipo   = Column(String(16), nullable=True)
    zonas_objetivo = Column(Text, nullable=True)
    tiempo_min    = Column(Integer, nullable=True)
    rutina_json   = Column(Text, nullable=False)
    hit_count     = Column(Integer, default=1, nullable=False)
    expires_at    = Column(DateTime(timezone=True), nullable=True)
    created_at    = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    # Relationships
    client = relationship("Client", foreign_keys=[user_id])

    __table_args__ = (
        Index("idx_cache_rutinas_expires", "expires_at"),
    )

    def __repr__(self):
        return f"<AppCacheRutinas(cache_key={self.cache_key!r}, perfil={self.perfil_tipo!r})>"


class AlimentoSinResolver(Base):
    """
    Registro de alimentos que ninguna capa pudo resolver.
    El nutricionista los revisa y asigna macros manualmente.
    """
    __tablename__ = "alimentos_sin_resolver"

    id                 = Column(Integer, primary_key=True, index=True)
    nombre_original    = Column(String(512), nullable=False)
    nombre_normalizado = Column(String(512), nullable=True, index=True)
    user_id            = Column(Integer, ForeignKey("clients.id",  ondelete="SET NULL"), nullable=True, index=True)
    reporter_id        = Column(Integer, ForeignKey("users.id",    ondelete="SET NULL"), nullable=True)
    mensaje_contexto   = Column(Text, nullable=True)
    intentos           = Column(Integer, default=1, nullable=False)
    estado             = Column(String(32), default="pendiente", nullable=False, index=True)
    notas              = Column(Text, nullable=True)
    fecha_reporte      = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    fecha_resolucion   = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    client    = relationship("Client", foreign_keys=[user_id])
    reporter  = relationship("User",   foreign_keys=[reporter_id])

    __table_args__ = (
        CheckConstraint(
            "estado IN ('pendiente', 'validado', 'rechazado')",
            name="ck_alimento_sin_resolver_estado",
        ),
    )

    def __repr__(self):
        return f"<AlimentoSinResolver(nombre={self.nombre_original!r}, estado={self.estado!r})>"
