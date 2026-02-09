from sqlalchemy import Column, Integer, String, Text, TIMESTAMP, ForeignKey
from sqlalchemy.sql import func
from app.core.database import Base

class AuditoriaAdmin(Base):
    """
    Tabla para registrar eventos administrativos (logs).
    Ejemplo: Creación de personal, cambio de contraseñas, reseteo de accesos.
    """
    __tablename__ = "auditoria_admin"

    id = Column(Integer, primary_key=True, index=True)
    admin_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    accion = Column(String(100), nullable=False)  # 'REGISTRO_STAFF', 'CAMBIO_PASSWORD', etc.
    descripcion = Column(Text, nullable=False)
    tabla_afectada = Column(String(50), nullable=True)
    registro_id = Column(Integer, nullable=True) # ID del registro afectado (ej: id del staff)
    
    fecha_evento = Column(TIMESTAMP, nullable=False, server_default=func.now())
    ip_origen = Column(String(45), nullable=True)

    def __repr__(self):
        return f"<AuditoriaAdmin(accion={self.accion}, fecha={self.fecha_evento})>"
