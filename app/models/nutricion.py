from sqlalchemy import Column, Integer, Float, ForeignKey, DateTime, String, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class PlanNutricional(Base):
    __tablename__ = "planes_nutricionales"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    nutricionista_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Puede ser NULL si es generado por IA
    
    # --- Datos Biométricos ---
    genero = Column(Integer, nullable=False)
    edad = Column(Integer, nullable=False)
    peso = Column(Float, nullable=False)
    talla = Column(Float, nullable=False)
    nivel_actividad = Column(Float, nullable=False) 
    objetivo = Column(String, nullable=False) 
    
    # --- Control de Contingencia ---
    # Marcamos si este plan fue una extensión automática de la IA por falta de cita
    es_contingencia_ia = Column(Boolean, default=False) 
    calorias_ia_base = Column(Float, nullable=True) 
    
    fecha_creacion = Column(DateTime, default=datetime.utcnow)
    observaciones = Column(String, nullable=True)
    
    # --- Estados de Validación (Flujo Gym Real) ---
    # draft_ia: Generado por IA, pendiente revisión
    # validado: Revisado y aprobado por nutricionista
    # archivado: Plan antiguo
    status = Column(String, default="draft_ia")
    validated_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    validated_at = Column(DateTime, nullable=True)

    cliente = relationship("Client", back_populates="planes_nutricionales")
    detalles_diarios = relationship("PlanDiario", back_populates="plan_maestro", cascade="all, delete-orphan")

class PlanDiario(Base):
    __tablename__ = "planes_diarios"

    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("planes_nutricionales.id"))
    
    dia_numero = Column(Integer, nullable=False) # 1 al 7
    calorias_dia = Column(Float, nullable=False)
    proteinas_g = Column(Float, nullable=False)
    carbohidratos_g = Column(Float, nullable=False)
    grasas_g = Column(Float, nullable=False)
    
    # --- Campos de Soporte al Coach y Cliente ---
    # Si el Coach está full, el cliente lee esto:
    sugerencia_entrenamiento_ia = Column(String, nullable=True) 
    # Si el Nutri no llegó a la cita, la IA explica el ajuste aquí:
    nota_asistente_ia = Column(String, nullable=True) 

    # --- Estados ---
    validado_nutri = Column(Boolean, default=False)
    # Cambiamos 'pendiente' a 'sugerencia_ia' por defecto para disponibilidad inmediata
    estado = Column(String, default="sugerencia_ia") 

    plan_maestro = relationship("PlanNutricional", back_populates="detalles_diarios")