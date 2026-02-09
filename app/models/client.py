from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, DateTime, Date
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy.dialects.postgresql import ARRAY
from app.core.database import Base

class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name_paternal = Column(String, nullable=False)
    last_name_maternal = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    planes_nutricionales = relationship("PlanNutricional", back_populates="cliente")
    
    flutter_uid = Column(String, unique=True, nullable=False, index=True)  # ✅ UID de Firebase/Flutter para vincular usuario con perfil de salud
    
    birth_date = Column(Date, nullable=True)  # Fecha de nacimiento
    weight = Column(Float)  # Peso actual (kg)
    height = Column(Float)  # Altura (cm) - ESTANDARIZADA A CENTÍMETROS
    gender = Column(String(1), default='M', nullable=False)  # 'M' (Hombre) o 'F' (Mujer) - NECESARIO para fórmula Harris-Benedict
    medical_conditions = Column(ARRAY(String), nullable=True, default=[])
    activity_level = Column(String, nullable=True, default='Moderado')  # Nivel de actividad física: Sedentario, Ligero, Moderado, Intenso, Muy intenso
    goal = Column(String, nullable=True, default='Mantener peso')  # Objetivo principal: Perder peso, Mantener peso, Ganar masa
    
    assigned_coach_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    assigned_nutri_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    
    coach = relationship("User", foreign_keys="[Client.assigned_coach_id]", back_populates="clients_as_coach")
    nutritionist = relationship("User", foreign_keys="[Client.assigned_nutri_id]", back_populates="clients_as_nutri")

    # Nuevas relaciones para historial
    historial_peso = relationship("HistorialPeso", back_populates="cliente", cascade="all, delete-orphan")
    historial_imc = relationship("HistorialIMC", back_populates="cliente", cascade="all, delete-orphan")
    progreso_calorias = relationship("ProgresoCalorias", back_populates="cliente", cascade="all, delete-orphan")
    alertas_salud = relationship("AlertaSalud", back_populates="cliente", cascade="all, delete-orphan")
    
    # Relaciones para sistema de aprendizaje de preferencias
    preferencias_alimentos = relationship("PreferenciaAlimento", back_populates="cliente", cascade="all, delete-orphan")
    preferencias_ejercicios = relationship("PreferenciaEjercicio", back_populates="cliente", cascade="all, delete-orphan")

    verification_code = Column(String(6), nullable=True)
    code_expires_at = Column(DateTime, nullable=True)