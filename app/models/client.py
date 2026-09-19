from sqlalchemy import Column, Integer, String, Float, ForeignKey, Text, DateTime, Date, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from sqlalchemy.dialects.postgresql import ARRAY
from app.core.database import Base


class Client(Base):
    __tablename__ = "clients"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=True)
    last_name_paternal = Column(String, nullable=True)
    last_name_maternal = Column(String, nullable=True)
    dni = Column(String, unique=True, index=True, nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    planes_nutricionales = relationship(
        "PlanNutricional",
        back_populates="cliente",
        cascade="all, delete-orphan",
    )

    flutter_uid = Column(String, unique=True, nullable=True, index=True)
    fcm_token = Column(String, nullable=True)
    notificaciones_activas = Column(Boolean, nullable=True, default=True)

    birth_date = Column(Date, nullable=True)
    weight = Column(Float)
    height = Column(Float)
    gender = Column(String(1), default="M", nullable=False)
    medical_conditions = Column(ARRAY(String), nullable=True, default=[])
    activity_level = Column(String, nullable=True, default="Moderado")
    goal = Column(String, nullable=True, default="Mantener peso")
    workout_type = Column(String, nullable=True, default="Cardio")
    session_duration = Column(Float, nullable=True, default=1.0)

    assigned_coach_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    assigned_nutri_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    ai_strategic_focus = Column(String, nullable=True)
    recommended_foods = Column(ARRAY(String), nullable=True, default=[])
    forbidden_foods = Column(ARRAY(String), nullable=True, default=[])
    nutri_weekly_note = Column(Text, nullable=True)
    coach_notes = Column(Text, nullable=True)
    is_strategic_guide_validated = Column(Boolean, default=False)
    profile_picture_url = Column(String, nullable=True)
    is_profile_complete = Column(Boolean, default=False)
    terms_accepted_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    coach = relationship("User", foreign_keys="[Client.assigned_coach_id]", back_populates="clients_as_coach")
    nutritionist = relationship("User", foreign_keys="[Client.assigned_nutri_id]", back_populates="clients_as_nutri")

    historial_peso = relationship("HistorialPeso", back_populates="cliente", cascade="all, delete-orphan")
    historial_imc = relationship("HistorialIMC", back_populates="cliente", cascade="all, delete-orphan")
    progreso_calorias = relationship("ProgresoCalorias", back_populates="cliente", cascade="all, delete-orphan")
    alertas_salud = relationship("AlertaSalud", back_populates="cliente", cascade="all, delete-orphan")
    sugerencias_guardadas = relationship("SugerenciaGuardada", back_populates="cliente", cascade="all, delete-orphan")

    preferencias_alimentos = relationship("PreferenciaAlimento", back_populates="cliente", cascade="all, delete-orphan")
    preferencias_ejercicios = relationship(
        "PreferenciaEjercicio", back_populates="cliente", cascade="all, delete-orphan"
    )

    comida_registros = relationship("ComidaRegistro", back_populates="cliente", cascade="all, delete-orphan")

    verification_code = Column(String(6), nullable=True)
    code_expires_at = Column(DateTime, nullable=True)
