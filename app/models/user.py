from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.core.database import Base

class User(Base):
    """
    Modelo para el personal del Gimnasio World Light.
    Incluye Administradores, Coaches y Nutricionistas.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name_paternal = Column(String, nullable=False)
    last_name_maternal = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    role_name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    role_relation = relationship("Role", back_populates="users")
    
    clients_as_coach = relationship(
        "Client", 
        foreign_keys="Client.assigned_coach_id", 
        back_populates="coach"
    )
    clients_as_nutri = relationship(
        "Client", 
        foreign_keys="Client.assigned_nutri_id", 
        back_populates="nutritionist"
    )