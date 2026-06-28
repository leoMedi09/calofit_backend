from sqlalchemy import Column, Integer, String, Float, Date, Text, TIMESTAMP, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class HistorialPeso(Base):
    """
    Tabla para almacenar el historial de peso de los clientes
    """
    __tablename__ = "historial_peso"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    peso_kg = Column(Float, nullable=False)
    fecha_registro = Column(Date, nullable=False, default=func.current_date())
    notas = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP, nullable=False, default=func.now())

    # Relación con cliente
    cliente = relationship("Client", back_populates="historial_peso")

class HistorialIMC(Base):
    """
    Tabla para almacenar el historial de IMC de los clientes
    """
    __tablename__ = "historial_imc"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    imc = Column(Float, nullable=False)
    categoria = Column(String(50), nullable=False)
    fecha_registro = Column(Date, nullable=False, default=func.current_date())
    created_at = Column(TIMESTAMP, nullable=False, default=func.now())

    # Relación con cliente
    cliente = relationship("Client", back_populates="historial_imc")

class ProgresoCalorias(Base):
    """
    Tabla para almacenar el progreso diario de calorías consumidas y quemadas
    """
    __tablename__ = "progreso_calorias"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    fecha = Column(Date, nullable=False, default=func.current_date())
    calorias_consumidas = Column(Integer, nullable=True, default=0)
    calorias_quemadas = Column(Integer, nullable=False, default=0)
    
    # Nuevos campos para tracking de macros (v41.0)
    proteinas_consumidas = Column(Float, nullable=True, default=0.0)
    carbohidratos_consumidos = Column(Float, nullable=True, default=0.0)
    grasas_consumidas = Column(Float, nullable=True, default=0.0)
    
    deficit_superavit = Column(Integer, nullable=True)  # Calculado: consumidas - (tmb + quemadas)
    alerta_exceso_enviada = Column(Boolean, nullable=False, default=False)  # push "saliste de tu meta" ya enviado hoy
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

    # Relación con cliente
    cliente = relationship("Client", back_populates="progreso_calorias")

    def calcular_deficit_superavit(self, tmb_diario: float):
        """
        Calcula el déficit o superávit calórico del día
        """
        if self.calorias_consumidas is not None:
            self.deficit_superavit = int(self.calorias_consumidas - (tmb_diario + self.calorias_quemadas))
        return self.deficit_superavit

class AlertaSalud(Base):
    """
    Tabla para registrar alertas de salud (fatiga, lesiones, desánimo)
    detectadas por la IA o reportadas por el cliente.
    """
    __tablename__ = "alertas_salud"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    
    # Tipo de alerta: 'fatiga', 'lesion', 'desanimo', 'otro'
    tipo = Column(String(50), nullable=False)
    descripcion = Column(Text, nullable=False)
    
    # Nivel de severidad: 'bajo', 'medio', 'alto'
    severidad = Column(String(20), default='bajo')
    
    # Estado: 'pendiente', 'en_proceso', 'atendida'
    estado = Column(String(20), default='pendiente')
    
    # Quién atendió la alerta (si aplica)
    atendido_por_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Notas del staff sobre la atención
    notas = Column(Text, nullable=True)
    
    fecha_deteccion = Column(TIMESTAMP, nullable=False, default=func.now())
    fecha_atencion = Column(TIMESTAMP, nullable=True)  # Cuando se marcó como atendida
    created_at = Column(TIMESTAMP, nullable=False, default=func.now())

    # Relación con cliente
    cliente = relationship("Client", back_populates="alertas_salud")


class SugerenciaGuardada(Base):
    """
    Sugerencias (recetas/rutinas) guardadas por el usuario desde el chat.
    Permite al usuario guardar platos o ejercicios sugeridos por la IA
    para prepararlos o hacerlos después.
    """
    __tablename__ = "sugerencias_guardadas"

    id = Column(Integer, primary_key=True, index=True)
    client_id = Column(Integer, ForeignKey("clients.id"), nullable=False)
    tipo = Column(String(20), nullable=False)        # 'comida' o 'ejercicio'
    nombre = Column(String(255), nullable=False)
    ingredientes = Column(JSON, nullable=True)       # Lista de ingredientes
    preparacion = Column(JSON, nullable=True)         # Lista de pasos
    macros = Column(String(255), nullable=True)       # "P: 25g | C: 40g | G: 12g | Cal: 380kcal"
    nota = Column(Text, nullable=True)
    completada = Column(Boolean, default=False)       # Ya la preparó/hizo
    fecha_guardado = Column(TIMESTAMP, nullable=False, default=func.now())

    cliente = relationship("Client", back_populates="sugerencias_guardadas")