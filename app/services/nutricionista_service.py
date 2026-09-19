from sqlalchemy.orm import Session
from app.models.client import Client
from app.models.historial import ProgresoCalorias, AlertaSalud
from app.services.ia_service import ia_engine
from app.core.utils import get_peru_date
from datetime import datetime
import json
import re

class NutricionistaIAService:
    def __init__(self):
        self.ia = ia_engine

    async def consultar(self, mensaje: str, db: Session, current_user, historial=None):
        """
        Cerebro Clínico para Nutricionistas y Coaches.
        Enfoque: Adherencia, Alertas de Salud, Ajustes Técnicos.
        """
        entidades = self._extraer_entidades_paciente(mensaje)
        contexto_paciente = ""
        
        if entidades:
            query = db.query(Client)
            if hasattr(current_user, 'id') and current_user.role_name.lower() != "admin":
                query = query.filter(Client.nutritionist_id == current_user.id)
            
            for entidad in entidades:
                query = query.filter(
                    (Client.first_name.ilike(f"%{entidad}%")) | 
                    (Client.last_name_paternal.ilike(f"%{entidad}%")) |
                    (Client.last_name_maternal.ilike(f"%{entidad}%"))
                )
            
            paciente = query.first()
            if paciente:
                contexto_paciente = self._generar_contexto_clinico(paciente, db)
        
        nombre_staff = current_user.first_name if hasattr(current_user, 'first_name') else "colega"
        
        prompt_sistema = (
            f"Eres el Asistente Clínico Inteligente (Copiloto) de Calofit. "
            f"Estás hablando con {nombre_staff}, un profesional de la salud ({current_user.role_name}). "
            f"\n\nTU MISIÓN CLÍNICA: "
            f"1. Analizar el progreso clínico y nutricional de los pacientes. "
            f"2. Identificar riesgos en alertas de salud y baja adherencia. "
            f"3. Sugerir ajustes en el plan nutricional basados en la evidencia y el contexto del paciente. "
            f"4. NO realices registros de comida ni ejercicio. "
            f"\n\nDIRECTRIZ MANDATORIA: Debes priorizar las listas de alimentos Recomendados/Prohibidos definidas en el contexto."
            f"\n\nFORMATO DE RESPUESTA: Usa Markdown rico. Estructura con ## Títulos y **Negritas**."
            f"\n\nCONTEXTO DEL PACIENTE:\n{contexto_paciente if contexto_paciente else 'No se ha detectado un paciente asignado. Pide el nombre si es necesario.'}"
        )

        respuesta_ia = await self.ia.asistir_cliente(
            contexto=prompt_sistema,
            mensaje_usuario=mensaje,
            historial=historial,
            tono_applied="Profesional clínico-nutricional"
        )

        from app.services.response_parser import parsear_respuesta_para_frontend
        respuesta_estructurada = parsear_respuesta_para_frontend(respuesta_ia, mensaje_usuario=mensaje)
        
        if "secciones" in respuesta_estructurada:
            respuesta_estructurada["secciones"] = [s for s in respuesta_estructurada["secciones"] if s.get("tipo") != "comida"]

        return {
            "staff": nombre_staff,
            "respuesta_ia": respuesta_ia,
            "respuesta_estructurada": respuesta_estructurada,
            "rol": "nutricionista"
        }

    def _extraer_entidades_paciente(self, mensaje: str):
        palabras = mensaje.split()
        claves = ["paciente", "sobre", "de", "cliente", "revisa", "analiza"]
        entidades = []
        for i, palabra in enumerate(palabras):
            clean_word = palabra.replace("?", "").replace(".", "").replace(",", "")
            if clean_word.lower() in claves and i + 1 < len(palabras):
                entidades.append(palabras[i+1].replace("?", "").replace(".", "").replace(",", ""))
            elif clean_word and clean_word[0].isupper() and i > 0:
                entidades.append(clean_word)
        return list(set(entidades))

    def _generar_contexto_clinico(self, paciente: Client, db: Session):
        hoy = get_peru_date()
        alertas = db.query(AlertaSalud).filter(AlertaSalud.client_id == paciente.id).order_by(AlertaSalud.fecha_deteccion.desc()).limit(3).all()
        progreso = db.query(ProgresoCalorias).filter(ProgresoCalorias.client_id == paciente.id).order_by(ProgresoCalorias.fecha.desc()).limit(7).all()
        
        texto_alertas = "; ".join([f"{a.tipo}: {a.descripcion} ({a.severidad})" for a in alertas]) or "Sin alertas."
        media_adherencia = sum([p.calorias_consumidas for p in progreso]) / len(progreso) if progreso else 0

        return (
            f"PACIENTE: {paciente.first_name} {paciente.last_name_paternal}. "
            f"META: {paciente.goal}. PESO: {paciente.weight}kg. "
            f"ALERTAS: {texto_alertas}. "
            f"ADHERENCIA MEDIA: {media_adherencia:.0f} kcal."
        )

nutricionista_ia_service = NutricionistaIAService()
