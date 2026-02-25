from sqlalchemy.orm import Session
from app.models.client import Client
from app.models.historial import ProgresoCalorias, AlertaSalud
from app.services.ia_service import ia_engine
from app.core.utils import get_peru_date
from datetime import datetime
import json

class CopilotoService:
    def __init__(self):
        self.ia = ia_engine

    async def consultar_copiloto(self, mensaje: str, db: Session, current_user, historial=None):
        """
        Cerebro Clínico para el Personal de Salud (Nutricionistas/Admins).
        Proporciona análisis de datos sin realizar registros automáticos de comida.
        """
        
        # 1. Detectar si la consulta es sobre un paciente específico
        # Heurística mejorada: Busca nombres y apellidos
        entidades = self._extraer_entidades_paciente(mensaje)
        contexto_paciente = ""
        
        if entidades:
            # Buscar por nombre o apellido
            query = db.query(Client)
            for entidad in entidades:
                query = query.filter(
                    (Client.first_name.ilike(f"%{entidad}%")) | 
                    (Client.last_name_paternal.ilike(f"%{entidad}%")) |
                    (Client.last_name_maternal.ilike(f"%{entidad}%"))
                )
            
            paciente = query.first()
            
            if paciente:
                contexto_paciente = self._generar_contexto_clinico(paciente, db)
        
        # 2. Construir Prompt de Sistema para Staff
        nombre_staff = current_user.first_name if hasattr(current_user, 'first_name') else "colega"
        
        prompt_sistema = (
            f"Eres el Asistente Clínico Inteligente (Copiloto) de Calofit. "
            f"Estás hablando con {nombre_staff}, un profesional de la salud con rol {current_user.role_name}. "
            f"\n\nTU MISIÓN: "
            f"1. Analizar datos de pacientes (adherencia, alertas, progreso). "
            f"2. Sugerir ajustes técnicos en macros o entrenamiento. "
            f"3. NO realices registros de comida ni ejercicio. "
            f"\n\nFORMATO DE RESPUESTA OBLIGATORIO: "
            f"Usa Markdown rico para que la información sea fácil de leer de un vistazo: "
            f"- Usa **negritas** para datos clave (pesos, calorías, nombres). "
            f"- Usa ## Títulos para separar secciones (ej: ## Análisis de Progreso). "
            f"- Usa Listas con puntos para sugerencias o alertas. "
            f"- Al iniciar un reporte de paciente, usa SIEMPRE el formato: ## Información de [Nombre del Paciente]"
            f"\n\nDIRECTRIZ DE CONCISIÓN: "
            f"No envíes paredes de texto. Sé ejecutivo. Si el historial es largo, resume las tendencias. "
            f"Prioriza la legibilidad técnica sobre la narración extensa. "
            f"\n\nCONTEXTO DEL PACIENTE CONSULTADO:\n{contexto_paciente if contexto_paciente else 'No se ha detectado un paciente específico aún. Pide el nombre o apellido si es necesario.'}"
        )

        # 3. Llamar a Groq con el nuevo contexto clínico
        respuesta_ia = await self.ia.asistir_cliente(
            contexto=prompt_sistema,
            mensaje_usuario=mensaje,
            historial=historial,
            tono_aplicado="Profesional clínico"
        )

        # 4. Parsear respuesta (Usamos el mismo parser para mantener compatibilidad de UI)
        from app.services.response_parser import parsear_respuesta_para_frontend
        respuesta_estructurada = parsear_respuesta_para_frontend(respuesta_ia, mensaje_usuario=mensaje)
        
        # Limpieza de secciones: En el copiloto staff NO queremos cards de "Añadir Comida" 
        if "secciones" in respuesta_estructurada:
            respuesta_estructurada["secciones"] = [
                s for s in respuesta_estructurada["secciones"] 
                if s.get("tipo") != "comida"
            ]

        return {
            "staff": nombre_staff,
            "respuesta_ia": respuesta_ia,
            "respuesta_estructurada": respuesta_estructurada,
            "rol_detectado": "clinico"
        }

    def _extraer_entidades_paciente(self, mensaje: str):
        # Heurística mejorada: Extrae palabras con mayúscula inicial o después de palabras clave
        palabras = mensaje.split()
        claves = ["paciente", "sobre", "de", "cliente", "revisa", "analiza"]
        entidades = []
        
        for i, palabra in enumerate(palabras):
            clean_word = palabra.replace("?", "").replace(".", "").replace(",", "")
            
            # Si es una palabra clave, la siguiente podría ser un nombre
            if clean_word.lower() in claves and i + 1 < len(palabras):
                next_word = palabras[i+1].replace("?", "").replace(".", "").replace(",", "")
                if next_word.lower() not in claves:
                    entidades.append(next_word)
            
            # Si empieza con Mayúscula y no es la primera palabra del mensaje (pobre hombre's NER)
            elif clean_word and clean_word[0].isupper() and i > 0:
                entidades.append(clean_word)
                
        return list(set(entidades))

    def _generar_contexto_clinico(self, paciente: Client, db: Session):
        hoy = get_peru_date()
        # FIX: AlertaSalud usa 'fecha_deteccion' en lugar de 'fecha'
        alertas = db.query(AlertaSalud).filter(AlertaSalud.client_id == paciente.id).order_by(AlertaSalud.fecha_deteccion.desc()).limit(3).all()
        progreso = db.query(ProgresoCalorias).filter(ProgresoCalorias.client_id == paciente.id).order_by(ProgresoCalorias.fecha.desc()).limit(7).all()
        
        texto_alertas = "; ".join([f"{a.tipo}: {a.descripcion} ({a.severidad})" for a in alertas]) if alertas else "Sin alertas recientes."
        
        media_adherencia = 0
        if progreso:
             media_adherencia = sum([p.calorias_consumidas for p in progreso]) / len(progreso)

        return (
            f"PACIENTE: {paciente.first_name} {paciente.last_name_paternal}. "
            f"META: {paciente.goal}. PESO: {paciente.weight}kg. "
            f"ALERTAS RECIENTES: {texto_alertas}. "
            f"CONSUMO MEDIO SEMANAL: {media_adherencia:.0f} kcal."
        )

copiloto_service = CopilotoService()
