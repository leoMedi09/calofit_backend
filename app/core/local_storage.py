import os
import uuid
from datetime import datetime
from app.core.config import settings
from app.core.firebase import upload_to_firebase

class LocalStorage:
    @staticmethod
    def save_file(file_bytes: bytes, original_filename: str) -> str:
        """
        Sube un archivo a Firebase Storage y retorna la URL pública válida.
        """
        try:
            # Generar un nombre único para el archivo
            ext = os.path.splitext(original_filename)[1]
            if not ext:
                ext = ".jpg"
            
            # Ruta en Firebase: profiles/UUID.ext
            remote_path = f"profiles/{uuid.uuid4()}{ext}"
            
            # Detectar tipo de contenido
            content_type = "image/jpeg"
            if ext.lower() == ".png":
                content_type = "image/png"
            elif ext.lower() == ".webp":
                content_type = "image/webp"
                
            # Usar el servicio de Firebase ya existente
            public_url = upload_to_firebase(file_bytes, remote_path, content_type=content_type)
            
            if public_url:
                print(f"✅ Imagen subida con éxito a Firebase: {public_url}")
                return public_url
            
        except Exception as e:
            print(f"❌ Error al subir a Firebase Storage: {e}")
            
        return ""

    @staticmethod
    def get_public_url(relative_path: str) -> str:
        """
        Si la ruta ya es una URL (Firebase/Cloudinary), la retorna tal cual.
        Si es relativa, le añade la base URL.
        """
        if not relative_path:
            return ""
            
        if relative_path.startswith("http"):
            return relative_path
            
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        return f"{base_url}{relative_path}"

    @staticmethod
    def delete_file(public_url: str) -> bool:
        """
        Maneja la eliminación de archivos. Para servicios en la nube, marcamos como éxito.
        """
        if not public_url:
            return False
            
        try:
            # Si es una URL de Cloudinary o Firebase, no intentamos borrar de local
            if any(domain in public_url for domain in ["cloudinary.com", "firebasestorage.googleapis.com", "storage.googleapis.com"]):
                return True

            relative_path = ""
            if "/uploads/" in public_url:
                relative_path = "/uploads/" + public_url.split("/uploads/")[-1]
            else:
                return False

            # Convertir ruta relativa a ruta de sistema
            system_path = os.path.join("app", relative_path.lstrip("/"))
            
            if os.path.exists(system_path) and os.path.isfile(system_path):
                if system_path.startswith("app/uploads"):
                    os.remove(system_path)
                    return True
        except Exception as e:
            print(f"❌ Error eliminando archivo local: {e}")
            
        return False

local_storage = LocalStorage()
