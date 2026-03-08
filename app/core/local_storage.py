import os
import uuid
import cloudinary
import cloudinary.uploader
from datetime import datetime
from app.core.config import settings

# Configurar Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True
)

class LocalStorage:
    @staticmethod
    def save_file(file_bytes: bytes, original_filename: str) -> str:
        """
        Sube un archivo a Cloudinary y retorna la URL pública válida.
        """
        try:
            # Subir a Cloudinary
            # Usamos el buffer de bytes directamente
            upload_result = cloudinary.uploader.upload(
                file_bytes,
                folder="profiles",
                resource_type="image"
            )
            
            public_url = upload_result.get("secure_url")
            
            if public_url:
                print(f"✅ Imagen subida exitosamente a Cloudinary: {public_url}")
                return public_url
            
        except Exception as e:
            print(f"❌ Error al subir a Cloudinary: {e}")
            
        return ""

    @staticmethod
    def get_public_url(relative_path: str) -> str:
        """
        Si la ruta ya es una URL (Cloudinary), la retorna tal cual.
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
        Maneja la eliminación de archivos. Para Cloudinary, simplemente retorna True
        (se podría implementar destroy, pero por ahora evitamos complejidad extra).
        """
        if not public_url:
            return False
            
        try:
            # Si es una URL de Cloudinary o Firebase, omitimos eliminación física local
            if "cloudinary.com" in public_url or "storage.googleapis.com" in public_url:
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
                    print(f"✅ Archivo local eliminado: {system_path}")
                    return True
        except Exception as e:
            print(f"❌ Error eliminando archivo: {e}")
            
        return False

local_storage = LocalStorage()
