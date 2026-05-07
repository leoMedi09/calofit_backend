import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from sqlalchemy import text

def clean_corrupt_plates():
    db = SessionLocal()
    try:
        print("Iniciando limpieza de platos corruptos (sin ingredientes)...")
        # Encontrar platos que no tienen ningún ingrediente asociado
        query = text("""
            SELECT p.id, p.nombre 
            FROM platos p 
            LEFT JOIN plato_ingredientes pi ON p.id = pi.plato_id 
            WHERE pi.id IS NULL
        """)
        corrupt_plates = db.execute(query).fetchall()
        
        if not corrupt_plates:
            print("No se encontraron platos corruptos. Todo en orden.")
            return

        print(f"Se encontraron {len(corrupt_plates)} platos corruptos.")
        for p in corrupt_plates:
            print(f"Eliminando plato corrupto: [{p[0]}] {p[1]}")
            
        # Eliminar
        delete_query = text("""
            DELETE FROM platos 
            WHERE id IN (
                SELECT p.id 
                FROM platos p 
                LEFT JOIN plato_ingredientes pi ON p.id = pi.plato_id 
                WHERE pi.id IS NULL
            )
        """)
        result = db.execute(delete_query)
        db.commit()
        print(f"Limpieza completada. {result.rowcount} platos eliminados.")
        
    except Exception as e:
        print(f"Error durante limpieza: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    clean_corrupt_plates()
