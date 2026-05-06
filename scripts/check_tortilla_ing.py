import sys
import os
sys.path.append(os.path.dirname(__file__))

from app.core.database import SessionLocal
from sqlalchemy import text
db = SessionLocal()
res = db.execute(text("SELECT a.nombre FROM plato_ingredientes pi JOIN alimentos a ON pi.alimento_id = a.id WHERE pi.plato_id = 385")).fetchall()
print([r[0] for r in res])
