import sys
import os
sys.path.append(os.path.dirname(__file__))

from app.core.database import SessionLocal
from app.models.plato import Plato

db = SessionLocal()
platos = db.query(Plato).filter(Plato.nombre.ilike('%tortilla%')).all()
for p in platos:
    print(p.id, p.nombre)
