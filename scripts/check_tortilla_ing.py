import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from sqlalchemy import text
db = SessionLocal()
res = db.execute(text("SELECT id, nombre, proteina_100g FROM alimentos WHERE nombre ILIKE '%avena%' LIMIT 10")).fetchall()
for r in res:
    print(f"{r[0]}: {r[1]} -> P: {r[2]}g")
