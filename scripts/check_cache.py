import json
import sqlite3
import os

# The cache_manager uses a simple SQLite dict or the Postgres DB? 
# In dependencies.py: CacheManager(db)
# Let's just query the raw DB table 'cache_entries' using sqlalchemy raw query.

from app.core.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
results = db.execute(text("SELECT key, value FROM cache_entries ORDER BY created_at DESC LIMIT 20")).fetchall()

for r in results:
    key, value = r
    if 'calofit:consulta:' in key:
        print(f"--- {key} ---")
        try:
            data = json.loads(value)
            print(f"Plato: {data.get('nombre')}")
            print(f"Kcal: {data.get('calorias')} | Prot: {data.get('proteinas_g')}")
        except:
            print(value)
