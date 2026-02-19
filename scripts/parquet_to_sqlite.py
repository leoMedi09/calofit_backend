import sqlite3
import pyarrow.parquet as pq
import os
import gc
import numpy as np

def parquet_a_sqlite():
    # Rutas
    base_dir = os.path.dirname(__file__)
    parquet_path = os.path.join(base_dir, "..", "app", "data", "food.parquet")
    db_path = os.path.join(base_dir, "..", "app", "data", "alimentos_mundo.db")
    
    print(f"🚀 Iniciando migración PREMIUM (15 NUTRIENTES): Parquet -> SQLite")
    print(f"📂 Origen: {parquet_path}")
    print(f"🗄️ Destino: {db_path}")
    
    # 1. Conectar a SQLite
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Limpieza inicial
    cursor.execute("DROP TABLE IF EXISTS alimentos")
    
    # 2. Crear Tabla con los 15 MAGNÍFICOS
    cursor.execute("""
        CREATE TABLE alimentos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT,
            marca TEXT,
            calorias REAL DEFAULT 0,
            proteinas REAL DEFAULT 0,
            carbohidratos REAL DEFAULT 0,
            azucares REAL DEFAULT 0,
            grasas REAL DEFAULT 0,
            grasas_saturadas REAL DEFAULT 0,
            grasas_trans REAL DEFAULT 0,
            grasas_mono REAL DEFAULT 0,
            grasas_poli REAL DEFAULT 0,
            fibra REAL DEFAULT 0,
            sodio REAL DEFAULT 0,
            calcio REAL DEFAULT 0,
            hierro REAL DEFAULT 0,
            vitamina_a REAL DEFAULT 0,
            vitamina_c REAL DEFAULT 0,
            pais TEXT
        )
    """)
    conn.commit()
    
    # 3. Procesar Parquet
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        total_importado = 0
        total_filas = parquet_file.metadata.num_rows
        
        print(f"📊 Total de filas a procesar: {total_filas:,}")
        
        # Iterar por bloques
        for i in range(parquet_file.num_row_groups):
            cols = ['product_name', 'brands', 'countries_tags', 'nutriments']
            table = parquet_file.read_row_group(i, columns=cols)
            df = table.to_pandas()
            
            batch_data = []
            
            for index, row in df.iterrows():
                try:
                    # Inicializar vector de 15 nutrientes
                    n_vals = {
                        'calorias': 0.0, 'prot': 0.0, 'carb': 0.0, 'azu': 0.0,
                        'fat': 0.0, 'sat': 0.0, 'trans': 0.0, 'mono': 0.0, 'poli': 0.0,
                        'fib': 0.0, 'sod': 0.0, 'cal': 0.0, 'fe': 0.0, 'vit_a': 0.0, 'vit_c': 0.0
                    }
                    
                    nutris = row['nutriments']
                    found_energy_kcal = False
                    
                    # Extracción inteligente
                    if isinstance(nutris, (list, np.ndarray)) or hasattr(nutris, '__iter__'):
                         if not isinstance(nutris, (str, dict)):
                            for n in nutris:
                                if isinstance(n, dict):
                                    name = n.get('name')
                                    val = n.get('100g')
                                    if val is None: val = n.get('value')
                                    if val is None: val = 0.0
                                    
                                    val = float(val)
                                    
                                    # Mapeo de claves OpenFoodFacts -> Nuestras variables
                                    if name == 'energy-kcal':
                                        n_vals['calorias'] = val
                                        found_energy_kcal = True
                                    elif name == 'energy' and not found_energy_kcal:
                                        n_vals['calorias'] = val * 0.239
                                    elif name == 'proteins': n_vals['prot'] = val
                                    elif name == 'carbohydrates': n_vals['carb'] = val
                                    elif name == 'sugars': n_vals['azu'] = val
                                    elif name == 'fat': n_vals['fat'] = val
                                    elif name == 'saturated-fat': n_vals['sat'] = val
                                    elif name == 'trans-fat': n_vals['trans'] = val
                                    elif name == 'monounsaturated-fat': n_vals['mono'] = val
                                    elif name == 'polyunsaturated-fat': n_vals['poli'] = val
                                    elif name in ['fiber', 'dietary-fiber']: n_vals['fib'] = val
                                    elif name == 'sodium': n_vals['sod'] = val
                                    elif name == 'salt' and n_vals['sod'] == 0: n_vals['sod'] = val / 2.5
                                    elif name == 'calcium': n_vals['cal'] = val
                                    elif name == 'iron': n_vals['fe'] = val
                                    elif name == 'vitamin-a': n_vals['vit_a'] = val
                                    elif name == 'vitamin-c': n_vals['vit_c'] = val

                    # Extracción de Nombre
                    nombre_raw = row['product_name']
                    nombre = ""
                    if isinstance(nombre_raw, (list, np.ndarray)) and not isinstance(nombre_raw, str):
                        for n in nombre_raw:
                            if isinstance(n, dict):
                                lang = n.get('lang')
                                text = n.get('text')
                                if lang in ['es', 'main', 'en'] and text:
                                    nombre = text
                                    if lang == 'es': break
                    elif isinstance(nombre_raw, str):
                        nombre = nombre_raw
                    
                    # Filtro de Calidad
                    if len(nombre) > 2 and (n_vals['calorias'] > 0 or n_vals['prot'] > 0):
                        marca = str(row['brands']) if row['brands'] else ""
                        pais = str(row['countries_tags']) if row['countries_tags'] else ""
                        
                        # Tupla ordenada para insertar
                        batch_data.append((
                            nombre, marca, 
                            n_vals['calorias'], n_vals['prot'], n_vals['carb'], n_vals['azu'],
                            n_vals['fat'], n_vals['sat'], n_vals['trans'], n_vals['mono'], n_vals['poli'],
                            n_vals['fib'], n_vals['sod'], n_vals['cal'], n_vals['fe'], n_vals['vit_a'], n_vals['vit_c'],
                            pais
                        ))
                        
                except Exception:
                    continue 
            
            # Insertar en DB
            if batch_data:
                cursor.executemany("""
                    INSERT INTO alimentos (
                        nombre, marca, calorias, proteinas, carbohidratos, azucares,
                        grasas, grasas_saturadas, grasas_trans, grasas_mono, grasas_poli,
                        fibra, sodio, calcio, hierro, vitamina_a, vitamina_c, pais
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch_data)
                conn.commit()
                total_importado += len(batch_data)
            
            # Limpiar RAM
            del df
            del table
            del batch_data
            gc.collect()
            
            if i % 10 == 0:
                print(f"⏳ Progreso: Grupo {i}/{parquet_file.num_row_groups} | Importados: {total_importado:,}")

        # Indices finales
        print("⚙️ Creando índices finales...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_nombre ON alimentos(nombre)")
        conn.commit()
        
        print(f"\n✅ MIGRACIÓN PREMIUM COMPLETADA. Total: {total_importado:,}")
        
    except Exception as e:
        print(f"❌ Error Crítico: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    parquet_a_sqlite()
