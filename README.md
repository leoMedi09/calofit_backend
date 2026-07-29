# CaloFit — Backend

> Tesis 2026 · Gimnasio World Light · Lambayeque, Perú
> FastAPI · SQLAlchemy · PostgreSQL 15 · Groq (Llama-3) · Random Forest · KNN

Backend del sistema CaloFit. El frontend Flutter vive en [`../calofit_frontend`](../calofit_frontend) (ver su propio README).

---

## Tabla de contenidos

1. [Stack tecnológico](#stack-tecnológico)
2. [Arquitectura](#arquitectura)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Levantar el entorno](#levantar-el-entorno)
5. [Cómo implementar un endpoint](#cómo-implementar-un-endpoint)
6. [Autenticación JWT](#autenticación-jwt)
7. [Endpoints disponibles](#endpoints-disponibles)
8. [Base de datos](#base-de-datos)
9. [Comandos útiles](#comandos-útiles)
10. [Credenciales de prueba](#credenciales-de-prueba)

---

## Stack tecnológico

| Elemento | Detalle |
|---|---|
| Framework | FastAPI (Python 3.11) |
| ORM | SQLAlchemy 2.x |
| Migraciones | `Base.metadata.create_all()` + `ALTER TABLE IF NOT EXISTS` manual en `app/main.py` |
| Base de datos | PostgreSQL 15 (Docker `calofit_db`, puerto externo 5433) |
| IA / LLM | Llama-3 vía Groq API (`ia_service.py`) |
| ML Perfil | Random Forest (`perfil_adherencia.pkl`) — 14 features → PERFIL A/B/C |
| ML Recomendador | KNN Coseno (`recomendador_knn.pkl`) — vector `[kcal, prot, carb, grasa]` |
| Auth | Firebase Admin SDK + JWT |
| Contenedores | Docker Compose (`calofit_db`, `calofit_backend`, `calofit_frontend`) |

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                   Flutter (Cliente)                      │
│   Provider + Dio → lib/services/api_service.dart        │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/JSON  (Bearer JWT)
                       ▼
┌─────────────────────────────────────────────────────────┐
│               FastAPI (Puerto 8000)                      │
│                                                          │
│  app/api/routes/ (api_router)     app/api/v1/           │
│  ├─ /auth               ├─ /alertas                     │
│  ├─ /asistente          ├─ /balance                     │
│  ├─ /clientes           ├─ /alimentos                   │
│  ├─ /ejercicios         ├─ /nutricionista                │
│  ├─ /nutricion          ├─ /admin                        │
│  ├─ /dashboard          ├─ /usuarios                     │
│  ├─ /copiloto           └─ /notifications                │
│  └─ /api/v1/nutrition/parse_ingredients (único v1 activo)│
│                                                          │
│  app/core/               app/models/                    │
│  ├─ database.py (SessionLocal)  ├─ client.py            │
│  ├─ security.py (JWT)           ├─ alimento.py, plato.py │
│  └─ config.py (.env vars)       └─ nutricion.py, ...    │
│                                                          │
│  app/services/asistente/  (orquestador + 5 módulos NLP) │
│  app/services/nutrition/  (food resolver, plate builder)│
│  app/services/ml_service.py, rutina_service.py, ...      │
└──────────────────────┬──────────────────────────────────┘
                       │ SQLAlchemy ORM
                       ▼
┌─────────────────────────────────────────────────────────┐
│         PostgreSQL 15 — BD_Calofit (Puerto 5433)        │
│  39 tablas · pg_trgm · unaccent                          │
└─────────────────────────────────────────────────────────┘
```

> Nota histórica: hubo una "arquitectura nueva" bajo `/api/v1/assistant`, `/api/v1/exercise` y `app/api/dependencies.py` que nunca recibió tráfico real de Flutter — se eliminó el 2026-06-13 (ver `CLAUDE.md`). El único endpoint `/api/v1` en uso hoy es `parse_ingredients`.

### Flujo de una petición (ejemplo: registro de comida)

```
Flutter                Backend                     BD
  │                       │                          │
  ├─POST /asistente/       │                          │
  │  log-inteligente ─────►│                          │
  │  {mensaje: "..."}      │                          │
  │                        ├─ get_current_user()      │
  │                        ├─ asistente_service       │
  │                        │   .registrar_por_nlp()   │
  │                        ├─ NLP 5 capas             │
  │                        ├─ INSERT progreso_calorias►│
  │                        ├─ INSERT workout_logs ────►│
  │◄───────────────────────┤                          │
  │  {success, balance_    │                          │
  │   actualizado, ...}    │                          │
```

---

## Estructura del proyecto

```
calofit_backend/
├── app/
│   ├── main.py                 # Entrada FastAPI: middlewares, routers, migraciones manuales
│   ├── core/
│   │   ├── config.py           # Variables de entorno (settings)
│   │   ├── database.py         # Engine + SessionLocal + get_db()
│   │   ├── security.py         # Hash de contraseñas + JWT
│   │   ├── utils.py            # Fecha Perú, parseo de macros, TMB
│   │   ├── logging_config.py   # get_logger() con RotatingFileHandler
│   │   ├── mets_gym.py         # Diccionario MET por ejercicio (ACSM)
│   │   └── macros_diarios.py   # Cálculo P/C/G desde kcal y objetivo
│   │
│   ├── models/                 # SQLAlchemy ORM (una clase por tabla)
│   ├── schemas/                # Pydantic: request/response validation
│   │
│   ├── api/
│   │   ├── __init__.py         # api_router: agrupa todas las rutas activas
│   │   ├── routes/             # auth, asistente, clientes, ejercicios, balance,
│   │   │                       # dashboard, nutricion, nutricionista, copiloto,
│   │   │                       # alertas, alimentos, admin, usuarios, notifications
│   │   └── v1/
│   │       └── nutrition/parser.py  # único endpoint v1 activo
│   │
│   └── services/
│       ├── asistente/          # orquestador + 5 módulos NLP (ver más abajo)
│       ├── nutrition/          # food resolver, plate builder, validators
│       ├── ml_service.py       # Clasificador RF + KNN
│       ├── rutina_service.py   # Generador de rutinas adaptativas
│       ├── plato_constructor.py # Constructor dinámico de platos vía LLM
│       ├── nlp_food_extractor.py
│       ├── ia_service.py       # Wrapper Groq API
│       └── fatsecret_client.py # FatSecret API
│
├── scripts/                    # Entrenamiento ML, QA, seeds puntuales
├── tests/                      # Pytest (unit, integration, e2e, external)
└── logs/                       # error.log (RotatingFileHandler)
```

### Módulos del asistente (`app/services/asistente/`)

| Módulo | Responsabilidad |
|---|---|
| `asistente_service.py` | Orquestador puro |
| `asistente_plan.py` | `obtener_plan_hoy()` — Mifflin-St Jeor fallback |
| `asistente_registro_comida.py` | 5 capas NLP + registro manual |
| `asistente_registro_ejercicio.py` | MET formula + `workout_logs` (SQL raw) |
| `asistente_recomendaciones.py` | 14 features RF + KNN coseno |
| `asistente_prompt.py` | Builder de prompt + fallbacks deterministas |
| `asistente_modos.py` | Clasificación de intención |

---

## Levantar el entorno

### Requisitos

- Docker Desktop >= 24.0
- Git

### 1. Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env`:

| Variable | Cómo obtenerla |
|----------|----------------|
| `SECRET_KEY` | `python -c "import secrets; print(secrets.token_hex(32))"` |
| `GROQ_API_KEY` | https://console.groq.com/keys |
| `FATSECRET_CLIENT_ID` / `_SECRET` | https://platform.fatsecret.com/api/ |
| `USDA_API_KEY` | https://fdc.nal.usda.gov/api-guide.html |
| `POSTGRES_PASSWORD` | Contraseña que elijas |

### 2. Levantar contenedores (desde la raíz del repo)

```bash
docker-compose up --build -d
```

| Contenedor | Puerto | Descripción |
|------------|--------|-------------|
| `calofit_db` | 5433 → 5432 | PostgreSQL 15 |
| `calofit_backend` | 8000 | FastAPI |
| `calofit_frontend` | 80 | Flutter Web |

### 3. Verificar

```bash
curl http://localhost:8000/health
# {"status":"OK","version":"1.0.0"}
```

Documentación interactiva: http://localhost:8000/docs

### 4. Cargar datos de prueba (esquema + datos completos)

```bash
docker exec -i calofit_db psql -U postgres -d BD_Calofit < ../backups/calofit_backup.sql
```

---

## Cómo implementar un endpoint

Ejemplo completo: agregar `GET /clientes/{id}/resumen-macros` que devuelve los macros consumidos hoy.

### Paso 1 — Schema Pydantic

```python
# app/schemas/client.py
class ResumenMacrosResponse(BaseModel):
    fecha: str
    calorias_consumidas: float
    proteinas_g: float
    carbohidratos_g: float
    grasas_g: float
```

### Paso 2 — Lógica en el servicio

```python
# app/services/nutricion_service.py (o el servicio que corresponda)
from app.models.nutricion import ProgresoCalorias
from app.core.utils import get_peru_date

def obtener_resumen_macros_hoy(client_id: int, db: Session) -> dict:
    hoy = get_peru_date()
    prog = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == client_id,
        ProgresoCalorias.fecha == hoy,
    ).first()
    if not prog:
        return {"fecha": str(hoy), "calorias_consumidas": 0, "proteinas_g": 0,
                "carbohidratos_g": 0, "grasas_g": 0}
    return {
        "fecha": str(hoy),
        "calorias_consumidas": prog.calorias_consumidas or 0,
        "proteinas_g": prog.proteinas_consumidas or 0,
        "carbohidratos_g": prog.carbohidratos_consumidos or 0,
        "grasas_g": prog.grasas_consumidas or 0,
    }
```

### Paso 3 — Endpoint en la ruta correspondiente

```python
# app/api/routes/clientes.py
from app.services.nutricion_service import obtener_resumen_macros_hoy
from app.schemas.client import ResumenMacrosResponse

@router.get("/{client_id}/resumen-macros", response_model=ResumenMacrosResponse)
def resumen_macros(
    client_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    return obtener_resumen_macros_hoy(client_id, db)
```

> El router de `clientes.py` ya está registrado en `app/api/__init__.py` con el prefijo `/clientes`, así que el endpoint queda disponible en `GET /clientes/{id}/resumen-macros` sin ningún cambio adicional.

### Checklist

```
[ ] Schema Pydantic en app/schemas/
[ ] Lógica en app/services/
[ ] Endpoint en app/api/routes/<modulo>.py
[ ] El router ya está en app/api/__init__.py (verificar)
[ ] Si necesita nueva tabla: modelo SQLAlchemy + ALTER TABLE en main.py si aplica
```

Para consumirlo desde Flutter, ver [`../calofit_frontend/README.md`](../calofit_frontend/README.md).

---

## Autenticación JWT

Todos los endpoints (excepto `/auth/login` y `/auth/register`) requieren el header:

```
Authorization: Bearer <token>
```

### Obtener el token

```bash
POST /auth/login
{
  "email": "usuario@email.com",
  "password": "CaloFit2024!",
  "firebase_uid": "uid_de_firebase"   # solo para clientes
}
# Responde: {"access_token": "eyJ...", "token_type": "bearer"}
```

### Proteger un endpoint en FastAPI

```python
from app.api.routes.auth import get_current_user

@router.get("/mi-ruta")
def mi_ruta(current_user=Depends(get_current_user)):
    # current_user es el objeto Client o User autenticado
    return {"email": current_user.email}
```

---

## Endpoints disponibles

### Autenticación — `/auth`

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/auth/login` | Login cliente o staff |
| POST | `/auth/register` | Registro de usuario staff |

### Clientes — `/clientes`

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/clientes/registrar` | Registro de cliente (Firebase) |
| GET | `/clientes/` | Lista clientes (staff) |
| GET | `/clientes/perfil` | Perfil del cliente autenticado |
| PUT | `/clientes/perfil` | Actualizar perfil |
| GET | `/clientes/por-uid/{uid}` | Buscar cliente por Firebase UID |

### Asistente IA — `/asistente`

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/asistente/consultar` | Chat con el asistente (nutrición, rutinas, progreso) |
| POST | `/asistente/log-inteligente` | Registrar comida o ejercicio por texto/NLP |
| POST | `/asistente/log-manual` | Registro manual con etiqueta nutricional |
| POST | `/asistente/confirmar-registro` | Confirmar registro desde tarjeta del chat |
| POST | `/asistente/calcular-ejercicio` | Calcular kcal de un ejercicio sin registrar |
| POST | `/asistente/guardar-sugerencia` | Guardar receta/rutina sugerida |
| GET | `/asistente/mis-sugerencias` | Listar recetas/rutinas guardadas |
| DELETE | `/asistente/sugerencia/{id}` | Eliminar sugerencia guardada |

### Ejercicios — `/ejercicios`

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/ejercicios/` | Catálogo con filtros (grupo, nivel, metrica) |
| GET | `/ejercicios/grupos` | Grupos musculares con conteo |
| POST | `/ejercicios/rutina` | Generar rutina adaptativa (ML) |
| POST | `/ejercicios/log-series` | Registrar series/reps/peso |
| GET | `/ejercicios/logs` | Historial de workout_logs del usuario |

### Balance — `/balance`

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/balance/hoy` | Consumo y quemado del día actual |
| DELETE | `/balance/registro/{id}` | Eliminar registro de comida |

### Dashboard — `/dashboard`

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/dashboard/clientes/{id}/resumen-diario` | Resumen calórico del día |
| GET | `/dashboard/clientes/{id}/calorias-tendencia` | Tendencia últimos 7/30 días |
| GET | `/dashboard/clientes/{id}/peso-historial` | Historial de peso |
| GET | `/dashboard/clientes/{id}/imc-historial` | Historial de IMC |
| GET | `/dashboard/clientes/{id}/analisis-ia` | Análisis ML del perfil |

### Nutrición — `/nutricion`

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/nutricion/` | Plan nutricional del usuario |
| GET | `/nutricion/recomendaciones` | Recomendaciones KNN personalizadas |

### Nutricionista — `/nutricionista`

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/nutricionista/clientes` | Lista de clientes asignados |
| GET | `/nutricionista/stats` | Estadísticas globales |
| GET | `/nutricionista/cliente/{id}/progreso` | Progreso del cliente |
| GET | `/nutricionista/cliente/{id}/plan` | Plan nutricional del cliente |
| PUT | `/nutricionista/cliente/{id}/plan` | Actualizar plan |
| POST | `/nutricionista/validar-plan/{id}` | Validar plan nutricional |

### Copiloto Staff — `/copiloto`

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/copiloto/consultar` | Chat del asistente para staff |

### Alertas — `/alertas`

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/alertas/mis-clientes` | Alertas de salud de los clientes |

### Notificaciones — `/notifications`

| Método | Ruta | Descripción |
|--------|------|-------------|
| — | `/notifications/*` | Notificaciones push (FCM) |

### API v1 — `/api/v1`

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/v1/nutrition/parse_ingredients` | Parsear ingredientes por texto |

---

## Base de datos

### Conexión directa

```bash
docker exec -it calofit_db psql -U postgres -d BD_Calofit
```

### Tablas principales

| Tabla | Descripción |
|-------|-------------|
| `clients` | Clientes del gimnasio |
| `users` | Staff (nutricionistas, admin) |
| `alimentos` | 734+ alimentos con macros/100g |
| `platos` | Platos con ingredientes relacionados |
| `plato_ingredientes` | Relación platos ↔ alimentos con gramos |
| `ejercicios` | 139 ejercicios con MET y grupos musculares |
| `progreso_calorias` | Consumo diario por cliente |
| `workout_logs` | Sesiones de ejercicio registradas |
| `planes_nutricionales` | Planes asignados por nutricionista |
| `planes_diarios` | Plan por día de la semana |
| `historial_peso` | Evolución del peso del cliente |
| `alertas_salud` | Alertas generadas por el sistema ML |
| `sugerencias_guardadas` | Recetas/rutinas guardadas del chat |

### Calcular macros de un plato

Los macros **nunca** se almacenan en `platos` — siempre se calculan en tiempo real:

```sql
SELECT
  p.nombre,
  SUM(a.calorias_100g  * pi.gramos / 100) AS kcal,
  SUM(a.proteina_100g  * pi.gramos / 100) AS proteinas_g,
  SUM(a.carbohidrato_100g * pi.gramos / 100) AS carbs_g,
  SUM(a.grasa_100g     * pi.gramos / 100) AS grasas_g
FROM platos p
JOIN plato_ingredientes pi ON pi.plato_id = p.id
JOIN alimentos a ON a.id = pi.alimento_id
WHERE p.id = 28
GROUP BY p.nombre;
```

### Regenerar el backup completo

```bash
docker exec calofit_db pg_dump -U postgres -d BD_Calofit > ../backups/calofit_backup.sql
```

---

## Comandos útiles

```bash
# Logs del backend en tiempo real
docker logs calofit_backend -f --tail=50

# Reiniciar solo el backend (tras cambios de código)
docker-compose restart backend

# Acceder a la BD
docker exec -it calofit_db psql -U postgres -d BD_Calofit

# Ejecutar tests de integración
docker exec calofit_backend python -m pytest tests/integration/ -v

# QA del motor de alimentos
docker exec calofit_backend python scripts/qa_asistente_alimentos.py

# Verificar coherencia de platos LLM
docker exec calofit_backend python scripts/verificar_coherencia_platos_llm.py

# Entrenar modelo Random Forest (cuando haya ≥100 usuarios reales)
docker exec calofit_backend python scripts/entrenar_perfil_adherencia.py

# Evaluar métricas RF + KNN
docker exec calofit_backend python scripts/evaluar_modelos_ml.py
```

---

## Credenciales de prueba

| Perfil | IDs | Contraseña | Características |
|--------|-----|------------|-----------------|
| A — Disciplinado | 55–59 | `CaloFit2024!` | Adherencia ≥90%, ejercicio 4-5x/sem |
| B — Intermedio | 60–64 | `CaloFit2024!` | Adherencia ~60%, ejercicio 2-3x/sem |
| C — Crítico | 65–69 | `CaloFit2024!` | Exceso carbs/grasas, sedentarios |

```sql
SELECT id, first_name, email FROM clients WHERE id BETWEEN 55 AND 69;
```

---

## Logs de errores

Los errores WARNING+ se guardan automáticamente en `logs/error.log`.
Rotación: 5 MB por archivo, 3 backups. Configurado en `app/core/logging_config.py`.
