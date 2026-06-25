"""
Configuración global de pytest y fixtures.
Auto-detecta entorno Docker o local.
"""
import os
import pytest
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def _get_test_db_url() -> str:
    """
    Determina la URL de BD de test según el entorno.
    - Si TEST_DATABASE_URL está seteada, la usa.
    - Si DATABASE_URL tiene host 'db' (Docker), reemplaza BD por test_calofit.
    - Si no, usa localhost:5433 (Windows local).
    """
    if os.getenv("TEST_DATABASE_URL"):
        return os.getenv("TEST_DATABASE_URL")

    db_url = os.getenv("DATABASE_URL", "")

    if "db:5432" in db_url:
        # Entorno Docker: host=db, puerto=5432
        return "postgresql://postgres:leomeflo09@db:5432/test_calofit"

    # Entorno Windows local con Docker en puerto 5433
    return "postgresql://postgres:leomeflo09@localhost:5433/test_calofit"


# ─────────────────────────────────────────────────────────────────────────────
# Engine y sesión — se crean de forma lazy dentro de fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_db_url():
    """URL de BD de test calculada una sola vez por sesión."""
    return _get_test_db_url()


@pytest.fixture(scope="session")
def test_engine(test_db_url):
    """Engine de SQLAlchemy para tests (sesión completa)."""
    from sqlalchemy import create_engine, text
    from app.core.database import Base

    engine = create_engine(test_db_url, echo=False, pool_pre_ping=True)
    logger.info(f"[TEST] Conectando a BD de test: {test_db_url}")

    # BD_Calofit (producción/desarrollo real) ya tiene estas extensiones
    # activas; test_calofit se crea desde cero y no las trae — sin esto,
    # cualquier query que use unaccent()/similaridad de texto (varias en
    # app/services/) falla con UndefinedFunction y aborta la transacción.
    with engine.connect() as _conn_ext:
        _conn_ext.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
        _conn_ext.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        _conn_ext.commit()

    # Crear todas las tablas
    Base.metadata.create_all(bind=engine)
    logger.info("[TEST] Tablas creadas correctamente.")

    yield engine

    # Limpieza al final de la sesión
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    logger.info("[TEST] BD de test limpiada.")


@pytest.fixture(scope="session")
def TestingSessionLocal(test_engine):
    """Sessionmaker vinculado al engine de test."""
    from sqlalchemy.orm import sessionmaker
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture
def db(test_engine, TestingSessionLocal):
    """
    Fixture de sesión de BD con rollback automático por test.
    Cada test opera en su propia transacción que se revierte al final.
    """
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


# ─────────────────────────────────────────────────────────────────────────────
# Aislamiento de estado global entre tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _limpiar_macro_cache():
    """_macro_cache (llm_registro.py) es un dict en memoria de proceso, sin
    relación con la BD de test ni con el rollback por test — un alimento
    cacheado por un test con Groq real (ej. "arroz", "huevos") sobrevive y
    contamina la extracción de otro test que use el mismo nombre, haciendo
    que el resultado dependa del ORDEN de ejecución de la suite. Se limpia
    antes y después de cada test para que cada uno empiece en blanco."""
    from app.services.llm_registro import _macro_cache
    _macro_cache.clear()
    yield
    _macro_cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures de datos de prueba
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_role(db):
    """Crea rol de prueba."""
    from app.models import Role
    role = Role(name="Nutricionista", description="Nutricionista de prueba")
    db.add(role)
    db.commit()
    db.refresh(role)
    return role


@pytest.fixture
def sample_user(db, sample_role):
    """Crea usuario de prueba."""
    from app.models import User
    user = User(
        first_name="Test",
        last_name_paternal="User",
        last_name_maternal="Demo",
        email=f"test_{datetime.utcnow().timestamp()}@test.com",
        hashed_password="hashedpwd123",
        role_id=sample_role.id,
        role_name="Nutricionista",
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture
def sample_client(db):
    """Crea cliente de prueba."""
    from app.models import Client
    client = Client(
        first_name="Cliente",
        last_name_paternal="Test",
        last_name_maternal="Demo",
        dni=f"9{str(int(datetime.utcnow().timestamp()))[-7:]}",
        email=f"client_{datetime.utcnow().timestamp()}@test.com",
        hashed_password="hashedpwd123",
        birth_date=datetime(1990, 1, 1).date(),
        weight=70.0,
        height=175,
        gender="M",
        medical_conditions=[],
        activity_level="moderado",
        goal="perdida_de_peso",
        is_profile_complete=True,
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    return client


@pytest.fixture
def sample_alimentos(db):
    """Crea ingredientes de prueba."""
    from app.models import Alimento
    alimentos = [
        Alimento(
            nombre="Arroz blanco cocido",
            nombre_normalizado="arroz blanco cocido",
            calorias_100g=130.0,
            proteina_100g=2.7,
            carbohidratos_100g=28.0,
            grasas_100g=0.3,
            fibra_100g=0.4,
            azucar_100g=0.0,
            categoria="granos",
            fuente="BD",
        ),
        Alimento(
            nombre="Pollo pechuga cocida",
            nombre_normalizado="pollo pechuga cocida",
            calorias_100g=165.0,
            proteina_100g=31.0,
            carbohidratos_100g=0.0,
            grasas_100g=3.6,
            fibra_100g=0.0,
            azucar_100g=0.0,
            categoria="proteina",
            fuente="BD",
        ),
        Alimento(
            nombre="Brocoli cocido",
            nombre_normalizado="brocoli cocido",
            calorias_100g=34.0,
            proteina_100g=2.8,
            carbohidratos_100g=7.0,
            grasas_100g=0.4,
            fibra_100g=2.4,
            azucar_100g=1.4,
            categoria="verdura",
            fuente="BD",
        ),
    ]
    db.add_all(alimentos)
    db.commit()
    for ali in alimentos:
        db.refresh(ali)
    return alimentos


@pytest.fixture
def sample_plato(db, sample_alimentos):
    """Crea plato de prueba con ingredientes."""
    from app.models import Plato, PlatoIngrediente
    plato = Plato(
        nombre="Arroz con pollo y brocoli",
        nombre_normalizado="arroz con pollo y brocoli",
        tipo_plato="almuerzo",
        origen="manual",
    )
    db.add(plato)
    db.flush()

    porciones = [200, 150, 100]
    for idx, (alimento, gramos) in enumerate(zip(sample_alimentos, porciones), 1):
        ing = PlatoIngrediente(
            plato_id=plato.id,
            alimento_id=alimento.id,
            gramos=gramos,
            orden=idx,
        )
        db.add(ing)

    db.commit()
    db.refresh(plato)
    return plato


@pytest.fixture
def sample_ejercicios(db):
    """Crea ejercicios de prueba."""
    from app.models import Ejercicio
    ejercicios = [
        Ejercicio(
            nombre="Sentadilla",
            nombre_normalizado="sentadilla",
            alias=["squat"],
            descripcion="Sentadilla con peso",
            met=5.0,
            grupo_muscular="piernas",
            origen="gold_standard",
        ),
        Ejercicio(
            nombre="Press de banca",
            nombre_normalizado="press de banca",
            alias=["bench press"],
            descripcion="Press de banca con barra",
            met=6.0,
            grupo_muscular="pecho",
            origen="gold_standard",
        ),
        Ejercicio(
            nombre="Curl de biceps",
            nombre_normalizado="curl de biceps",
            alias=["bicep curl"],
            descripcion="Curl de biceps con mancuernas",
            met=3.5,
            grupo_muscular="brazos",
            origen="gold_standard",
        ),
    ]
    db.add_all(ejercicios)
    db.commit()
    for ej in ejercicios:
        db.refresh(ej)
    return ejercicios


@pytest.fixture
def sample_meta_usuario(db, sample_client):
    """Crea meta nutricional de prueba."""
    from app.models import MetaUsuario
    meta = MetaUsuario(
        client_id=sample_client.id,
        genero="M",
        edad=34,
        peso_kg=70,
        talla_cm=175,
        nivel_actividad="Moderado",
        objetivo="perdida_de_peso",
        tmb=1700,
        get=2635,
        calorias_objetivo=2100,
        proteinas_g=150,
        carbohidratos_g=260,
        grasas_g=70,
    )
    db.add(meta)
    db.commit()
    db.refresh(meta)
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Mocks de servicios externos
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_service():
    """Mock de LLMService para tests sin IA real."""
    class MockLLMService:
        def generar_propuesta_plato(self, contexto, cantidad=3):
            return [
                "Arroz con pollo y brocoli",
                "Tiradito de atun",
                "Ensalada de quinoa",
            ][:cantidad]

        def generar_propuesta_rutina(self, contexto, cantidad=2):
            return [
                {"nombre": "Rutina 1", "descripcion": "Brazos"},
                {"nombre": "Rutina 2", "descripcion": "Piernas"},
            ][:cantidad]

        def chat_conversacional(self, mensaje_usuario, historial=None, sistema_prompt=None):
            return "Respuesta simulada del LLM"

        def extraer_entidades(self, texto, tipo_entidad="ingredientes"):
            if "pollo" in texto.lower():
                return ["pollo"]
            return []

    return MockLLMService()


@pytest.fixture
def mock_usda_client():
    """Mock de USDA Client (siempre retorna None → fallback a BD)."""
    class MockUSDAClient:
        def buscar_alimento(self, nombre):
            return None

        def obtener_macros(self, food_id):
            return None

    return MockUSDAClient()


@pytest.fixture
def mock_fatsecret_client():
    """Mock de FatSecret Client (siempre retorna None → fallback a BD)."""
    class MockFatSecretClient:
        def buscar_alimento(self, nombre):
            return None

        def obtener_macros(self, food_id):
            return None

    return MockFatSecretClient()


# ─────────────────────────────────────────────────────────────────────────────
# Configuración de markers
# ─────────────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    """Registra markers personalizados."""
    config.addinivalue_line("markers", "unit: test unitario (sin BD ni red)")
    config.addinivalue_line("markers", "integration: test de integración (con BD)")
    config.addinivalue_line("markers", "e2e: test end-to-end (con API completa)")
    config.addinivalue_line("markers", "adversarial: caso adversarial / caso límite")
    config.addinivalue_line("markers", "slow: test lento (>10s)")
