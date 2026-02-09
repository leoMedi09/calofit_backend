-- Migración 003: Sistema de Aprendizaje de Preferencias
-- Fecha: 2026-01-24
-- Descripción: Tablas para almacenar preferencias de alimentos y ejercicios

-- 1. Tabla de preferencias de alimentos
CREATE TABLE IF NOT EXISTS preferencias_alimentos (
    id SERIAL PRIMARY KEY,
    client_id INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    alimento VARCHAR(200) NOT NULL,
    frecuencia INTEGER DEFAULT 1,
    puntuacion FLOAT DEFAULT 1.0,
    ultima_vez TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(client_id, alimento)  -- Un cliente no puede tener duplicados del mismo alimento
);

-- 2. Tabla de preferencias de ejercicios
CREATE TABLE IF NOT EXISTS preferencias_ejercicios (
    id SERIAL PRIMARY KEY,
    client_id INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    ejercicio VARCHAR(200) NOT NULL,
    frecuencia INTEGER DEFAULT 1,
    puntuacion FLOAT DEFAULT 1.0,
    ultima_vez TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(client_id, ejercicio)
);

-- 3. Índices para búsquedas rápidas
CREATE INDEX IF NOT EXISTS idx_pref_alimentos_client ON preferencias_alimentos(client_id);
CREATE INDEX IF NOT EXISTS idx_pref_alimentos_frecuencia ON preferencias_alimentos(frecuencia DESC);
CREATE INDEX IF NOT EXISTS idx_pref_ejercicios_client ON preferencias_ejercicios(client_id);
CREATE INDEX IF NOT EXISTS idx_pref_ejercicios_frecuencia ON preferencias_ejercicios(frecuencia DESC);

-- 4. Comentarios explicativos
COMMENT ON TABLE preferencias_alimentos IS 'Almacena las preferencias alimenticias de cada cliente basadas en su historial de consumo';
COMMENT ON COLUMN preferencias_alimentos.frecuencia IS 'Número de veces que el cliente ha consumido este alimento';
COMMENT ON COLUMN preferencias_alimentos.puntuacion IS 'Score de preferencia (1.0 = normal, >1.0 = le gusta más)';

-- Verificación
SELECT 'Tablas de preferencias creadas exitosamente' AS status;
