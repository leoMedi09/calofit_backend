-- Migración 004: Catálogo de Platos Peruanos (recetas simples)
-- Fecha: 2026-04-26
-- Descripción: Tabla `platos_peruanos` para recomendaciones peruanas consistentes

-- Nota: en algunos entornos esta tabla ya existe (schema base) con columnas:
--   nombre, momento, nivel, tags, ingredientes_base, preparacion_base, activo, nota...
-- Esta migración crea la tabla SOLO si no existe; la extensión vive en 005_*.sql

CREATE TABLE IF NOT EXISTS platos_peruanos (
    id SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    momento VARCHAR(32) NOT NULL DEFAULT 'otro',      -- desayuno|snack|almuerzo|cena|nocturno|otro
    nivel VARCHAR(16) NOT NULL DEFAULT 'normal',      -- ligero|normal|contundente
    tags JSONB NOT NULL DEFAULT '{}'::jsonb,
    ingredientes_base JSONB NOT NULL DEFAULT '[]'::jsonb,
    preparacion_base JSONB NOT NULL DEFAULT '[]'::jsonb,
    nota TEXT NULL,
    activo BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_platos_peruanos_momento ON platos_peruanos(momento);
CREATE INDEX IF NOT EXISTS idx_platos_peruanos_nivel ON platos_peruanos(nivel);

SELECT 'Tabla platos_peruanos creada/asegurada (schema base)' AS status;

