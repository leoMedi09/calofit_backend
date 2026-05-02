-- Migración 005: Extender `platos_peruanos` con normalización y alias
-- Fecha: 2026-04-26
-- Descripción: Agrega `nombre_normalizado` y `alias` para búsquedas/variación.

ALTER TABLE platos_peruanos
  ADD COLUMN IF NOT EXISTS nombre_normalizado VARCHAR(255);

ALTER TABLE platos_peruanos
  ADD COLUMN IF NOT EXISTS alias JSONB NULL;

-- Índice único para evitar duplicados de nombres normalizados
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes
    WHERE schemaname = 'public'
      AND indexname = 'uq_platos_peruanos_nombre_norm'
  ) THEN
    CREATE UNIQUE INDEX uq_platos_peruanos_nombre_norm ON platos_peruanos(nombre_normalizado);
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_platos_peruanos_momento ON platos_peruanos(momento);

SELECT 'platos_peruanos extendida (nombre_normalizado, alias)' AS status;

