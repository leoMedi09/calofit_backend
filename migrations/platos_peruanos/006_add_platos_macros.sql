-- Migración 006: Macros numéricos en `platos_peruanos`
-- Fecha: 2026-04-26
-- Descripción: Guarda kcal/prote/carb/grasas por plato (porción sugerida) para mostrar chips estables.

ALTER TABLE platos_peruanos
  ADD COLUMN IF NOT EXISTS kcal_aprox DOUBLE PRECISION NULL;

ALTER TABLE platos_peruanos
  ADD COLUMN IF NOT EXISTS proteinas_g DOUBLE PRECISION NULL;

ALTER TABLE platos_peruanos
  ADD COLUMN IF NOT EXISTS carbohidratos_g DOUBLE PRECISION NULL;

ALTER TABLE platos_peruanos
  ADD COLUMN IF NOT EXISTS grasas_g DOUBLE PRECISION NULL;

ALTER TABLE platos_peruanos
  ADD COLUMN IF NOT EXISTS macros_fuente VARCHAR(64) NULL;

SELECT 'platos_peruanos: columnas de macros añadidas' AS status;

