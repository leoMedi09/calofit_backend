-- Migración 002: Agregar campos de gestión a alertas_salud
-- Fecha: 2026-01-21
-- Descripción: Agrega columnas para notas y fecha de atención

-- 1. Agregar columna de notas para que el staff agregue comentarios
ALTER TABLE alertas_salud 
ADD COLUMN IF NOT EXISTS notas TEXT;

-- 2. Agregar columna para registrar cuándo se atendió la alerta
ALTER TABLE alertas_salud 
ADD COLUMN IF NOT EXISTS fecha_atencion TIMESTAMP;

-- 3. Actualizar estados permitidos (agregar 'en_proceso')
-- Nota: PostgreSQL no tiene CHECK constraints fácilmente modificables,
-- pero la validación se hará a nivel de aplicación

-- 4. Crear índice para mejorar búsquedas por atendido_por_id
CREATE INDEX IF NOT EXISTS idx_alertas_atendido_por 
ON alertas_salud(atendido_por_id);

-- Verificación
SELECT 
    column_name, 
    data_type, 
    is_nullable
FROM information_schema.columns
WHERE table_name = 'alertas_salud'
ORDER BY ordinal_position;
