-- ============================================
-- Migración 001: Agregar campos de validación
-- Fecha: 2026-01-20
-- Descripción: Agrega campos para el flujo de validación de planes nutricionales
--              y crea la tabla de alertas de salud
-- ============================================

-- 1. Agregar campos de validación a planes_nutricionales
ALTER TABLE planes_nutricionales 
ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'draft_ia',
ADD COLUMN IF NOT EXISTS validated_by_id INTEGER REFERENCES users(id),
ADD COLUMN IF NOT EXISTS validated_at TIMESTAMP;

-- 2. Crear tabla de alertas de salud
CREATE TABLE IF NOT EXISTS alertas_salud (
    id SERIAL PRIMARY KEY,
    client_id INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    tipo VARCHAR(50) NOT NULL,
    descripcion TEXT NOT NULL,
    severidad VARCHAR(20) DEFAULT 'bajo',
    estado VARCHAR(20) DEFAULT 'pendiente',
    atendido_por_id INTEGER REFERENCES users(id),
    fecha_deteccion TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- 3. Crear índices para mejorar el rendimiento
CREATE INDEX IF NOT EXISTS idx_planes_status ON planes_nutricionales(status);
CREATE INDEX IF NOT EXISTS idx_alertas_cliente ON alertas_salud(client_id);
CREATE INDEX IF NOT EXISTS idx_alertas_estado ON alertas_salud(estado);

-- 4. Comentarios para documentación
COMMENT ON COLUMN planes_nutricionales.status IS 'Estado del plan: draft_ia, validado, archivado';
COMMENT ON COLUMN planes_nutricionales.validated_by_id IS 'ID del nutricionista que validó el plan';
COMMENT ON COLUMN planes_nutricionales.validated_at IS 'Fecha y hora de validación del plan';
COMMENT ON TABLE alertas_salud IS 'Registro de alertas de salud detectadas por la IA (fatiga, lesiones, desánimo)';

-- 5. Verificación
SELECT 'Migración 001 completada exitosamente' AS resultado;
