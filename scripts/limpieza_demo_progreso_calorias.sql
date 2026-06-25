-- ============================================================================
-- Limpieza de datos de demo — progreso_calorias inflado
-- ============================================================================
-- Contexto: se detectaron filas con calorias_consumidas > 5000 (umbral muy por
-- encima de cualquier ingesta real, incluso en superávit agresivo) en la BD
-- local de Docker (BD_Calofit), originadas por pruebas repetidas sobre la
-- misma fecha durante el desarrollo. No es un bug del algoritmo de balance
-- (ver auditoría previa) — es contaminación de datos de prueba.
--
-- Alcance: SOLO la tabla progreso_calorias. No toca estructura, modelos,
-- migraciones ni ninguna otra tabla. No afecta a clients/usuarios reales.
--
-- Reversibilidad: se crea una tabla de respaldo ANTES del DELETE. Mientras
-- no se ejecute el COMMIT final, todo el bloque puede revertirse con
-- ROLLBACK. Después del COMMIT, el respaldo permite restaurar manualmente
-- (ver bloque de reversión al final).
--
-- Cómo ejecutar:
--   docker exec -i calofit_db psql -U postgres -d BD_Calofit < scripts/limpieza_demo_progreso_calorias.sql
-- ============================================================================

BEGIN;

-- 1) Respaldo de las filas candidatas a borrar (reversibilidad real)
DROP TABLE IF EXISTS progreso_calorias_backup_demo;
CREATE TABLE progreso_calorias_backup_demo AS
SELECT *
FROM progreso_calorias
WHERE calorias_consumidas > 5000;

-- 2) Vista previa — REVISAR esta salida antes de seguir.
--    Si algo aquí parece un dato real (no de prueba), detenerse y hacer
--    ROLLBACK en vez de continuar.
SELECT client_id, fecha, calorias_consumidas, calorias_quemadas
FROM progreso_calorias
WHERE calorias_consumidas > 5000
ORDER BY client_id, fecha;

-- 3) Borrado controlado — mismo criterio que la vista previa de arriba.
DELETE FROM progreso_calorias
WHERE calorias_consumidas > 5000;

-- 4) Verificación post-borrado (debe devolver 0).
SELECT count(*) AS filas_restantes_infladas
FROM progreso_calorias
WHERE calorias_consumidas > 5000;

-- 5) Confirmar o revertir:
--    - Si la verificación del paso 4 dio 0 y la vista previa del paso 2 se
--      veía correcta -> dejar el COMMIT de abajo.
--    - Si algo no cuadra -> comentar el COMMIT y descomentar el ROLLBACK.
COMMIT;
-- ROLLBACK;

-- ============================================================================
-- REVERSIÓN MANUAL (solo si ya se hizo COMMIT y se necesita deshacer):
--
-- INSERT INTO progreso_calorias
-- SELECT b.* FROM progreso_calorias_backup_demo b
-- WHERE NOT EXISTS (
--     SELECT 1 FROM progreso_calorias p
--     WHERE p.client_id = b.client_id AND p.fecha = b.fecha
-- );
--
-- Limpiar el respaldo cuando ya no se necesite:
-- DROP TABLE IF EXISTS progreso_calorias_backup_demo;
-- ============================================================================
