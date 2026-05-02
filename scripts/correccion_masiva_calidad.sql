-- ===========================================================================
-- Corrección Masiva de Calidad — CaloFit (2026-05-01)
-- Ortografía, autenticidad lambayecana, poda kcal fantasma, rangos horarios
-- Ejecutar: docker exec -i calofit_db psql -U postgres -d BD_Calofit < scripts/correccion_masiva_calidad.sql
-- ===========================================================================

BEGIN;

-- ===========================================================================
-- BLOQUE 1: Ortografía — Ceviche → Cebiche
-- ===========================================================================
UPDATE platos
SET nombre            = 'Cebiche De Pescado',
    nombre_normalizado = 'cebiche de pescado',
    tipo_plato         = 'almuerzo'
WHERE id = 28;

-- ===========================================================================
-- BLOQUE 2: Cebiche id=28 — Autenticidad Lambayecana
-- Quitar aceite de oliva (no existe en cebiche tradicional)
-- Subir cebolla a 60g (juliana), ají amarillo a 15g
-- Añadir limón (marinado) y sal
-- ===========================================================================

-- Quitar aceite de oliva
DELETE FROM plato_ingredientes
WHERE plato_id = 28 AND alimento_id = 52;

-- Ajustar cebolla
UPDATE plato_ingredientes
SET gramos = 60
WHERE plato_id = 28 AND alimento_id = 38;

-- Ajustar ají amarillo
UPDATE plato_ingredientes
SET gramos = 15
WHERE plato_id = 28 AND alimento_id = 86;

-- Añadir jugo de limón (marinado del cebiche) si no existe
INSERT INTO plato_ingredientes (plato_id, alimento_id, gramos, orden)
SELECT 28, 426, 50, 10
WHERE NOT EXISTS (
    SELECT 1 FROM plato_ingredientes WHERE plato_id = 28 AND alimento_id = 426
);

-- Añadir sal si no existe
INSERT INTO plato_ingredientes (plato_id, alimento_id, gramos, orden)
SELECT 28, 60, 3, 11
WHERE NOT EXISTS (
    SELECT 1 FROM plato_ingredientes WHERE plato_id = 28 AND alimento_id = 60
);

-- ===========================================================================
-- BLOQUE 3: Cebiche de pescado con aguacate (id=176) — Corrección total
-- Nombre honesto, quitar zanahoria / tomate / jengibre (no tradicionales),
-- reducir palta a máx 30g (acompañamiento), añadir limón y sal
-- ===========================================================================
UPDATE platos
SET nombre             = 'Cebiche de Pescado con Palta',
    nombre_normalizado  = 'cebiche de pescado con palta',
    tipo_plato          = 'almuerzo'
WHERE id = 176;

-- Quitar zanahoria (no tradicional en cebiche)
DELETE FROM plato_ingredientes WHERE plato_id = 176 AND alimento_id = 39;

-- Quitar tomate (no tradicional en cebiche lambayecano)
DELETE FROM plato_ingredientes WHERE plato_id = 176 AND alimento_id = 705;

-- Quitar jengibre (fusión asiática — no aplica aquí)
DELETE FROM plato_ingredientes WHERE plato_id = 176 AND alimento_id = 767;

-- Reducir palta a 30g (acompañamiento, no protagonista)
UPDATE plato_ingredientes
SET gramos = 30
WHERE plato_id = 176 AND alimento_id = 33;

-- Ajustar cebolla a 60g
UPDATE plato_ingredientes
SET gramos = 60
WHERE plato_id = 176 AND alimento_id = 38;

-- Ajustar ají amarillo a 15g
UPDATE plato_ingredientes
SET gramos = 15
WHERE plato_id = 176 AND alimento_id = 86;

-- Añadir jugo de limón
INSERT INTO plato_ingredientes (plato_id, alimento_id, gramos, orden)
SELECT 176, 426, 50, 10
WHERE NOT EXISTS (
    SELECT 1 FROM plato_ingredientes WHERE plato_id = 176 AND alimento_id = 426
);

-- Añadir sal
INSERT INTO plato_ingredientes (plato_id, alimento_id, gramos, orden)
SELECT 176, 60, 3, 11
WHERE NOT EXISTS (
    SELECT 1 FROM plato_ingredientes WHERE plato_id = 176 AND alimento_id = 60
);

-- ===========================================================================
-- BLOQUE 4: Pan con queso fresco y tomate (id=177) — Poda kcal fantasma
-- Pan 150g→70g, queso 120g→50g, eliminar aceite de oliva
-- Target: ~370 kcal (rango desayuno 300-500)
-- ===========================================================================
UPDATE platos
SET tipo_plato = 'desayuno'
WHERE id = 177;

UPDATE plato_ingredientes SET gramos = 70  WHERE plato_id = 177 AND alimento_id = 21;  -- Pan Integral
UPDATE plato_ingredientes SET gramos = 50  WHERE plato_id = 177 AND alimento_id = 15;  -- Queso Fresco
DELETE FROM plato_ingredientes               WHERE plato_id = 177 AND alimento_id = 52; -- Aceite De Oliva

-- ===========================================================================
-- BLOQUE 5: Pan con jamón y queso (id=178) — Poda masiva + renombrado
-- Pan 150g→70g, jamón 120g→40g, queso 100g→30g, aceite→eliminar, aceitunas 50g→20g
-- Target: ~497 kcal (rango desayuno)
-- ===========================================================================
UPDATE platos
SET nombre             = 'Sándwich de Jamón y Queso',
    nombre_normalizado  = 'sandwich de jamon y queso',
    tipo_plato          = 'desayuno'
WHERE id = 178;

UPDATE plato_ingredientes SET gramos = 70  WHERE plato_id = 178 AND alimento_id = 21;   -- Pan Integral
UPDATE plato_ingredientes SET gramos = 40  WHERE plato_id = 178 AND alimento_id = 389;  -- Jamonada
UPDATE plato_ingredientes SET gramos = 30  WHERE plato_id = 178 AND alimento_id = 15;   -- Queso Fresco
UPDATE plato_ingredientes SET gramos = 20  WHERE plato_id = 178 AND alimento_id = 101;  -- Aceitunas
DELETE FROM plato_ingredientes               WHERE plato_id = 178 AND alimento_id = 52; -- Aceite De Oliva

-- ===========================================================================
-- BLOQUE 6: Tostada de pan con atún (id=180) — Renombrado + doble pan eliminado
-- Pan 120g→60g, eliminar Pan Bollo (pan doble), atún 150g→80g, mayonesa 50g→15g
-- Eliminar Maíz Amarillo (no aplica en tostada)
-- Target: ~318 kcal (rango desayuno)
-- ===========================================================================
UPDATE platos
SET nombre             = 'Pan Tostado con Atún',
    nombre_normalizado  = 'pan tostado con atun',
    tipo_plato          = 'desayuno'
WHERE id = 180;

UPDATE plato_ingredientes SET gramos = 60  WHERE plato_id = 180 AND alimento_id = 21;  -- Pan Integral
UPDATE plato_ingredientes SET gramos = 80  WHERE plato_id = 180 AND alimento_id = 8;   -- Atun En Agua
UPDATE plato_ingredientes SET gramos = 15  WHERE plato_id = 180 AND alimento_id = 54;  -- Mayonesa
DELETE FROM plato_ingredientes               WHERE plato_id = 180 AND alimento_id = 530; -- Pan Bollo (redundante)
DELETE FROM plato_ingredientes               WHERE plato_id = 180 AND alimento_id = 445; -- Maíz Amarillo

-- ===========================================================================
-- BLOQUE 7: Eliminar platos LLM fantasma (sin ingredientes)
-- IDs 179 y 181: duplicados sin datos, creados por LLM sin resolución
-- ===========================================================================
DELETE FROM platos WHERE id IN (179, 181);

-- ===========================================================================
-- VERIFICACIÓN FINAL
-- ===========================================================================
SELECT
    p.id,
    p.nombre,
    p.tipo_plato,
    ROUND(SUM(a.calorias_100g * pi2.gramos / 100.0)::numeric, 1) AS kcal_recalc,
    SUM(pi2.gramos) AS gramos_total,
    COUNT(pi2.id) AS n_ingredientes
FROM platos p
JOIN plato_ingredientes pi2 ON pi2.plato_id = p.id
JOIN alimentos a ON a.id = pi2.alimento_id
WHERE p.id IN (28, 176, 177, 178, 180)
GROUP BY p.id, p.nombre, p.tipo_plato
ORDER BY p.id;

COMMIT;
