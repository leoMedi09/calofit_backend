-- Migration 008 — Campos Entrenador Adaptativo
-- Añade tipo_metrica y grupo_padre a ejercicios
-- Añade calorias_quemadas, session_duration_min, intensity a workout_logs

ALTER TABLE ejercicios
    ADD COLUMN IF NOT EXISTS tipo_metrica   VARCHAR(50) DEFAULT 'peso_reps',
    ADD COLUMN IF NOT EXISTS grupo_padre    VARCHAR(100);

ALTER TABLE workout_logs
    ADD COLUMN IF NOT EXISTS calorias_quemadas    FLOAT,
    ADD COLUMN IF NOT EXISTS session_duration_min FLOAT,
    ADD COLUMN IF NOT EXISTS intensity            VARCHAR(50);

-- Poblar grupo_padre desde musculo_principal para ejercicios existentes
UPDATE ejercicios SET grupo_padre = CASE
    WHEN musculo_principal IN ('Pectorales','Pectoral','Pectoral Mayor','Pecho') THEN 'Pecho'
    WHEN musculo_principal IN ('Dorsales','Dorsal Ancho','Espalda','Espalda Baja','Espalda Media') THEN 'Espalda'
    WHEN musculo_principal IN ('Cuádriceps','Isquiosurales','Piernas','Glúteos','Glúteos Mayor','Pantorrillas','Aductores') THEN 'Piernas'
    WHEN musculo_principal IN ('Deltoides','Deltoides Medio','Hombros') THEN 'Hombros'
    WHEN musculo_principal IN ('Bíceps','Bíceps Braquial','Antebrazos') THEN 'Bíceps'
    WHEN musculo_principal IN ('Tríceps') THEN 'Tríceps'
    WHEN musculo_principal IN ('Abdominales','Core','Abdomen') THEN 'Core'
    WHEN musculo_principal IN ('Cuerpo Completo') THEN 'Cuerpo Completo'
    ELSE 'Cardio'
END
WHERE grupo_padre IS NULL;

-- Poblar tipo_metrica desde tipo para ejercicios existentes
UPDATE ejercicios SET tipo_metrica = CASE
    WHEN tipo IN ('Cardio','Cardio Ligero','Resistencia/Isométrico') THEN 'tiempo'
    WHEN tipo IN ('Fuerza/Autocarga','Pliométricos','Metabólico/HIIT') THEN 'solo_reps'
    ELSE 'peso_reps'
END
WHERE tipo_metrica = 'peso_reps';
