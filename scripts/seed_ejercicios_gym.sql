-- Seed 40 ejercicios variados — CaloFit Gym World Light
-- Cubre grupos subrepresentados: Pecho, Hombros, Tríceps, Glúteos, Funcional
-- Campos nuevos: tipo_metrica, grupo_padre
-- Ejecutar: docker exec -i calofit_db psql -U postgres -d BD_Calofit < scripts/seed_ejercicios_gym.sql

INSERT INTO ejercicios (id, nombre, nombre_normalizado, musculo_principal, tipo, nivel, met, es_cardio, tecnica, tipo_metrica, grupo_padre)
VALUES

-- ══════════════════════════════════════════════════════════════
-- PECHO
-- ══════════════════════════════════════════════════════════════
('press_banca_plano',
 'Press de Banca Plano',
 'press banca plano',
 'Pectoral Mayor', 'Fuerza', 'Intermedio', 5.0, FALSE,
 '1. Acuéstate en el banco con los pies en el suelo y agarra la barra con agarre al ancho de hombros. '
 '2. Baja la barra lentamente hasta rozar el pecho manteniendo los codos a 75°. '
 '3. Empuja la barra hacia arriba extendiendo completamente los brazos sin bloquear los codos. '
 '4. Controla el movimiento en todo momento; no rebootes la barra en el pecho. '
 '5. Completa las repeticiones planificadas y devuelve la barra al soporte con ayuda si es necesario.',
 'peso_reps', 'Pecho'),

('press_banca_inclinado',
 'Press de Banca Inclinado',
 'press banca inclinado',
 'Pectoral Mayor', 'Fuerza', 'Intermedio', 5.0, FALSE,
 '1. Ajusta el banco a 30-45°; acuéstate y sujeta la barra al ancho de hombros. '
 '2. Baja la barra hacia la parte alta del pecho con control. '
 '3. Empuja hacia arriba y ligeramente hacia atrás sin arquear la espalda. '
 '4. Mantén los omóplatos retraídos durante todo el movimiento para proteger el hombro.',
 'peso_reps', 'Pecho'),

('press_banca_declinado',
 'Press de Banca Declinado',
 'press banca declinado',
 'Pectoral Mayor', 'Fuerza', 'Intermedio', 5.0, FALSE,
 '1. Asegura los pies en el soporte y acuéstate en el banco declinado. '
 '2. Sujeta la barra con agarre separado al ancho de hombros y bájala hasta el pecho inferior. '
 '3. Empuja de vuelta a la posición inicial concentrando la contracción en el pectoral inferior. '
 '4. Evita la hiperextensión de muñecas durante el movimiento.',
 'peso_reps', 'Pecho'),

('aperturas_con_mancuernas',
 'Aperturas con Mancuernas (Flyes)',
 'aperturas mancuernas flyes',
 'Pectoral Mayor', 'Hipertrofia', 'Intermedio', 4.5, FALSE,
 '1. Acuéstate en banco plano con una mancuerna en cada mano y brazos extendidos arriba. '
 '2. Baja los brazos en arco amplio hasta que el pecho se estire completamente. '
 '3. Contrae el pecho para subir las mancuernas de vuelta trazando el mismo arco. '
 '4. Mantén una ligera flexión en los codos para no sobrecargar las articulaciones.',
 'peso_reps', 'Pecho'),

('fondos_en_paralelas_pecho',
 'Fondos en Paralelas (Pecho)',
 'fondos paralelas pecho',
 'Pectoral Mayor', 'Fuerza/Autocarga', 'Intermedio', 4.8, FALSE,
 '1. Toma las paralelas con agarre neutro y suspéndete con brazos extendidos. '
 '2. Inclínate ligeramente hacia adelante (≈30°) para activar más el pecho. '
 '3. Baja hasta que los codos estén a 90° o hasta sentir estiramiento en el pectoral. '
 '4. Empuja de vuelta a la posición inicial sin balancear el cuerpo. '
 '5. Para mayor dificultad añade peso con cinturón.',
 'solo_reps', 'Pecho'),

('press_pectoral_maquina',
 'Press Pectoral en Máquina',
 'press pectoral maquina',
 'Pectoral Mayor', 'Hipertrofia', 'Principiante', 4.5, FALSE,
 '1. Ajusta el asiento de la máquina para que los mangos queden al nivel del pecho. '
 '2. Coloca los pies en el suelo, espalda pegada al respaldo y agarra los mangos. '
 '3. Empuja los mangos hacia adelante extendiendo los brazos sin llegar a bloquear. '
 '4. Regresa con control; no sueltes el peso abruptamente. '
 '5. Ideal para principiantes o para añadir volumen al final del entrenamiento.',
 'peso_reps', 'Pecho'),

-- ══════════════════════════════════════════════════════════════
-- ESPALDA
-- ══════════════════════════════════════════════════════════════
('remo_con_barra',
 'Remo con Barra',
 'remo barra',
 'Dorsales', 'Fuerza', 'Intermedio', 5.0, FALSE,
 '1. Párate con los pies al ancho de cadera, dobla rodillas ligeramente y lleva el torso a ~45°. '
 '2. Sujeta la barra con agarre prono al ancho de hombros. '
 '3. Lleva la barra hacia el abdomen inferior apretando los omóplatos al final del movimiento. '
 '4. Baja la barra con control sin que el torso suba o se balancee. '
 '5. Mantén la espalda neutral en todo momento para evitar lesiones lumbares.',
 'peso_reps', 'Espalda'),

('remo_en_polea_baja',
 'Remo en Polea Baja',
 'remo polea baja',
 'Dorsales', 'Hipertrofia', 'Principiante', 4.5, FALSE,
 '1. Siéntate en la máquina de polea baja con las rodillas ligeramente flexionadas. '
 '2. Agarra el aditamento y mantén la espalda recta. '
 '3. Jala el aditamento hacia el abdomen llevando los codos hacia atrás y apretando la espalda. '
 '4. Estira los brazos de vuelta con control para una elongación completa.',
 'peso_reps', 'Espalda'),

('face_pull',
 'Face Pull',
 'face pull',
 'Espalda Media', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Ajusta la polea a la altura de la cara y usa la cuerda de tríceps. '
 '2. Agarra la cuerda con ambas manos, palmas hacia abajo, y da un paso atrás. '
 '3. Jala la cuerda hacia la frente separando las manos al final del movimiento. '
 '4. Aprieta los músculos del manguito rotador y los romboides en la contracción. '
 '5. Excelente para la salud del hombro y la postura; incluirlo en cada sesión de empuje.',
 'peso_reps', 'Espalda'),

-- ══════════════════════════════════════════════════════════════
-- HOMBROS
-- ══════════════════════════════════════════════════════════════
('press_militar_barra',
 'Press Militar con Barra',
 'press militar barra',
 'Deltoides', 'Fuerza', 'Intermedio', 5.0, FALSE,
 '1. Párate con los pies al ancho de cadera, barra sujeta al ancho de hombros delante del pecho. '
 '2. Empuja la barra verticalmente por encima de la cabeza hasta extender los brazos. '
 '3. Baja la barra de vuelta al pecho con control; no arquees la espalda lumbar. '
 '4. Activa el core durante todo el movimiento para estabilizar la columna.',
 'peso_reps', 'Hombros'),

('press_hombros_mancuernas',
 'Press de Hombros con Mancuernas',
 'press hombros mancuernas',
 'Deltoides', 'Hipertrofia', 'Principiante', 4.5, FALSE,
 '1. Siéntate en banco con respaldo vertical, mancuernas a la altura de los hombros. '
 '2. Empuja las mancuernas hacia arriba hasta casi juntar las cabezas arriba. '
 '3. Baja con control hasta la posición inicial; los codos deben quedar a 90° en el punto bajo. '
 '4. Evita encogerte de hombros al subir; mantén los trapecios relajados.',
 'peso_reps', 'Hombros'),

('elevaciones_laterales',
 'Elevaciones Laterales con Mancuernas',
 'elevaciones laterales mancuernas',
 'Deltoides Medio', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Párate con una mancuerna en cada mano, brazos al costado y ligera flexión en codos. '
 '2. Eleva los brazos hacia los lados hasta la altura de los hombros con las palmas hacia abajo. '
 '3. Controla el descenso en 2-3 segundos para maximizar el estímulo en el deltoides medio. '
 '4. No uses impulso con las caderas; si lo haces, reduce el peso.',
 'peso_reps', 'Hombros'),

('elevaciones_frontales',
 'Elevaciones Frontales con Mancuernas',
 'elevaciones frontales mancuernas',
 'Deltoides', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Párate con mancuernas al frente de los muslos, agarre pronado (palmas hacia abajo). '
 '2. Eleva un brazo hacia adelante hasta la altura del hombro, manteniendo el codo ligeramente flexionado. '
 '3. Baja con control y repite con el otro brazo de forma alterna. '
 '4. Evita balancear el torso; si ocurre, usa menos peso.',
 'peso_reps', 'Hombros'),

('pajaros_inclinado',
 'Pájaros (Rear Delt Fly) Inclinado',
 'pajaros rear delt fly inclinado',
 'Deltoides', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Inclínate hacia adelante a 45°, mancuernas colgando con ligera flexión en codos. '
 '2. Eleva los brazos hacia los lados (similar a extensión lateral pero inclinado). '
 '3. Aprieta los romboides y el deltoides posterior en la contracción máxima. '
 '4. Baja lentamente; no uses demasiado peso para no perder la técnica.',
 'peso_reps', 'Hombros'),

('encogimiento_de_hombros',
 'Encogimiento de Hombros (Shrugs)',
 'encogimiento hombros shrugs',
 'Hombros', 'Fuerza', 'Principiante', 4.0, FALSE,
 '1. Párate erguido con mancuernas o barra en los costados, agarre neutro o pronado. '
 '2. Sube los hombros verticalmente lo más alto posible (sin rotar) y mantén 1 segundo. '
 '3. Baja con control para elongar el trapecio. '
 '4. Evita mover la cabeza hacia adelante; mantén el cuello neutral.',
 'peso_reps', 'Hombros'),

-- ══════════════════════════════════════════════════════════════
-- TRÍCEPS
-- ══════════════════════════════════════════════════════════════
('press_frances',
 'Press Francés (Skull Crushers)',
 'press frances skull crushers',
 'Tríceps', 'Hipertrofia', 'Intermedio', 4.5, FALSE,
 '1. Acuéstate en banco plano con una barra Z o mancuernas arriba, brazos extendidos. '
 '2. Dobla los codos hacia la frente bajando el peso cerca de la frente o detrás de la cabeza. '
 '3. Extiende los codos de vuelta sin mover la posición del húmero. '
 '4. Mantén los codos juntos y apuntando al techo durante todo el movimiento.',
 'peso_reps', 'Tríceps'),

('extension_tricep_polea',
 'Extensión de Tríceps en Polea Alta',
 'extension tricep polea alta',
 'Tríceps', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Párate frente a la polea alta con la cuerda de tríceps y sujétala con ambas manos. '
 '2. Con los codos pegados a los costados, extiende los antebrazos hacia abajo separando la cuerda. '
 '3. Aprieta el tríceps en el punto más bajo antes de subir. '
 '4. Sube controlando el movimiento; no dejes que los codos se abran.',
 'peso_reps', 'Tríceps'),

('kickback_tricep',
 'Kickback de Tríceps con Mancuerna',
 'kickback tricep mancuerna',
 'Tríceps', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Con una rodilla y mano apoyadas en el banco, sujeta la mancuerna y lleva el brazo a 90°. '
 '2. Extiende el antebrazo hacia atrás hasta que el brazo quede completamente recto. '
 '3. Aprieta el tríceps al final y regresa con control. '
 '4. Evita balar el peso usando el hombro; el movimiento es solo del codo hacia atrás.',
 'peso_reps', 'Tríceps'),

('fondos_en_banco_tricep',
 'Fondos en Banco (Dips de Tríceps)',
 'fondos banco dips tricep',
 'Tríceps', 'Fuerza/Autocarga', 'Principiante', 4.5, FALSE,
 '1. Apoya las manos en el borde de un banco con los dedos apuntando hacia adelante. '
 '2. Extiende las piernas al frente (o dóblalas para facilitar) y levanta el cuerpo. '
 '3. Baja doblando los codos hasta ~90° sin que los hombros suban a las orejas. '
 '4. Empuja hacia arriba extendiendo los tríceps. '
 '5. Para progresar eleva los pies en un segundo banco.',
 'solo_reps', 'Tríceps'),

-- ══════════════════════════════════════════════════════════════
-- BÍCEPS
-- ══════════════════════════════════════════════════════════════
('curl_concentrado',
 'Curl Concentrado',
 'curl concentrado',
 'Bíceps Braquial', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Siéntate en un banco, apoya el codo en el interior del muslo y sujeta la mancuerna. '
 '2. Curla el peso hacia el hombro contrayendo el bíceps al máximo. '
 '3. Baja con control total; no dejes caer el peso. '
 '4. Excelente para pico del bíceps por aislar completamente la cabeza larga.',
 'peso_reps', 'Bíceps'),

('curl_martillo',
 'Curl Martillo (Hammer Curl)',
 'curl martillo hammer curl',
 'Bíceps Braquial', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Párate con mancuernas en agarre neutro (palmas hacia el cuerpo). '
 '2. Curla las mancuernas hacia los hombros sin rotar las muñecas. '
 '3. Baja con control hasta la extensión completa. '
 '4. Activa también el braquial y braquiorradial para brazos más gruesos.',
 'peso_reps', 'Bíceps'),

-- ══════════════════════════════════════════════════════════════
-- PIERNAS
-- ══════════════════════════════════════════════════════════════
('sentadilla_goblet',
 'Sentadilla Goblet',
 'sentadilla goblet',
 'Cuádriceps', 'Fuerza', 'Principiante', 5.0, FALSE,
 '1. Sujeta una mancuerna o kettlebell a la altura del pecho con ambas manos. '
 '2. Separa los pies al ancho de hombros con las puntas ligeramente hacia afuera. '
 '3. Baja en sentadilla manteniendo el torso erguido y el peso en los talones. '
 '4. Sube apretando cuádriceps y glúteos. '
 '5. Ideal para principiantes para aprender la mecánica sin cargar barra.',
 'peso_reps', 'Piernas'),

('peso_muerto_rumano',
 'Peso Muerto Rumano',
 'peso muerto rumano',
 'Isquiosurales', 'Fuerza', 'Intermedio', 5.5, FALSE,
 '1. Párate con los pies al ancho de cadera y la barra frente a los muslos. '
 '2. Empuja las caderas hacia atrás manteniendo las rodillas ligeramente flexionadas. '
 '3. Baja la barra por los muslos hasta sentir fuerte estiramiento en los isquiosurales. '
 '4. Contrae glúteos e isquiosurales para subir de vuelta a la posición inicial. '
 '5. Mantén la espalda neutral en todo momento.',
 'peso_reps', 'Piernas'),

('hip_thrust_barra',
 'Hip Thrust con Barra',
 'hip thrust barra',
 'Glúteos Mayor', 'Hipertrofia', 'Intermedio', 5.5, FALSE,
 '1. Apoya los omóplatos en un banco y coloca la barra sobre las caderas (usa almohadilla). '
 '2. Planta los pies en el suelo con rodillas a 90°. '
 '3. Empuja las caderas hacia arriba hasta que el torso quede horizontal, apretando glúteos. '
 '4. Baja con control sin tocar el suelo entre repeticiones para mantener tensión. '
 '5. Es el ejercicio rey de activación del glúteo mayor.',
 'peso_reps', 'Piernas'),

('step_up_con_mancuernas',
 'Step Up con Mancuernas',
 'step up mancuernas',
 'Cuádriceps', 'Fuerza/Unilateral', 'Principiante', 4.5, FALSE,
 '1. Párate frente a una caja o step con mancuernas en los costados. '
 '2. Sube un pie al step y empuja con ese talón para llevar el cuerpo hacia arriba. '
 '3. Une el otro pie al step y baja con control con la misma pierna que subió primero. '
 '4. Alterna piernas o completa todas las reps con una antes de cambiar.',
 'peso_reps', 'Piernas'),

('sentadilla_sumo',
 'Sentadilla Sumo con Mancuerna',
 'sentadilla sumo mancuerna',
 'Cuádriceps', 'Fuerza', 'Principiante', 5.0, FALSE,
 '1. Separa los pies más allá del ancho de hombros con las puntas a 45°. '
 '2. Sujeta una mancuerna pesada con ambas manos frente al cuerpo. '
 '3. Baja manteniendo el pecho alto y rodillas en línea con los pies. '
 '4. Sube apretando aductores y glúteos. '
 '5. Activa más aductores y glúteos que la sentadilla convencional.',
 'peso_reps', 'Piernas'),

('curl_femoral_maquina',
 'Curl Femoral en Máquina (Tumbado)',
 'curl femoral maquina tumbado',
 'Isquiosurales', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Acuéstate boca abajo en la máquina con el rodillo a la altura de los tobillos. '
 '2. Curla las piernas hacia los glúteos contrayendo los isquiosurales. '
 '3. Baja con control hasta la extensión completa; no bloquees la rodilla. '
 '4. Ideal para hipertrofia de isquiosurales como complemento del peso muerto.',
 'peso_reps', 'Piernas'),

('extensiones_cuadriceps_maquina',
 'Extensiones de Cuádriceps en Máquina',
 'extensiones cuadriceps maquina',
 'Cuádriceps', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Siéntate en la máquina con el respaldo ajustado para que los muslos queden horizontales. '
 '2. Coloca los tobillos debajo del rodillo y extiende las piernas hasta casi bloquear las rodillas. '
 '3. Contrae el cuádriceps un segundo antes de bajar lentamente. '
 '4. Úsalo como ejercicio de aislamiento al final de la sesión de piernas.',
 'peso_reps', 'Piernas'),

-- ══════════════════════════════════════════════════════════════
-- GLÚTEOS
-- ══════════════════════════════════════════════════════════════
('patada_trasera_polea',
 'Patada Trasera con Polea (Glúteo)',
 'patada trasera polea gluteo',
 'Glúteos Mayor', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Conecta la correa de tobillo en la polea baja y ajusta el peso. '
 '2. Párate frente a la máquina agarrando el soporte para equilibrio. '
 '3. Extiende la pierna hacia atrás y arriba apretando el glúteo al máximo. '
 '4. Regresa con control sin soltar tensión en el glúteo. '
 '5. Evita arquear la espalda lumbar; el movimiento viene solo de la cadera.',
 'peso_reps', 'Glúteos'),

('abduccion_cadera_maquina',
 'Abducción de Cadera en Máquina',
 'abduccion cadera maquina',
 'Glúteos', 'Hipertrofia', 'Principiante', 4.0, FALSE,
 '1. Siéntate en la máquina con las almohadillas en los muslos exteriores. '
 '2. Abre las piernas hacia los lados empujando contra la resistencia. '
 '3. Mantén la contracción del glúteo medio y tensor de la fascia lata al final. '
 '4. Regresa con control sin dejar que el peso cierre las piernas de golpe.',
 'peso_reps', 'Glúteos'),

-- ══════════════════════════════════════════════════════════════
-- CORE
-- ══════════════════════════════════════════════════════════════
('dead_bug',
 'Dead Bug (Bicho Muerto)',
 'dead bug bicho muerto',
 'Core', 'Fuerza/Autocarga', 'Principiante', 3.5, FALSE,
 '1. Acuéstate boca arriba con brazos extendidos al techo y rodillas a 90°. '
 '2. Activa el core presionando la zona lumbar contra el suelo. '
 '3. Baja simultáneamente el brazo derecho y la pierna izquierda hacia el suelo sin tocarlo. '
 '4. Vuelve al centro y repite con el lado contrario. '
 '5. El movimiento debe ser lento y controlado; si pierdes la tensión lumbar, para.',
 'solo_reps', 'Core'),

('rueda_abdominales',
 'Rueda Abdominal (Ab Roller)',
 'rueda abdominal ab roller',
 'Core', 'Fuerza', 'Avanzado', 4.5, FALSE,
 '1. Arrodíllate con la rueda frente a ti, manos en las manivelas. '
 '2. Rueda hacia adelante extendiendo el cuerpo casi horizontal manteniendo la espalda neutral. '
 '3. Contrae el core para jalar la rueda de vuelta hacia las rodillas. '
 '4. No permitas que la espalda se arquee; si no puedes controlar, solo sal a la mitad del movimiento.',
 'solo_reps', 'Core'),

('elevacion_de_piernas_colgado',
 'Elevación de Piernas Colgado en Barra',
 'elevacion piernas colgado barra',
 'Abdominales', 'Fuerza', 'Avanzado', 5.0, FALSE,
 '1. Cuélgate de una barra de dominadas con agarre al ancho de hombros. '
 '2. Eleva las piernas rectas hasta la horizontal (o más) contrayendo el abdomen. '
 '3. Baja con control sin balancearte; si te balanceas, reduce el rango. '
 '4. Versión más fácil: dobla las rodillas al elevar.',
 'solo_reps', 'Core'),

('rotacion_con_disco',
 'Rotación con Disco (Russian Twist)',
 'rotacion disco russian twist',
 'Abdominales', 'Fuerza', 'Intermedio', 4.0, FALSE,
 '1. Siéntate con rodillas flexionadas y pies ligeramente levantados del suelo. '
 '2. Sujeta un disco o mancuerna frente al pecho con ambas manos. '
 '3. Rota el torso de lado a lado tocando el peso cerca del suelo en cada repetición. '
 '4. Mantén el core tenso y la espalda neutral; no redondees la zona lumbar.',
 'peso_reps', 'Core'),

-- ══════════════════════════════════════════════════════════════
-- CARDIO / METABÓLICO
-- ══════════════════════════════════════════════════════════════
('salto_a_cajón',
 'Salto al Cajón (Box Jump)',
 'salto cajon box jump',
 'Cuádriceps', 'Pliométricos', 'Intermedio', 8.0, TRUE,
 '1. Párate frente al cajón con los pies al ancho de hombros. '
 '2. Flexiona rodillas y caderas y usa el impulso de los brazos para saltar al cajón. '
 '3. Aterriza con ambos pies al mismo tiempo, amortiguando la caída con rodillas flexionadas. '
 '4. Baja del cajón dando un paso (no saltando) para reducir el impacto. '
 '5. Empieza con cajones bajos (30cm) y progresa conforme mejores la técnica.',
 'solo_reps', 'Cardio'),

('sprint_en_cinta',
 'Sprint Intervalo en Cinta (HIIT)',
 'sprint intervalo cinta hiit',
 'Cuerpo Completo', 'Metabólico/HIIT', 'Intermedio', 11.0, TRUE,
 '1. Calienta 5 min a marcha rápida (6 km/h). '
 '2. Aumenta la velocidad a sprint (14-18 km/h) por 30 segundos. '
 '3. Reduce a caminata activa (5-6 km/h) por 90 segundos para recuperar. '
 '4. Repite 6-10 ciclos según nivel de condición. '
 '5. Termina con 5 min de caminata y estiramiento. '
 '6. No uses el pasamanos de la cinta durante los sprints.',
 'tiempo', 'Cardio'),

('remo_ergometro',
 'Remo en Ergómetro',
 'remo ergometro',
 'Cuerpo Completo', 'Cardio', 'Principiante', 7.0, TRUE,
 '1. Ajusta el pie del ergómetro y siéntate con la espalda recta. '
 '2. Agarra el remo, extiende las piernas empujando hacia atrás, luego inclínate ligeramente y jala el remo al abdomen. '
 '3. Invierte la secuencia para regresar: brazos, torso, rodillas. '
 '4. Mantén un ritmo constante de 24-28 jaladas por minuto para condición aeróbica.',
 'tiempo', 'Cardio'),

('cuerda_de_combate',
 'Cuerda de Combate (Battle Ropes)',
 'cuerda combate battle ropes',
 'Cuerpo Completo', 'Metabólico/HIIT', 'Intermedio', 9.0, TRUE,
 '1. Párate a la mitad de las cuerdas con pies al ancho de hombros y rodillas flexionadas. '
 '2. Agarra un extremo con cada mano y crea ondas alternando los brazos arriba-abajo. '
 '3. Trabaja en intervalos: 20-30 s de esfuerzo máximo, 30-40 s de descanso. '
 '4. Varía el patrón: ondas dobles, círculos, slams para estimular distintos músculos.',
 'tiempo', 'Cardio'),

('saltar_soga_doble',
 'Saltar Soga (Doble Velocidad)',
 'saltar soga doble velocidad',
 'Pantorrillas', 'Cardio', 'Intermedio', 10.0, TRUE,
 '1. Sujeta la cuerda con ambas manos a la altura de las caderas. '
 '2. Gira las muñecas (no los brazos completos) y salta con ambos pies juntos. '
 '3. Para doble unders: salta un poco más alto para que la cuerda pase dos veces por vuelta. '
 '4. Mantén los codos cerca del cuerpo y la vista al frente.',
 'tiempo', 'Cardio'),

-- ══════════════════════════════════════════════════════════════
-- FUNCIONAL / CUERPO COMPLETO
-- ══════════════════════════════════════════════════════════════
('thruster',
 'Thruster (Sentadilla + Press)',
 'thruster sentadilla press',
 'Cuerpo Completo', 'Metabólico/HIIT', 'Avanzado', 9.0, TRUE,
 '1. Sostén mancuernas o barra a la altura de los hombros con agarre frontal. '
 '2. Realiza una sentadilla completa descendiendo hasta que los muslos queden paralelos al suelo. '
 '3. Al subir usa el impulso para empujar el peso por encima de la cabeza. '
 '4. Baja el peso de vuelta a los hombros y repite sin pausas entre fases. '
 '5. Es uno de los ejercicios con mayor gasto calórico por repetición.',
 'peso_reps', 'Cuerpo Completo'),

('clean_and_jerk_mancuernas',
 'Clean & Press con Mancuernas',
 'clean press mancuernas',
 'Cuerpo Completo', 'Halterofilia', 'Avanzado', 6.0, FALSE,
 '1. Coloca las mancuernas al frente de los pies, rodillas flexionadas y espalda plana. '
 '2. Jalona las mancuernas hacia los hombros usando la extensión explosiva de caderas. '
 '3. Desde los hombros empuja las mancuernas sobre la cabeza en un movimiento continuo. '
 '4. Baja con control hasta la posición inicial. '
 '5. Exige coordinación; practica primero con poco peso.',
 'peso_reps', 'Cuerpo Completo'),

('farmer_walk',
 'Caminata del Granjero (Farmer Walk)',
 'caminata granjero farmer walk',
 'Cuerpo Completo', 'Strongman', 'Principiante', 5.0, FALSE,
 '1. Sujeta mancuernas pesadas o kettlebells en cada mano con agarre firme. '
 '2. Párate erguido con los hombros hacia atrás y el core activado. '
 '3. Camina pasos firmes a ritmo moderado durante la distancia o tiempo establecido. '
 '4. Beneficia el agarre, trapecios, core y resistencia metabólica simultáneamente.',
 'tiempo', 'Cuerpo Completo'),

('turkish_get_up',
 'Turkish Get-Up (TGU)',
 'turkish get up tgu',
 'Cuerpo Completo', 'Fuerza', 'Avanzado', 5.0, FALSE,
 '1. Acuéstate boca arriba con kettlebell o mancuerna en la mano derecha, brazo extendido al techo. '
 '2. Apoya el codo derecho en el suelo y luego la mano para incorporarte hasta una rodilla. '
 '3. Levántate completamente de pie sin perder el brazo extendido con el peso. '
 '4. Revierte cada paso hasta volver a la posición inicial en el suelo. '
 '5. Desarrolla estabilidad de hombro, movilidad y fuerza funcional simultáneamente.',
 'solo_reps', 'Cuerpo Completo')

ON CONFLICT (id) DO NOTHING;

-- Insertar músculos secundarios en ejercicio_musculo para los nuevos ejercicios
INSERT INTO ejercicio_musculo (ejercicio_id, musculo) VALUES
('press_banca_plano',          'Tríceps'),
('press_banca_plano',          'Deltoides Anterior'),
('press_banca_inclinado',      'Tríceps'),
('press_banca_inclinado',      'Deltoides Anterior'),
('press_banca_declinado',      'Tríceps'),
('aperturas_con_mancuernas',   'Deltoides Anterior'),
('fondos_en_paralelas_pecho',  'Tríceps'),
('fondos_en_paralelas_pecho',  'Deltoides'),
('press_pectoral_maquina',     'Tríceps'),
('remo_con_barra',             'Bíceps'),
('remo_con_barra',             'Romboides'),
('remo_en_polea_baja',         'Bíceps'),
('remo_en_polea_baja',         'Romboides'),
('face_pull',                  'Romboides'),
('face_pull',                  'Manguito Rotador'),
('press_militar_barra',        'Tríceps'),
('press_militar_barra',        'Trapecio'),
('press_hombros_mancuernas',   'Tríceps'),
('press_frances',              'Codo'),
('extension_tricep_polea',     'Antebrazo'),
('hip_thrust_barra',           'Isquiosurales'),
('hip_thrust_barra',           'Core'),
('sentadilla_goblet',          'Glúteos'),
('sentadilla_goblet',          'Core'),
('peso_muerto_rumano',         'Glúteos'),
('peso_muerto_rumano',         'Espalda Baja'),
('salto_a_cajón',              'Glúteos'),
('salto_a_cajón',              'Pantorrillas'),
('sprint_en_cinta',            'Pantorrillas'),
('sprint_en_cinta',            'Glúteos'),
('thruster',                   'Deltoides'),
('thruster',                   'Tríceps'),
('cuerda_de_combate',          'Bíceps'),
('cuerda_de_combate',          'Core'),
('turkish_get_up',             'Core'),
('turkish_get_up',             'Hombros')
ON CONFLICT DO NOTHING;

SELECT COUNT(*) AS total_ejercicios FROM ejercicios;
SELECT grupo_padre, COUNT(*) AS total FROM ejercicios GROUP BY grupo_padre ORDER BY total DESC;
