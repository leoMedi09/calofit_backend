-- ==========================================================
-- CALOFIT - GIMNASIO WORLD LIGHT
-- ESQUEMA COMPLETO DE BASE DE DATOS (POSTGRESQL DDL)
-- Generado por reflection directa desde la BD de produccion (Neon)
-- ==========================================================

-- TABLA: alertas_salud
CREATE TABLE alertas_salud (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	tipo VARCHAR(50) NOT NULL, 
	descripcion TEXT NOT NULL, 
	severidad VARCHAR(20), 
	estado VARCHAR(20), 
	atendido_por_id INTEGER, 
	notas TEXT, 
	fecha_deteccion TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	fecha_atencion TIMESTAMP WITHOUT TIME ZONE, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	CONSTRAINT alertas_salud_pkey PRIMARY KEY (id), 
	CONSTRAINT alertas_salud_atendido_por_id_fkey FOREIGN KEY(atendido_por_id) REFERENCES users (id), 
	CONSTRAINT alertas_salud_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: alimento_alias
CREATE TABLE alimento_alias (
	id SERIAL NOT NULL, 
	alimento_id INTEGER NOT NULL, 
	alias VARCHAR(255) NOT NULL, 
	alias_normalizado VARCHAR(255) NOT NULL, 
	CONSTRAINT alimento_alias_pkey PRIMARY KEY (id), 
	CONSTRAINT alimento_alias_alimento_id_fkey FOREIGN KEY(alimento_id) REFERENCES alimentos (id) ON DELETE CASCADE
);

-- TABLA: alimento_unidades
CREATE TABLE alimento_unidades (
	id SERIAL NOT NULL, 
	alimento_id INTEGER NOT NULL, 
	nombre VARCHAR(100) NOT NULL, 
	gramos DOUBLE PRECISION NOT NULL, 
	CONSTRAINT alimento_unidades_pkey PRIMARY KEY (id), 
	CONSTRAINT alimento_unidades_alimento_id_fkey FOREIGN KEY(alimento_id) REFERENCES alimentos (id) ON DELETE CASCADE
);

-- TABLA: alimentos
CREATE TABLE alimentos (
	id SERIAL NOT NULL, 
	nombre VARCHAR(255) NOT NULL, 
	nombre_normalizado VARCHAR(255) NOT NULL, 
	calorias_100g DOUBLE PRECISION NOT NULL, 
	proteina_100g DOUBLE PRECISION NOT NULL, 
	carbohidratos_100g DOUBLE PRECISION NOT NULL, 
	grasas_100g DOUBLE PRECISION NOT NULL, 
	fibra_100g DOUBLE PRECISION, 
	azucar_100g DOUBLE PRECISION, 
	categoria VARCHAR(100), 
	fuente VARCHAR(255), 
	id_externo VARCHAR(100), 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now(), 
	es_confiable BOOLEAN DEFAULT true, 
	pendiente_validacion BOOLEAN DEFAULT false, 
	CONSTRAINT alimentos_pkey PRIMARY KEY (id), 
	CONSTRAINT alimentos_nombre_key UNIQUE (nombre)
);

-- TABLA: alimentos_sin_resolver
CREATE TABLE alimentos_sin_resolver (
	id SERIAL NOT NULL, 
	nombre_original VARCHAR(512) NOT NULL, 
	nombre_normalizado VARCHAR(512), 
	user_id INTEGER, 
	reporter_id INTEGER, 
	mensaje_contexto TEXT, 
	intentos INTEGER DEFAULT 1 NOT NULL, 
	estado VARCHAR(32) DEFAULT 'pendiente'::character varying NOT NULL, 
	notas TEXT, 
	fecha_reporte TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	fecha_resolucion TIMESTAMP WITH TIME ZONE, 
	CONSTRAINT alimentos_sin_resolver_pkey PRIMARY KEY (id), 
	CONSTRAINT alimentos_sin_resolver_reporter_id_fkey FOREIGN KEY(reporter_id) REFERENCES users (id) ON DELETE SET NULL, 
	CONSTRAINT alimentos_sin_resolver_user_id_fkey FOREIGN KEY(user_id) REFERENCES clients (id) ON DELETE SET NULL
);

-- TABLA: app_cache_alimentos
CREATE TABLE app_cache_alimentos (
	id SERIAL NOT NULL, 
	food_normalized VARCHAR(255) NOT NULL, 
	user_id INTEGER, 
	alimento_id INTEGER, 
	source VARCHAR(64), 
	raw_response TEXT, 
	hit_count INTEGER DEFAULT 1 NOT NULL, 
	expires_at TIMESTAMP WITH TIME ZONE, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT app_cache_alimentos_pkey PRIMARY KEY (id), 
	CONSTRAINT app_cache_alimentos_alimento_id_fkey FOREIGN KEY(alimento_id) REFERENCES alimentos (id) ON DELETE CASCADE, 
	CONSTRAINT app_cache_alimentos_user_id_fkey FOREIGN KEY(user_id) REFERENCES clients (id) ON DELETE SET NULL
);

-- TABLA: app_cache_platos
CREATE TABLE app_cache_platos (
	id SERIAL NOT NULL, 
	plato_normalized VARCHAR(255) NOT NULL, 
	user_id INTEGER, 
	plato_id INTEGER, 
	source VARCHAR(64), 
	hit_count INTEGER DEFAULT 1 NOT NULL, 
	expires_at TIMESTAMP WITH TIME ZONE, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT app_cache_platos_pkey PRIMARY KEY (id), 
	CONSTRAINT app_cache_platos_plato_id_fkey FOREIGN KEY(plato_id) REFERENCES platos (id) ON DELETE CASCADE, 
	CONSTRAINT app_cache_platos_user_id_fkey FOREIGN KEY(user_id) REFERENCES clients (id) ON DELETE SET NULL
);

-- TABLA: auditoria_admin
CREATE TABLE auditoria_admin (
	id SERIAL NOT NULL, 
	admin_id INTEGER NOT NULL, 
	accion VARCHAR(100) NOT NULL, 
	descripcion TEXT NOT NULL, 
	tabla_afectada VARCHAR(50), 
	registro_id INTEGER, 
	fecha_evento TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL, 
	ip_origen VARCHAR(45), 
	CONSTRAINT auditoria_admin_pkey PRIMARY KEY (id), 
	CONSTRAINT auditoria_admin_admin_id_fkey FOREIGN KEY(admin_id) REFERENCES users (id)
);

-- TABLA: chat_historial
CREATE TABLE chat_historial (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	rol VARCHAR(20) NOT NULL, 
	contenido TEXT NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT chat_historial_pkey PRIMARY KEY (id), 
	CONSTRAINT chat_historial_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id) ON DELETE CASCADE
);

-- TABLA: clients
CREATE TABLE clients (
	id SERIAL NOT NULL, 
	first_name VARCHAR NOT NULL, 
	last_name_paternal VARCHAR NOT NULL, 
	last_name_maternal VARCHAR NOT NULL, 
	email VARCHAR NOT NULL, 
	hashed_password VARCHAR NOT NULL, 
	flutter_uid VARCHAR NOT NULL, 
	birth_date DATE, 
	weight DOUBLE PRECISION, 
	height DOUBLE PRECISION, 
	gender VARCHAR(1) NOT NULL, 
	medical_conditions VARCHAR[], 
	activity_level VARCHAR, 
	goal VARCHAR, 
	assigned_coach_id INTEGER, 
	assigned_nutri_id INTEGER, 
	ai_strategic_focus VARCHAR, 
	recommended_foods VARCHAR[], 
	forbidden_foods VARCHAR[], 
	is_strategic_guide_validated BOOLEAN, 
	profile_picture_url VARCHAR, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	verification_code VARCHAR(6), 
	code_expires_at TIMESTAMP WITHOUT TIME ZONE, 
	is_profile_complete BOOLEAN DEFAULT false, 
	dni VARCHAR, 
	workout_type VARCHAR, 
	session_duration DOUBLE PRECISION, 
	nutri_weekly_note TEXT, 
	coach_notes TEXT, 
	fcm_token VARCHAR, 
	notificaciones_activas BOOLEAN DEFAULT true, 
	terms_accepted_at TIMESTAMP WITHOUT TIME ZONE, 
	CONSTRAINT clients_pkey PRIMARY KEY (id), 
	CONSTRAINT clients_assigned_coach_id_fkey FOREIGN KEY(assigned_coach_id) REFERENCES users (id), 
	CONSTRAINT clients_assigned_nutri_id_fkey FOREIGN KEY(assigned_nutri_id) REFERENCES users (id), 
	CONSTRAINT clients_dni_key UNIQUE (dni)
);

-- TABLA: comida_registros
CREATE TABLE comida_registros (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	fecha DATE NOT NULL, 
	nombre_alimento VARCHAR(255) NOT NULL, 
	plato_id INTEGER, 
	alimento_id INTEGER, 
	gramos DOUBLE PRECISION, 
	kcal DOUBLE PRECISION NOT NULL, 
	proteina_g DOUBLE PRECISION NOT NULL, 
	carbohidratos_g DOUBLE PRECISION NOT NULL, 
	grasas_g DOUBLE PRECISION NOT NULL, 
	tipo_resolucion VARCHAR(50) NOT NULL, 
	confianza DOUBLE PRECISION NOT NULL, 
	texto_original VARCHAR(500), 
	momento VARCHAR(20), 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT comida_registros_pkey PRIMARY KEY (id), 
	CONSTRAINT comida_registros_alimento_id_fkey FOREIGN KEY(alimento_id) REFERENCES alimentos (id), 
	CONSTRAINT comida_registros_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id), 
	CONSTRAINT comida_registros_plato_id_fkey FOREIGN KEY(plato_id) REFERENCES platos (id)
);

-- TABLA: ejercicios
CREATE TABLE ejercicios (
	id VARCHAR(100) NOT NULL, 
	nombre VARCHAR(255) NOT NULL, 
	nombre_normalizado VARCHAR(255) NOT NULL, 
	musculo_principal VARCHAR(100), 
	tipo VARCHAR(100), 
	nivel VARCHAR(50), 
	met DOUBLE PRECISION NOT NULL, 
	es_cardio BOOLEAN, 
	tecnica TEXT, 
	equipo JSON, 
	ubicacion JSON, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now(), 
	tipo_metrica VARCHAR(50) DEFAULT 'peso_reps'::character varying, 
	grupo_padre VARCHAR(100), 
	CONSTRAINT ejercicios_pkey PRIMARY KEY (id)
);

-- TABLA: historial_imc
CREATE TABLE historial_imc (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	imc DOUBLE PRECISION NOT NULL, 
	categoria VARCHAR(50) NOT NULL, 
	fecha_registro DATE NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	CONSTRAINT historial_imc_pkey PRIMARY KEY (id), 
	CONSTRAINT historial_imc_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: historial_peso
CREATE TABLE historial_peso (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	peso_kg DOUBLE PRECISION NOT NULL, 
	fecha_registro DATE NOT NULL, 
	notas TEXT, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	CONSTRAINT historial_peso_pkey PRIMARY KEY (id), 
	CONSTRAINT historial_peso_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: historial_recomendaciones
CREATE TABLE historial_recomendaciones (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	plato_id INTEGER, 
	nombre_plato VARCHAR(255), 
	calorias DOUBLE PRECISION, 
	proteinas_g DOUBLE PRECISION, 
	carbohidratos_g DOUBLE PRECISION, 
	grasas_g DOUBLE PRECISION, 
	momento_dia VARCHAR(30), 
	fue_consumido BOOLEAN DEFAULT false, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT historial_recomendaciones_pkey PRIMARY KEY (id), 
	CONSTRAINT historial_recomendaciones_plato_id_fkey FOREIGN KEY(plato_id) REFERENCES platos (id) ON DELETE SET NULL
);

-- TABLA: metas_usuario
CREATE TABLE metas_usuario (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	genero VARCHAR(1) NOT NULL, 
	edad INTEGER NOT NULL, 
	peso_kg DOUBLE PRECISION NOT NULL, 
	talla_cm DOUBLE PRECISION NOT NULL, 
	nivel_actividad VARCHAR(32) NOT NULL, 
	objetivo VARCHAR(64) NOT NULL, 
	tmb DOUBLE PRECISION NOT NULL, 
	get DOUBLE PRECISION NOT NULL, 
	calorias_objetivo DOUBLE PRECISION NOT NULL, 
	proteinas_g DOUBLE PRECISION NOT NULL, 
	carbohidratos_g DOUBLE PRECISION NOT NULL, 
	grasas_g DOUBLE PRECISION NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now(), 
	updated_at TIMESTAMP WITH TIME ZONE, 
	CONSTRAINT metas_usuario_pkey PRIMARY KEY (id), 
	CONSTRAINT metas_usuario_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: password_resets
CREATE TABLE password_resets (
	id SERIAL NOT NULL, 
	email VARCHAR NOT NULL, 
	reset_code VARCHAR(64) NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	is_used BOOLEAN DEFAULT false, 
	used_at TIMESTAMP WITHOUT TIME ZONE, 
	CONSTRAINT password_resets_pkey PRIMARY KEY (id)
);

-- TABLA: planes_diarios
CREATE TABLE planes_diarios (
	id SERIAL NOT NULL, 
	plan_id INTEGER, 
	dia_numero INTEGER NOT NULL, 
	calorias_dia DOUBLE PRECISION NOT NULL, 
	proteinas_g DOUBLE PRECISION NOT NULL, 
	carbohidratos_g DOUBLE PRECISION NOT NULL, 
	grasas_g DOUBLE PRECISION NOT NULL, 
	sugerencia_entrenamiento_ia VARCHAR, 
	nota_asistente_ia VARCHAR, 
	validado_nutri BOOLEAN, 
	estado VARCHAR, 
	CONSTRAINT planes_diarios_pkey PRIMARY KEY (id), 
	CONSTRAINT planes_diarios_plan_id_fkey FOREIGN KEY(plan_id) REFERENCES planes_nutricionales (id)
);

-- TABLA: planes_nutricionales
CREATE TABLE planes_nutricionales (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	nutricionista_id INTEGER, 
	genero INTEGER NOT NULL, 
	edad INTEGER NOT NULL, 
	peso DOUBLE PRECISION NOT NULL, 
	talla DOUBLE PRECISION NOT NULL, 
	nivel_actividad DOUBLE PRECISION NOT NULL, 
	objetivo VARCHAR NOT NULL, 
	es_contingencia_ia BOOLEAN, 
	calorias_ia_base DOUBLE PRECISION, 
	fecha_creacion TIMESTAMP WITHOUT TIME ZONE, 
	observaciones VARCHAR, 
	status VARCHAR, 
	validated_by_id INTEGER, 
	validated_at TIMESTAMP WITHOUT TIME ZONE, 
	CONSTRAINT planes_nutricionales_pkey PRIMARY KEY (id), 
	CONSTRAINT planes_nutricionales_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id), 
	CONSTRAINT planes_nutricionales_nutricionista_id_fkey FOREIGN KEY(nutricionista_id) REFERENCES users (id), 
	CONSTRAINT planes_nutricionales_validated_by_id_fkey FOREIGN KEY(validated_by_id) REFERENCES users (id)
);

-- TABLA: plato_ingredientes
CREATE TABLE plato_ingredientes (
	id SERIAL NOT NULL, 
	plato_id INTEGER NOT NULL, 
	alimento_id INTEGER NOT NULL, 
	gramos DOUBLE PRECISION NOT NULL, 
	orden INTEGER DEFAULT 0 NOT NULL, 
	notas VARCHAR(255), 
	CONSTRAINT plato_ingredientes_pkey PRIMARY KEY (id), 
	CONSTRAINT plato_ingredientes_alimento_id_fkey FOREIGN KEY(alimento_id) REFERENCES alimentos (id) ON DELETE RESTRICT, 
	CONSTRAINT plato_ingredientes_plato_id_fkey FOREIGN KEY(plato_id) REFERENCES platos (id) ON DELETE CASCADE, 
	CONSTRAINT plato_ingredientes_gramos_check CHECK (gramos > 0::double precision)
);

-- TABLA: platos
CREATE TABLE platos (
	id SERIAL NOT NULL, 
	nombre VARCHAR(255) NOT NULL, 
	nombre_normalizado VARCHAR(255) NOT NULL, 
	tipo_plato VARCHAR(50) DEFAULT 'cualquiera'::character varying, 
	preparacion JSON, 
	nota TEXT, 
	origen VARCHAR(50) DEFAULT 'manual'::character varying, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	CONSTRAINT platos_pkey PRIMARY KEY (id), 
	CONSTRAINT platos_nombre_normalizado_key UNIQUE (nombre_normalizado)
);

-- TABLA: preferencias_alimentos
CREATE TABLE preferencias_alimentos (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	alimento VARCHAR(200) NOT NULL, 
	frecuencia INTEGER, 
	puntuacion DOUBLE PRECISION, 
	calorias DOUBLE PRECISION, 
	proteinas DOUBLE PRECISION, 
	carbohidratos DOUBLE PRECISION, 
	grasas DOUBLE PRECISION, 
	ultima_vez TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	es_favorito SMALLINT DEFAULT 0 NOT NULL, 
	CONSTRAINT preferencias_alimentos_pkey PRIMARY KEY (id), 
	CONSTRAINT preferencias_alimentos_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: preferencias_ejercicios
CREATE TABLE preferencias_ejercicios (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	ejercicio VARCHAR(200) NOT NULL, 
	frecuencia INTEGER, 
	puntuacion DOUBLE PRECISION, 
	calorias_quemadas DOUBLE PRECISION, 
	ultima_vez TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	CONSTRAINT preferencias_ejercicios_pkey PRIMARY KEY (id), 
	CONSTRAINT preferencias_ejercicios_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: progreso_calorias
CREATE TABLE progreso_calorias (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	fecha DATE NOT NULL, 
	calorias_consumidas INTEGER, 
	calorias_quemadas INTEGER NOT NULL, 
	proteinas_consumidas DOUBLE PRECISION, 
	carbohidratos_consumidos DOUBLE PRECISION, 
	grasas_consumidas DOUBLE PRECISION, 
	deficit_superavit INTEGER, 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL, 
	alerta_exceso_enviada BOOLEAN DEFAULT false NOT NULL, 
	CONSTRAINT progreso_calorias_pkey PRIMARY KEY (id), 
	CONSTRAINT progreso_calorias_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: roles
CREATE TABLE roles (
	id SERIAL NOT NULL, 
	name VARCHAR NOT NULL, 
	description VARCHAR, 
	CONSTRAINT roles_pkey PRIMARY KEY (id), 
	CONSTRAINT roles_name_key UNIQUE (name)
);

-- TABLA: sugerencias_guardadas
CREATE TABLE sugerencias_guardadas (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	tipo VARCHAR(20) NOT NULL, 
	nombre VARCHAR(255) NOT NULL, 
	ingredientes JSON, 
	preparacion JSON, 
	macros VARCHAR(255), 
	nota TEXT, 
	completada BOOLEAN, 
	fecha_guardado TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	CONSTRAINT sugerencias_guardadas_pkey PRIMARY KEY (id), 
	CONSTRAINT sugerencias_guardadas_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: users
CREATE TABLE users (
	id SERIAL NOT NULL, 
	first_name VARCHAR NOT NULL, 
	last_name_paternal VARCHAR NOT NULL, 
	last_name_maternal VARCHAR NOT NULL, 
	email VARCHAR NOT NULL, 
	hashed_password VARCHAR NOT NULL, 
	role_id INTEGER NOT NULL, 
	role_name VARCHAR NOT NULL, 
	is_active BOOLEAN NOT NULL, 
	profile_picture_url VARCHAR, 
	CONSTRAINT users_pkey PRIMARY KEY (id), 
	CONSTRAINT users_role_id_fkey FOREIGN KEY(role_id) REFERENCES roles (id)
);

-- TABLA: workout_logs
CREATE TABLE workout_logs (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	ejercicio VARCHAR NOT NULL, 
	series INTEGER NOT NULL, 
	reps INTEGER NOT NULL, 
	peso_kg DOUBLE PRECISION, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	calorias_quemadas DOUBLE PRECISION, 
	session_duration_min DOUBLE PRECISION, 
	intensity VARCHAR(50), 
	CONSTRAINT workout_logs_pkey PRIMARY KEY (id), 
	CONSTRAINT workout_logs_client_id_fkey FOREIGN KEY(client_id) REFERENCES clients (id)
);
