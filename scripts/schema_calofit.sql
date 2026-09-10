-- ==========================================================
-- CALOFIT - GIMNASIO WORLD LIGHT
-- ESQUEMA COMPLETO DE BASE DE DATOS (POSTGRESQL DDL)
-- ==========================================================

-- TABLA: alimentos
CREATE TABLE alimentos (
	id SERIAL NOT NULL, 
	nombre VARCHAR(255) NOT NULL, 
	nombre_normalizado VARCHAR(255) NOT NULL, 
	calorias_100g FLOAT NOT NULL, 
	proteina_100g FLOAT NOT NULL, 
	carbohidratos_100g FLOAT NOT NULL, 
	grasas_100g FLOAT NOT NULL, 
	fibra_100g FLOAT, 
	azucar_100g FLOAT, 
	categoria VARCHAR(100), 
	fuente VARCHAR(255), 
	id_externo VARCHAR(100), 
	es_confiable BOOLEAN, 
	pendiente_validacion BOOLEAN, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now(), 
	PRIMARY KEY (id)
);

-- TABLA: ejercicios
CREATE TABLE ejercicios (
	id SERIAL NOT NULL, 
	nombre VARCHAR(255) NOT NULL, 
	nombre_normalizado VARCHAR(255) NOT NULL, 
	alias TEXT, 
	descripcion TEXT, 
	met FLOAT NOT NULL, 
	grupo_muscular VARCHAR(128), 
	origen VARCHAR(64), 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now(), 
	updated_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id)
);

-- TABLA: platos
CREATE TABLE platos (
	id SERIAL NOT NULL, 
	nombre VARCHAR(255) NOT NULL, 
	nombre_normalizado VARCHAR(255) NOT NULL, 
	tipo_plato VARCHAR(50), 
	preparacion JSON, 
	nota TEXT, 
	origen VARCHAR(50), 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id)
);

-- TABLA: roles
CREATE TABLE roles (
	id SERIAL NOT NULL, 
	name VARCHAR NOT NULL, 
	description VARCHAR, 
	PRIMARY KEY (id), 
	UNIQUE (name)
);

-- TABLA: alimento_alias
CREATE TABLE alimento_alias (
	id SERIAL NOT NULL, 
	alimento_id INTEGER NOT NULL, 
	alias VARCHAR(255) NOT NULL, 
	alias_normalizado VARCHAR(255) NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(alimento_id) REFERENCES alimentos (id) ON DELETE CASCADE
);

-- TABLA: alimento_unidades
CREATE TABLE alimento_unidades (
	id SERIAL NOT NULL, 
	alimento_id INTEGER NOT NULL, 
	nombre VARCHAR(100) NOT NULL, 
	gramos FLOAT NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(alimento_id) REFERENCES alimentos (id) ON DELETE CASCADE
);

-- TABLA: historial_recomendaciones
CREATE TABLE historial_recomendaciones (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	plato_id INTEGER, 
	nombre_plato VARCHAR(255), 
	calorias FLOAT, 
	proteinas_g FLOAT, 
	carbohidratos_g FLOAT, 
	grasas_g FLOAT, 
	momento_dia VARCHAR(30), 
	fue_consumido BOOLEAN, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(plato_id) REFERENCES platos (id) ON DELETE SET NULL
);

-- TABLA: plato_ingredientes
CREATE TABLE plato_ingredientes (
	id SERIAL NOT NULL, 
	plato_id INTEGER NOT NULL, 
	alimento_id INTEGER NOT NULL, 
	gramos FLOAT NOT NULL, 
	orden INTEGER NOT NULL, 
	notas VARCHAR(255), 
	PRIMARY KEY (id), 
	CONSTRAINT ck_plato_ing_gramos_positivo CHECK (gramos > 0), 
	FOREIGN KEY(plato_id) REFERENCES platos (id) ON DELETE CASCADE, 
	FOREIGN KEY(alimento_id) REFERENCES alimentos (id) ON DELETE RESTRICT
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
	PRIMARY KEY (id), 
	FOREIGN KEY(role_id) REFERENCES roles (id)
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
	PRIMARY KEY (id), 
	FOREIGN KEY(admin_id) REFERENCES users (id)
);

-- TABLA: clients
CREATE TABLE clients (
	id SERIAL NOT NULL, 
	first_name VARCHAR, 
	last_name_paternal VARCHAR, 
	last_name_maternal VARCHAR, 
	dni VARCHAR, 
	email VARCHAR NOT NULL, 
	hashed_password VARCHAR NOT NULL, 
	flutter_uid VARCHAR, 
	fcm_token VARCHAR, 
	notificaciones_activas BOOLEAN, 
	birth_date DATE, 
	weight FLOAT, 
	height FLOAT, 
	gender VARCHAR(1) NOT NULL, 
	medical_conditions VARCHAR[], 
	activity_level VARCHAR, 
	goal VARCHAR, 
	workout_type VARCHAR, 
	session_duration FLOAT, 
	assigned_coach_id INTEGER, 
	assigned_nutri_id INTEGER, 
	ai_strategic_focus VARCHAR, 
	recommended_foods VARCHAR[], 
	forbidden_foods VARCHAR[], 
	nutri_weekly_note TEXT, 
	coach_notes TEXT, 
	is_strategic_guide_validated BOOLEAN, 
	profile_picture_url VARCHAR, 
	is_profile_complete BOOLEAN, 
	terms_accepted_at TIMESTAMP WITHOUT TIME ZONE, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	verification_code VARCHAR(6), 
	code_expires_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(assigned_coach_id) REFERENCES users (id), 
	FOREIGN KEY(assigned_nutri_id) REFERENCES users (id)
);

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
	PRIMARY KEY (id), 
	FOREIGN KEY(client_id) REFERENCES clients (id), 
	FOREIGN KEY(atendido_por_id) REFERENCES users (id)
);

-- TABLA: alimentos_sin_resolver
CREATE TABLE alimentos_sin_resolver (
	id SERIAL NOT NULL, 
	nombre_original VARCHAR(512) NOT NULL, 
	nombre_normalizado VARCHAR(512), 
	user_id INTEGER, 
	reporter_id INTEGER, 
	mensaje_contexto TEXT, 
	intentos INTEGER NOT NULL, 
	estado VARCHAR(32) NOT NULL, 
	notas TEXT, 
	fecha_reporte TIMESTAMP WITH TIME ZONE NOT NULL, 
	fecha_resolucion TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id), 
	CONSTRAINT ck_alimento_sin_resolver_estado CHECK (estado IN ('pendiente', 'validado', 'rechazado')), 
	FOREIGN KEY(user_id) REFERENCES clients (id) ON DELETE SET NULL, 
	FOREIGN KEY(reporter_id) REFERENCES users (id) ON DELETE SET NULL
);

-- TABLA: app_cache_alimentos
CREATE TABLE app_cache_alimentos (
	id SERIAL NOT NULL, 
	food_normalized VARCHAR(255) NOT NULL, 
	user_id INTEGER, 
	alimento_id INTEGER, 
	source VARCHAR(64), 
	raw_response TEXT, 
	hit_count INTEGER NOT NULL, 
	expires_at TIMESTAMP WITH TIME ZONE, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES clients (id) ON DELETE SET NULL, 
	FOREIGN KEY(alimento_id) REFERENCES alimentos (id) ON DELETE CASCADE
);

-- TABLA: app_cache_platos
CREATE TABLE app_cache_platos (
	id SERIAL NOT NULL, 
	plato_normalized VARCHAR(255) NOT NULL, 
	user_id INTEGER, 
	plato_id INTEGER, 
	source VARCHAR(64), 
	hit_count INTEGER NOT NULL, 
	expires_at TIMESTAMP WITH TIME ZONE, 
	created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(user_id) REFERENCES clients (id) ON DELETE SET NULL, 
	FOREIGN KEY(plato_id) REFERENCES platos (id) ON DELETE CASCADE
);

-- TABLA: comida_registros
CREATE TABLE comida_registros (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	fecha DATE NOT NULL, 
	nombre_alimento VARCHAR(255) NOT NULL, 
	plato_id INTEGER, 
	alimento_id INTEGER, 
	gramos FLOAT, 
	kcal FLOAT NOT NULL, 
	proteina_g FLOAT NOT NULL, 
	carbohidratos_g FLOAT NOT NULL, 
	grasas_g FLOAT NOT NULL, 
	tipo_resolucion VARCHAR(50) NOT NULL, 
	confianza FLOAT NOT NULL, 
	texto_original VARCHAR(500), 
	momento VARCHAR(20), 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(client_id) REFERENCES clients (id), 
	FOREIGN KEY(plato_id) REFERENCES platos (id), 
	FOREIGN KEY(alimento_id) REFERENCES alimentos (id)
);

-- TABLA: historial_imc
CREATE TABLE historial_imc (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	imc FLOAT NOT NULL, 
	categoria VARCHAR(50) NOT NULL, 
	fecha_registro DATE NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: historial_peso
CREATE TABLE historial_peso (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	peso_kg FLOAT NOT NULL, 
	fecha_registro DATE NOT NULL, 
	notas TEXT, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: metas_usuario
CREATE TABLE metas_usuario (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	genero VARCHAR(1) NOT NULL, 
	edad INTEGER NOT NULL, 
	peso_kg FLOAT NOT NULL, 
	talla_cm FLOAT NOT NULL, 
	nivel_actividad VARCHAR(32) NOT NULL, 
	objetivo VARCHAR(64) NOT NULL, 
	tmb FLOAT NOT NULL, 
	get FLOAT NOT NULL, 
	calorias_objetivo FLOAT NOT NULL, 
	proteinas_g FLOAT NOT NULL, 
	carbohidratos_g FLOAT NOT NULL, 
	grasas_g FLOAT NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now(), 
	updated_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: planes_nutricionales
CREATE TABLE planes_nutricionales (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	nutricionista_id INTEGER, 
	genero INTEGER NOT NULL, 
	edad INTEGER NOT NULL, 
	peso FLOAT NOT NULL, 
	talla FLOAT NOT NULL, 
	nivel_actividad FLOAT NOT NULL, 
	objetivo VARCHAR NOT NULL, 
	es_contingencia_ia BOOLEAN, 
	calorias_ia_base FLOAT, 
	fecha_creacion TIMESTAMP WITHOUT TIME ZONE, 
	observaciones VARCHAR, 
	status VARCHAR, 
	validated_by_id INTEGER, 
	validated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(client_id) REFERENCES clients (id), 
	FOREIGN KEY(nutricionista_id) REFERENCES users (id), 
	FOREIGN KEY(validated_by_id) REFERENCES users (id)
);

-- TABLA: preferencias_alimentos
CREATE TABLE preferencias_alimentos (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	alimento VARCHAR(200) NOT NULL, 
	frecuencia INTEGER, 
	puntuacion FLOAT, 
	calorias FLOAT, 
	proteinas FLOAT, 
	carbohidratos FLOAT, 
	grasas FLOAT, 
	es_favorito INTEGER NOT NULL, 
	ultima_vez TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	PRIMARY KEY (id), 
	FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: preferencias_ejercicios
CREATE TABLE preferencias_ejercicios (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	ejercicio VARCHAR(200) NOT NULL, 
	frecuencia INTEGER, 
	puntuacion FLOAT, 
	calorias_quemadas FLOAT, 
	ultima_vez TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now(), 
	PRIMARY KEY (id), 
	FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: progreso_calorias
CREATE TABLE progreso_calorias (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	fecha DATE NOT NULL, 
	calorias_consumidas INTEGER, 
	calorias_quemadas INTEGER NOT NULL, 
	proteinas_consumidas FLOAT, 
	carbohidratos_consumidos FLOAT, 
	grasas_consumidas FLOAT, 
	deficit_superavit INTEGER, 
	alerta_exceso_enviada BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT now() NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(client_id) REFERENCES clients (id)
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
	PRIMARY KEY (id), 
	FOREIGN KEY(client_id) REFERENCES clients (id)
);

-- TABLA: workout_logs
CREATE TABLE workout_logs (
	id SERIAL NOT NULL, 
	client_id INTEGER NOT NULL, 
	ejercicio VARCHAR(255) NOT NULL, 
	series INTEGER NOT NULL, 
	reps INTEGER NOT NULL, 
	peso_kg FLOAT, 
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	calorias_quemadas FLOAT, 
	session_duration_min FLOAT, 
	intensity VARCHAR(50), 
	PRIMARY KEY (id), 
	FOREIGN KEY(client_id) REFERENCES clients (id) ON DELETE CASCADE
);

-- TABLA: planes_diarios
CREATE TABLE planes_diarios (
	id SERIAL NOT NULL, 
	plan_id INTEGER, 
	dia_numero INTEGER NOT NULL, 
	calorias_dia FLOAT NOT NULL, 
	proteinas_g FLOAT NOT NULL, 
	carbohidratos_g FLOAT NOT NULL, 
	grasas_g FLOAT NOT NULL, 
	sugerencia_entrenamiento_ia VARCHAR, 
	nota_asistente_ia VARCHAR, 
	validado_nutri BOOLEAN, 
	estado VARCHAR, 
	PRIMARY KEY (id), 
	FOREIGN KEY(plan_id) REFERENCES planes_nutricionales (id)
);

