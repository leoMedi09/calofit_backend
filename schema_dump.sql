--
-- PostgreSQL database dump
--

\restrict 4wATf19WTADOrDnh7Ja0HwkPij6cBenfsVzWvtZtqXyaM2DIAg2BhNnRX0D8Ke4

-- Dumped from database version 15.17 (Debian 15.17-1.pgdg13+1)
-- Dumped by pg_dump version 15.17 (Debian 15.17-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: pg_trgm; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;


--
-- Name: EXTENSION pg_trgm; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION pg_trgm IS 'text similarity measurement and index searching based on trigrams';


--
-- Name: unaccent; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS unaccent WITH SCHEMA public;


--
-- Name: EXTENSION unaccent; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION unaccent IS 'text search dictionary that removes accents';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: alembic_version; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alembic_version (
    version_num character varying(32) NOT NULL
);


ALTER TABLE public.alembic_version OWNER TO postgres;

--
-- Name: alertas_salud; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alertas_salud (
    id integer NOT NULL,
    client_id integer NOT NULL,
    tipo character varying(50) NOT NULL,
    descripcion text NOT NULL,
    severidad character varying(20),
    estado character varying(20),
    atendido_por_id integer,
    notas text,
    fecha_deteccion timestamp without time zone NOT NULL,
    fecha_atencion timestamp without time zone,
    created_at timestamp without time zone NOT NULL
);


ALTER TABLE public.alertas_salud OWNER TO postgres;

--
-- Name: alertas_salud_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.alertas_salud_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.alertas_salud_id_seq OWNER TO postgres;

--
-- Name: alertas_salud_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.alertas_salud_id_seq OWNED BY public.alertas_salud.id;


--
-- Name: alimento_alias; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alimento_alias (
    id integer NOT NULL,
    alimento_id integer NOT NULL,
    alias character varying(255) NOT NULL,
    alias_normalizado character varying(255) NOT NULL
);


ALTER TABLE public.alimento_alias OWNER TO postgres;

--
-- Name: alimento_alias_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.alimento_alias_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.alimento_alias_id_seq OWNER TO postgres;

--
-- Name: alimento_alias_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.alimento_alias_id_seq OWNED BY public.alimento_alias.id;


--
-- Name: alimento_unidades; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alimento_unidades (
    id integer NOT NULL,
    alimento_id integer NOT NULL,
    nombre character varying(100) NOT NULL,
    gramos double precision NOT NULL
);


ALTER TABLE public.alimento_unidades OWNER TO postgres;

--
-- Name: alimento_unidades_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.alimento_unidades_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.alimento_unidades_id_seq OWNER TO postgres;

--
-- Name: alimento_unidades_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.alimento_unidades_id_seq OWNED BY public.alimento_unidades.id;


--
-- Name: alimentos; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alimentos (
    id integer NOT NULL,
    nombre character varying(255) NOT NULL,
    nombre_normalizado character varying(255) NOT NULL,
    calorias_100g double precision NOT NULL,
    proteina_100g double precision NOT NULL,
    carbohidratos_100g double precision NOT NULL,
    grasas_100g double precision NOT NULL,
    fibra_100g double precision,
    azucar_100g double precision,
    categoria character varying(100),
    fuente character varying(255),
    id_externo character varying(100),
    created_at timestamp with time zone DEFAULT now(),
    es_confiable boolean DEFAULT true,
    pendiente_validacion boolean DEFAULT false
);


ALTER TABLE public.alimentos OWNER TO postgres;

--
-- Name: alimentos_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.alimentos_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.alimentos_id_seq OWNER TO postgres;

--
-- Name: alimentos_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.alimentos_id_seq OWNED BY public.alimentos.id;


--
-- Name: alimentos_sin_resolver; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.alimentos_sin_resolver (
    id integer NOT NULL,
    nombre_original character varying(512) NOT NULL,
    nombre_normalizado character varying(512),
    user_id integer,
    reporter_id integer,
    mensaje_contexto text,
    intentos integer DEFAULT 1 NOT NULL,
    estado character varying(32) DEFAULT 'pendiente'::character varying NOT NULL,
    notas text,
    fecha_reporte timestamp with time zone DEFAULT now() NOT NULL,
    fecha_resolucion timestamp with time zone
);


ALTER TABLE public.alimentos_sin_resolver OWNER TO postgres;

--
-- Name: alimentos_sin_resolver_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.alimentos_sin_resolver_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.alimentos_sin_resolver_id_seq OWNER TO postgres;

--
-- Name: alimentos_sin_resolver_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.alimentos_sin_resolver_id_seq OWNED BY public.alimentos_sin_resolver.id;


--
-- Name: app_cache_alimentos; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.app_cache_alimentos (
    id integer NOT NULL,
    food_normalized character varying(255) NOT NULL,
    user_id integer,
    alimento_id integer,
    source character varying(64),
    raw_response text,
    hit_count integer DEFAULT 1 NOT NULL,
    expires_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.app_cache_alimentos OWNER TO postgres;

--
-- Name: app_cache_alimentos_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.app_cache_alimentos_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.app_cache_alimentos_id_seq OWNER TO postgres;

--
-- Name: app_cache_alimentos_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.app_cache_alimentos_id_seq OWNED BY public.app_cache_alimentos.id;


--
-- Name: app_cache_platos; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.app_cache_platos (
    id integer NOT NULL,
    plato_normalized character varying(255) NOT NULL,
    user_id integer,
    plato_id integer,
    source character varying(64),
    hit_count integer DEFAULT 1 NOT NULL,
    expires_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.app_cache_platos OWNER TO postgres;

--
-- Name: app_cache_platos_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.app_cache_platos_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.app_cache_platos_id_seq OWNER TO postgres;

--
-- Name: app_cache_platos_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.app_cache_platos_id_seq OWNED BY public.app_cache_platos.id;


--
-- Name: app_cache_rutinas; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.app_cache_rutinas (
    id integer NOT NULL,
    cache_key character varying(512) NOT NULL,
    user_id integer,
    perfil_tipo character varying(16),
    zonas_objetivo text,
    tiempo_min integer,
    rutina_json text NOT NULL,
    hit_count integer DEFAULT 1 NOT NULL,
    expires_at timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.app_cache_rutinas OWNER TO postgres;

--
-- Name: app_cache_rutinas_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.app_cache_rutinas_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.app_cache_rutinas_id_seq OWNER TO postgres;

--
-- Name: app_cache_rutinas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.app_cache_rutinas_id_seq OWNED BY public.app_cache_rutinas.id;


--
-- Name: auditoria_admin; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.auditoria_admin (
    id integer NOT NULL,
    admin_id integer NOT NULL,
    accion character varying(100) NOT NULL,
    descripcion text NOT NULL,
    tabla_afectada character varying(50),
    registro_id integer,
    fecha_evento timestamp without time zone DEFAULT now() NOT NULL,
    ip_origen character varying(45)
);


ALTER TABLE public.auditoria_admin OWNER TO postgres;

--
-- Name: auditoria_admin_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.auditoria_admin_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.auditoria_admin_id_seq OWNER TO postgres;

--
-- Name: auditoria_admin_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.auditoria_admin_id_seq OWNED BY public.auditoria_admin.id;


--
-- Name: clients; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.clients (
    id integer NOT NULL,
    first_name character varying NOT NULL,
    last_name_paternal character varying NOT NULL,
    last_name_maternal character varying NOT NULL,
    email character varying NOT NULL,
    hashed_password character varying NOT NULL,
    flutter_uid character varying NOT NULL,
    birth_date date,
    weight double precision,
    height double precision,
    gender character varying(1) NOT NULL,
    medical_conditions character varying[],
    activity_level character varying,
    goal character varying,
    assigned_coach_id integer,
    assigned_nutri_id integer,
    ai_strategic_focus character varying,
    recommended_foods character varying[],
    forbidden_foods character varying[],
    is_strategic_guide_validated boolean,
    profile_picture_url character varying,
    created_at timestamp without time zone,
    verification_code character varying(6),
    code_expires_at timestamp without time zone,
    is_profile_complete boolean DEFAULT false,
    dni character varying,
    workout_type character varying,
    session_duration double precision,
    nutri_weekly_note text,
    coach_notes text
);


ALTER TABLE public.clients OWNER TO postgres;

--
-- Name: clients_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.clients_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.clients_id_seq OWNER TO postgres;

--
-- Name: clients_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.clients_id_seq OWNED BY public.clients.id;


--
-- Name: comida_registros; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.comida_registros (
    id integer NOT NULL,
    client_id integer NOT NULL,
    fecha date NOT NULL,
    nombre_alimento character varying(255) NOT NULL,
    plato_id integer,
    alimento_id integer,
    gramos double precision,
    kcal double precision NOT NULL,
    proteina_g double precision NOT NULL,
    carbohidratos_g double precision NOT NULL,
    grasas_g double precision NOT NULL,
    tipo_resolucion character varying(50) NOT NULL,
    confianza double precision NOT NULL,
    texto_original character varying(500),
    momento character varying(20),
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.comida_registros OWNER TO postgres;

--
-- Name: comida_registros_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.comida_registros_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.comida_registros_id_seq OWNER TO postgres;

--
-- Name: comida_registros_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.comida_registros_id_seq OWNED BY public.comida_registros.id;


--
-- Name: ejercicio_alias; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ejercicio_alias (
    id integer NOT NULL,
    ejercicio_id character varying(100) NOT NULL,
    alias character varying(255) NOT NULL
);


ALTER TABLE public.ejercicio_alias OWNER TO postgres;

--
-- Name: ejercicio_alias_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ejercicio_alias_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ejercicio_alias_id_seq OWNER TO postgres;

--
-- Name: ejercicio_alias_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ejercicio_alias_id_seq OWNED BY public.ejercicio_alias.id;


--
-- Name: ejercicio_musculo; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ejercicio_musculo (
    id integer NOT NULL,
    ejercicio_id character varying(100) NOT NULL,
    musculo character varying(100) NOT NULL
);


ALTER TABLE public.ejercicio_musculo OWNER TO postgres;

--
-- Name: ejercicio_musculo_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.ejercicio_musculo_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.ejercicio_musculo_id_seq OWNER TO postgres;

--
-- Name: ejercicio_musculo_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.ejercicio_musculo_id_seq OWNED BY public.ejercicio_musculo.id;


--
-- Name: ejercicios; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ejercicios (
    id character varying(100) NOT NULL,
    nombre character varying(255) NOT NULL,
    nombre_normalizado character varying(255) NOT NULL,
    musculo_principal character varying(100),
    tipo character varying(100),
    nivel character varying(50),
    met double precision NOT NULL,
    es_cardio boolean,
    tecnica text,
    equipo json,
    ubicacion json,
    created_at timestamp with time zone DEFAULT now(),
    tipo_metrica character varying(50) DEFAULT 'peso_reps'::character varying,
    grupo_padre character varying(100)
);


ALTER TABLE public.ejercicios OWNER TO postgres;

--
-- Name: historial_imc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.historial_imc (
    id integer NOT NULL,
    client_id integer NOT NULL,
    imc double precision NOT NULL,
    categoria character varying(50) NOT NULL,
    fecha_registro date NOT NULL,
    created_at timestamp without time zone NOT NULL
);


ALTER TABLE public.historial_imc OWNER TO postgres;

--
-- Name: historial_imc_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.historial_imc_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.historial_imc_id_seq OWNER TO postgres;

--
-- Name: historial_imc_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.historial_imc_id_seq OWNED BY public.historial_imc.id;


--
-- Name: historial_peso; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.historial_peso (
    id integer NOT NULL,
    client_id integer NOT NULL,
    peso_kg double precision NOT NULL,
    fecha_registro date NOT NULL,
    notas text,
    created_at timestamp without time zone NOT NULL
);


ALTER TABLE public.historial_peso OWNER TO postgres;

--
-- Name: historial_peso_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.historial_peso_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.historial_peso_id_seq OWNER TO postgres;

--
-- Name: historial_peso_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.historial_peso_id_seq OWNED BY public.historial_peso.id;


--
-- Name: historial_recomendaciones; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.historial_recomendaciones (
    id integer NOT NULL,
    client_id integer NOT NULL,
    plato_id integer,
    nombre_plato character varying(255),
    calorias double precision,
    proteinas_g double precision,
    carbohidratos_g double precision,
    grasas_g double precision,
    momento_dia character varying(30),
    fue_consumido boolean DEFAULT false,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.historial_recomendaciones OWNER TO postgres;

--
-- Name: historial_recomendaciones_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.historial_recomendaciones_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.historial_recomendaciones_id_seq OWNER TO postgres;

--
-- Name: historial_recomendaciones_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.historial_recomendaciones_id_seq OWNED BY public.historial_recomendaciones.id;


--
-- Name: metas_usuario; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.metas_usuario (
    id integer NOT NULL,
    client_id integer NOT NULL,
    genero character varying(1) NOT NULL,
    edad integer NOT NULL,
    peso_kg double precision NOT NULL,
    talla_cm double precision NOT NULL,
    nivel_actividad character varying(32) NOT NULL,
    objetivo character varying(64) NOT NULL,
    tmb double precision NOT NULL,
    get double precision NOT NULL,
    calorias_objetivo double precision NOT NULL,
    proteinas_g double precision NOT NULL,
    carbohidratos_g double precision NOT NULL,
    grasas_g double precision NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone
);


ALTER TABLE public.metas_usuario OWNER TO postgres;

--
-- Name: metas_usuario_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.metas_usuario_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.metas_usuario_id_seq OWNER TO postgres;

--
-- Name: metas_usuario_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.metas_usuario_id_seq OWNED BY public.metas_usuario.id;


--
-- Name: password_resets; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.password_resets (
    id integer NOT NULL,
    email character varying NOT NULL,
    reset_code character varying(6) NOT NULL,
    created_at timestamp without time zone DEFAULT now(),
    is_used boolean DEFAULT false,
    used_at timestamp without time zone
);


ALTER TABLE public.password_resets OWNER TO postgres;

--
-- Name: password_resets_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.password_resets_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.password_resets_id_seq OWNER TO postgres;

--
-- Name: password_resets_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.password_resets_id_seq OWNED BY public.password_resets.id;


--
-- Name: planes_diarios; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.planes_diarios (
    id integer NOT NULL,
    plan_id integer,
    dia_numero integer NOT NULL,
    calorias_dia double precision NOT NULL,
    proteinas_g double precision NOT NULL,
    carbohidratos_g double precision NOT NULL,
    grasas_g double precision NOT NULL,
    sugerencia_entrenamiento_ia character varying,
    nota_asistente_ia character varying,
    validado_nutri boolean,
    estado character varying
);


ALTER TABLE public.planes_diarios OWNER TO postgres;

--
-- Name: planes_diarios_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.planes_diarios_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.planes_diarios_id_seq OWNER TO postgres;

--
-- Name: planes_diarios_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.planes_diarios_id_seq OWNED BY public.planes_diarios.id;


--
-- Name: planes_nutricionales; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.planes_nutricionales (
    id integer NOT NULL,
    client_id integer NOT NULL,
    nutricionista_id integer,
    genero integer NOT NULL,
    edad integer NOT NULL,
    peso double precision NOT NULL,
    talla double precision NOT NULL,
    nivel_actividad double precision NOT NULL,
    objetivo character varying NOT NULL,
    es_contingencia_ia boolean,
    calorias_ia_base double precision,
    fecha_creacion timestamp without time zone,
    observaciones character varying,
    status character varying,
    validated_by_id integer,
    validated_at timestamp without time zone
);


ALTER TABLE public.planes_nutricionales OWNER TO postgres;

--
-- Name: planes_nutricionales_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.planes_nutricionales_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.planes_nutricionales_id_seq OWNER TO postgres;

--
-- Name: planes_nutricionales_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.planes_nutricionales_id_seq OWNED BY public.planes_nutricionales.id;


--
-- Name: plato_ingredientes; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.plato_ingredientes (
    id integer NOT NULL,
    plato_id integer NOT NULL,
    alimento_id integer NOT NULL,
    gramos double precision NOT NULL,
    orden integer DEFAULT 0 NOT NULL,
    notas character varying(255),
    CONSTRAINT plato_ingredientes_gramos_check CHECK ((gramos > (0)::double precision))
);


ALTER TABLE public.plato_ingredientes OWNER TO postgres;

--
-- Name: plato_ingredientes_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.plato_ingredientes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.plato_ingredientes_id_seq OWNER TO postgres;

--
-- Name: plato_ingredientes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.plato_ingredientes_id_seq OWNED BY public.plato_ingredientes.id;


--
-- Name: platos; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.platos (
    id integer NOT NULL,
    nombre character varying(255) NOT NULL,
    nombre_normalizado character varying(255) NOT NULL,
    tipo_plato character varying(50) DEFAULT 'cualquiera'::character varying,
    preparacion json,
    nota text,
    origen character varying(50) DEFAULT 'manual'::character varying,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.platos OWNER TO postgres;

--
-- Name: platos_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.platos_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.platos_id_seq OWNER TO postgres;

--
-- Name: platos_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.platos_id_seq OWNED BY public.platos.id;


--
-- Name: preferencias_alimentos; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.preferencias_alimentos (
    id integer NOT NULL,
    client_id integer NOT NULL,
    alimento character varying(200) NOT NULL,
    frecuencia integer,
    puntuacion double precision,
    calorias double precision,
    proteinas double precision,
    carbohidratos double precision,
    grasas double precision,
    ultima_vez timestamp without time zone DEFAULT now(),
    created_at timestamp without time zone DEFAULT now(),
    es_favorito smallint DEFAULT 0 NOT NULL
);


ALTER TABLE public.preferencias_alimentos OWNER TO postgres;

--
-- Name: preferencias_alimentos_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.preferencias_alimentos_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.preferencias_alimentos_id_seq OWNER TO postgres;

--
-- Name: preferencias_alimentos_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.preferencias_alimentos_id_seq OWNED BY public.preferencias_alimentos.id;


--
-- Name: preferencias_ejercicios; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.preferencias_ejercicios (
    id integer NOT NULL,
    client_id integer NOT NULL,
    ejercicio character varying(200) NOT NULL,
    frecuencia integer,
    puntuacion double precision,
    calorias_quemadas double precision,
    ultima_vez timestamp without time zone DEFAULT now(),
    created_at timestamp without time zone DEFAULT now()
);


ALTER TABLE public.preferencias_ejercicios OWNER TO postgres;

--
-- Name: preferencias_ejercicios_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.preferencias_ejercicios_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.preferencias_ejercicios_id_seq OWNER TO postgres;

--
-- Name: preferencias_ejercicios_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.preferencias_ejercicios_id_seq OWNED BY public.preferencias_ejercicios.id;


--
-- Name: progreso_calorias; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.progreso_calorias (
    id integer NOT NULL,
    client_id integer NOT NULL,
    fecha date NOT NULL,
    calorias_consumidas integer,
    calorias_quemadas integer NOT NULL,
    proteinas_consumidas double precision,
    carbohidratos_consumidos double precision,
    grasas_consumidas double precision,
    deficit_superavit integer,
    created_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.progreso_calorias OWNER TO postgres;

--
-- Name: progreso_calorias_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.progreso_calorias_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.progreso_calorias_id_seq OWNER TO postgres;

--
-- Name: progreso_calorias_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.progreso_calorias_id_seq OWNED BY public.progreso_calorias.id;


--
-- Name: receta_ingredientes; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.receta_ingredientes (
    id integer NOT NULL,
    receta_id integer NOT NULL,
    alimento_id integer NOT NULL,
    gramos double precision NOT NULL
);


ALTER TABLE public.receta_ingredientes OWNER TO postgres;

--
-- Name: receta_ingredientes_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.receta_ingredientes_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.receta_ingredientes_id_seq OWNER TO postgres;

--
-- Name: receta_ingredientes_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.receta_ingredientes_id_seq OWNED BY public.receta_ingredientes.id;


--
-- Name: recetas; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.recetas (
    id integer NOT NULL,
    nombre character varying(255) NOT NULL,
    nombre_normalizado character varying(255) NOT NULL,
    descripcion text,
    fuente character varying(255),
    es_estimada boolean,
    porcion_g double precision,
    calorias_porcion double precision,
    proteina_porcion double precision,
    carbohidratos_porcion double precision,
    grasas_porcion double precision,
    created_at timestamp with time zone DEFAULT now()
);


ALTER TABLE public.recetas OWNER TO postgres;

--
-- Name: recetas_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.recetas_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.recetas_id_seq OWNER TO postgres;

--
-- Name: recetas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.recetas_id_seq OWNED BY public.recetas.id;


--
-- Name: roles; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.roles (
    id integer NOT NULL,
    name character varying NOT NULL,
    description character varying
);


ALTER TABLE public.roles OWNER TO postgres;

--
-- Name: roles_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.roles_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.roles_id_seq OWNER TO postgres;

--
-- Name: roles_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.roles_id_seq OWNED BY public.roles.id;


--
-- Name: rutinas; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.rutinas (
    id integer NOT NULL,
    nombre character varying(255) NOT NULL,
    descripcion text,
    perfil_tipo character varying(16),
    nivel character varying(32),
    grupo_muscular character varying(128),
    tiempo_min integer,
    series_config text,
    origen character varying(32) DEFAULT 'llm'::character varying NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone
);


ALTER TABLE public.rutinas OWNER TO postgres;

--
-- Name: rutinas_ejercicios; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.rutinas_ejercicios (
    id integer NOT NULL,
    rutina_id integer NOT NULL,
    ejercicio_id character varying(100) NOT NULL,
    orden integer DEFAULT 1 NOT NULL,
    series integer DEFAULT 3 NOT NULL,
    reps integer DEFAULT 12 NOT NULL,
    descanso_s integer DEFAULT 60 NOT NULL,
    peso_sugerido_kg double precision,
    notas text
);


ALTER TABLE public.rutinas_ejercicios OWNER TO postgres;

--
-- Name: rutinas_ejercicios_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.rutinas_ejercicios_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.rutinas_ejercicios_id_seq OWNER TO postgres;

--
-- Name: rutinas_ejercicios_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.rutinas_ejercicios_id_seq OWNED BY public.rutinas_ejercicios.id;


--
-- Name: rutinas_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.rutinas_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.rutinas_id_seq OWNER TO postgres;

--
-- Name: rutinas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.rutinas_id_seq OWNED BY public.rutinas.id;


--
-- Name: sugerencias_guardadas; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sugerencias_guardadas (
    id integer NOT NULL,
    client_id integer NOT NULL,
    tipo character varying(20) NOT NULL,
    nombre character varying(255) NOT NULL,
    ingredientes json,
    preparacion json,
    macros character varying(255),
    nota text,
    completada boolean,
    fecha_guardado timestamp without time zone NOT NULL
);


ALTER TABLE public.sugerencias_guardadas OWNER TO postgres;

--
-- Name: sugerencias_guardadas_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.sugerencias_guardadas_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.sugerencias_guardadas_id_seq OWNER TO postgres;

--
-- Name: sugerencias_guardadas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.sugerencias_guardadas_id_seq OWNED BY public.sugerencias_guardadas.id;


--
-- Name: users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.users (
    id integer NOT NULL,
    first_name character varying NOT NULL,
    last_name_paternal character varying NOT NULL,
    last_name_maternal character varying NOT NULL,
    email character varying NOT NULL,
    hashed_password character varying NOT NULL,
    role_id integer NOT NULL,
    role_name character varying NOT NULL,
    is_active boolean NOT NULL,
    profile_picture_url character varying
);


ALTER TABLE public.users OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.users_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.users_id_seq OWNER TO postgres;

--
-- Name: users_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.users_id_seq OWNED BY public.users.id;


--
-- Name: workout_logs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.workout_logs (
    id integer NOT NULL,
    client_id integer NOT NULL,
    ejercicio character varying NOT NULL,
    series integer NOT NULL,
    reps integer NOT NULL,
    peso_kg double precision,
    created_at timestamp without time zone NOT NULL,
    calorias_quemadas double precision,
    session_duration_min double precision,
    intensity character varying(50)
);


ALTER TABLE public.workout_logs OWNER TO postgres;

--
-- Name: workout_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.workout_logs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.workout_logs_id_seq OWNER TO postgres;

--
-- Name: workout_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.workout_logs_id_seq OWNED BY public.workout_logs.id;


--
-- Name: workout_session_ejercicios; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.workout_session_ejercicios (
    id integer NOT NULL,
    session_id integer NOT NULL,
    ejercicio_id character varying(100) NOT NULL,
    orden integer DEFAULT 1 NOT NULL,
    series_completadas integer,
    reps_completadas integer,
    peso_kg double precision,
    duracion_s integer,
    calorias_quemadas double precision,
    notas text
);


ALTER TABLE public.workout_session_ejercicios OWNER TO postgres;

--
-- Name: workout_session_ejercicios_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.workout_session_ejercicios_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.workout_session_ejercicios_id_seq OWNER TO postgres;

--
-- Name: workout_session_ejercicios_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.workout_session_ejercicios_id_seq OWNED BY public.workout_session_ejercicios.id;


--
-- Name: workout_sessions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.workout_sessions (
    id integer NOT NULL,
    client_id integer NOT NULL,
    rutina_id integer,
    nombre_rutina character varying(255),
    fecha date NOT NULL,
    duracion_min integer,
    calorias_quemadas double precision,
    intensity character varying(16),
    notas text,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.workout_sessions OWNER TO postgres;

--
-- Name: workout_sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.workout_sessions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.workout_sessions_id_seq OWNER TO postgres;

--
-- Name: workout_sessions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.workout_sessions_id_seq OWNED BY public.workout_sessions.id;


--
-- Name: alertas_salud id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alertas_salud ALTER COLUMN id SET DEFAULT nextval('public.alertas_salud_id_seq'::regclass);


--
-- Name: alimento_alias id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimento_alias ALTER COLUMN id SET DEFAULT nextval('public.alimento_alias_id_seq'::regclass);


--
-- Name: alimento_unidades id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimento_unidades ALTER COLUMN id SET DEFAULT nextval('public.alimento_unidades_id_seq'::regclass);


--
-- Name: alimentos id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimentos ALTER COLUMN id SET DEFAULT nextval('public.alimentos_id_seq'::regclass);


--
-- Name: alimentos_sin_resolver id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimentos_sin_resolver ALTER COLUMN id SET DEFAULT nextval('public.alimentos_sin_resolver_id_seq'::regclass);


--
-- Name: app_cache_alimentos id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_alimentos ALTER COLUMN id SET DEFAULT nextval('public.app_cache_alimentos_id_seq'::regclass);


--
-- Name: app_cache_platos id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_platos ALTER COLUMN id SET DEFAULT nextval('public.app_cache_platos_id_seq'::regclass);


--
-- Name: app_cache_rutinas id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_rutinas ALTER COLUMN id SET DEFAULT nextval('public.app_cache_rutinas_id_seq'::regclass);


--
-- Name: auditoria_admin id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auditoria_admin ALTER COLUMN id SET DEFAULT nextval('public.auditoria_admin_id_seq'::regclass);


--
-- Name: clients id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.clients ALTER COLUMN id SET DEFAULT nextval('public.clients_id_seq'::regclass);


--
-- Name: comida_registros id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.comida_registros ALTER COLUMN id SET DEFAULT nextval('public.comida_registros_id_seq'::regclass);


--
-- Name: ejercicio_alias id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ejercicio_alias ALTER COLUMN id SET DEFAULT nextval('public.ejercicio_alias_id_seq'::regclass);


--
-- Name: ejercicio_musculo id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ejercicio_musculo ALTER COLUMN id SET DEFAULT nextval('public.ejercicio_musculo_id_seq'::regclass);


--
-- Name: historial_imc id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historial_imc ALTER COLUMN id SET DEFAULT nextval('public.historial_imc_id_seq'::regclass);


--
-- Name: historial_peso id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historial_peso ALTER COLUMN id SET DEFAULT nextval('public.historial_peso_id_seq'::regclass);


--
-- Name: historial_recomendaciones id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historial_recomendaciones ALTER COLUMN id SET DEFAULT nextval('public.historial_recomendaciones_id_seq'::regclass);


--
-- Name: metas_usuario id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metas_usuario ALTER COLUMN id SET DEFAULT nextval('public.metas_usuario_id_seq'::regclass);


--
-- Name: password_resets id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.password_resets ALTER COLUMN id SET DEFAULT nextval('public.password_resets_id_seq'::regclass);


--
-- Name: planes_diarios id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.planes_diarios ALTER COLUMN id SET DEFAULT nextval('public.planes_diarios_id_seq'::regclass);


--
-- Name: planes_nutricionales id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.planes_nutricionales ALTER COLUMN id SET DEFAULT nextval('public.planes_nutricionales_id_seq'::regclass);


--
-- Name: plato_ingredientes id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.plato_ingredientes ALTER COLUMN id SET DEFAULT nextval('public.plato_ingredientes_id_seq'::regclass);


--
-- Name: platos id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.platos ALTER COLUMN id SET DEFAULT nextval('public.platos_id_seq'::regclass);


--
-- Name: preferencias_alimentos id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.preferencias_alimentos ALTER COLUMN id SET DEFAULT nextval('public.preferencias_alimentos_id_seq'::regclass);


--
-- Name: preferencias_ejercicios id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.preferencias_ejercicios ALTER COLUMN id SET DEFAULT nextval('public.preferencias_ejercicios_id_seq'::regclass);


--
-- Name: progreso_calorias id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.progreso_calorias ALTER COLUMN id SET DEFAULT nextval('public.progreso_calorias_id_seq'::regclass);


--
-- Name: receta_ingredientes id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.receta_ingredientes ALTER COLUMN id SET DEFAULT nextval('public.receta_ingredientes_id_seq'::regclass);


--
-- Name: recetas id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.recetas ALTER COLUMN id SET DEFAULT nextval('public.recetas_id_seq'::regclass);


--
-- Name: roles id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.roles ALTER COLUMN id SET DEFAULT nextval('public.roles_id_seq'::regclass);


--
-- Name: rutinas id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rutinas ALTER COLUMN id SET DEFAULT nextval('public.rutinas_id_seq'::regclass);


--
-- Name: rutinas_ejercicios id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rutinas_ejercicios ALTER COLUMN id SET DEFAULT nextval('public.rutinas_ejercicios_id_seq'::regclass);


--
-- Name: sugerencias_guardadas id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sugerencias_guardadas ALTER COLUMN id SET DEFAULT nextval('public.sugerencias_guardadas_id_seq'::regclass);


--
-- Name: users id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users ALTER COLUMN id SET DEFAULT nextval('public.users_id_seq'::regclass);


--
-- Name: workout_logs id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workout_logs ALTER COLUMN id SET DEFAULT nextval('public.workout_logs_id_seq'::regclass);


--
-- Name: workout_session_ejercicios id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workout_session_ejercicios ALTER COLUMN id SET DEFAULT nextval('public.workout_session_ejercicios_id_seq'::regclass);


--
-- Name: workout_sessions id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workout_sessions ALTER COLUMN id SET DEFAULT nextval('public.workout_sessions_id_seq'::regclass);


--
-- Name: alembic_version alembic_version_pkc; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alembic_version
    ADD CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num);


--
-- Name: alertas_salud alertas_salud_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alertas_salud
    ADD CONSTRAINT alertas_salud_pkey PRIMARY KEY (id);


--
-- Name: alimento_alias alimento_alias_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimento_alias
    ADD CONSTRAINT alimento_alias_pkey PRIMARY KEY (id);


--
-- Name: alimento_unidades alimento_unidades_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimento_unidades
    ADD CONSTRAINT alimento_unidades_pkey PRIMARY KEY (id);


--
-- Name: alimentos alimentos_nombre_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimentos
    ADD CONSTRAINT alimentos_nombre_key UNIQUE (nombre);


--
-- Name: alimentos alimentos_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimentos
    ADD CONSTRAINT alimentos_pkey PRIMARY KEY (id);


--
-- Name: alimentos_sin_resolver alimentos_sin_resolver_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimentos_sin_resolver
    ADD CONSTRAINT alimentos_sin_resolver_pkey PRIMARY KEY (id);


--
-- Name: app_cache_alimentos app_cache_alimentos_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_alimentos
    ADD CONSTRAINT app_cache_alimentos_pkey PRIMARY KEY (id);


--
-- Name: app_cache_platos app_cache_platos_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_platos
    ADD CONSTRAINT app_cache_platos_pkey PRIMARY KEY (id);


--
-- Name: app_cache_rutinas app_cache_rutinas_cache_key_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_rutinas
    ADD CONSTRAINT app_cache_rutinas_cache_key_key UNIQUE (cache_key);


--
-- Name: app_cache_rutinas app_cache_rutinas_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_rutinas
    ADD CONSTRAINT app_cache_rutinas_pkey PRIMARY KEY (id);


--
-- Name: auditoria_admin auditoria_admin_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auditoria_admin
    ADD CONSTRAINT auditoria_admin_pkey PRIMARY KEY (id);


--
-- Name: clients clients_dni_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.clients
    ADD CONSTRAINT clients_dni_key UNIQUE (dni);


--
-- Name: clients clients_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.clients
    ADD CONSTRAINT clients_pkey PRIMARY KEY (id);


--
-- Name: comida_registros comida_registros_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.comida_registros
    ADD CONSTRAINT comida_registros_pkey PRIMARY KEY (id);


--
-- Name: ejercicio_alias ejercicio_alias_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ejercicio_alias
    ADD CONSTRAINT ejercicio_alias_pkey PRIMARY KEY (id);


--
-- Name: ejercicio_musculo ejercicio_musculo_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ejercicio_musculo
    ADD CONSTRAINT ejercicio_musculo_pkey PRIMARY KEY (id);


--
-- Name: ejercicios ejercicios_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ejercicios
    ADD CONSTRAINT ejercicios_pkey PRIMARY KEY (id);


--
-- Name: historial_imc historial_imc_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historial_imc
    ADD CONSTRAINT historial_imc_pkey PRIMARY KEY (id);


--
-- Name: historial_peso historial_peso_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historial_peso
    ADD CONSTRAINT historial_peso_pkey PRIMARY KEY (id);


--
-- Name: historial_recomendaciones historial_recomendaciones_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historial_recomendaciones
    ADD CONSTRAINT historial_recomendaciones_pkey PRIMARY KEY (id);


--
-- Name: metas_usuario metas_usuario_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metas_usuario
    ADD CONSTRAINT metas_usuario_pkey PRIMARY KEY (id);


--
-- Name: password_resets password_resets_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.password_resets
    ADD CONSTRAINT password_resets_pkey PRIMARY KEY (id);


--
-- Name: planes_diarios planes_diarios_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.planes_diarios
    ADD CONSTRAINT planes_diarios_pkey PRIMARY KEY (id);


--
-- Name: planes_nutricionales planes_nutricionales_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.planes_nutricionales
    ADD CONSTRAINT planes_nutricionales_pkey PRIMARY KEY (id);


--
-- Name: plato_ingredientes plato_ingredientes_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.plato_ingredientes
    ADD CONSTRAINT plato_ingredientes_pkey PRIMARY KEY (id);


--
-- Name: platos platos_nombre_normalizado_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.platos
    ADD CONSTRAINT platos_nombre_normalizado_key UNIQUE (nombre_normalizado);


--
-- Name: platos platos_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.platos
    ADD CONSTRAINT platos_pkey PRIMARY KEY (id);


--
-- Name: preferencias_alimentos preferencias_alimentos_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.preferencias_alimentos
    ADD CONSTRAINT preferencias_alimentos_pkey PRIMARY KEY (id);


--
-- Name: preferencias_ejercicios preferencias_ejercicios_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.preferencias_ejercicios
    ADD CONSTRAINT preferencias_ejercicios_pkey PRIMARY KEY (id);


--
-- Name: progreso_calorias progreso_calorias_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.progreso_calorias
    ADD CONSTRAINT progreso_calorias_pkey PRIMARY KEY (id);


--
-- Name: receta_ingredientes receta_ingredientes_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.receta_ingredientes
    ADD CONSTRAINT receta_ingredientes_pkey PRIMARY KEY (id);


--
-- Name: recetas recetas_nombre_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.recetas
    ADD CONSTRAINT recetas_nombre_key UNIQUE (nombre);


--
-- Name: recetas recetas_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.recetas
    ADD CONSTRAINT recetas_pkey PRIMARY KEY (id);


--
-- Name: roles roles_name_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.roles
    ADD CONSTRAINT roles_name_key UNIQUE (name);


--
-- Name: roles roles_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.roles
    ADD CONSTRAINT roles_pkey PRIMARY KEY (id);


--
-- Name: rutinas_ejercicios rutinas_ejercicios_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rutinas_ejercicios
    ADD CONSTRAINT rutinas_ejercicios_pkey PRIMARY KEY (id);


--
-- Name: rutinas rutinas_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rutinas
    ADD CONSTRAINT rutinas_pkey PRIMARY KEY (id);


--
-- Name: sugerencias_guardadas sugerencias_guardadas_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sugerencias_guardadas
    ADD CONSTRAINT sugerencias_guardadas_pkey PRIMARY KEY (id);


--
-- Name: users users_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_pkey PRIMARY KEY (id);


--
-- Name: workout_logs workout_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workout_logs
    ADD CONSTRAINT workout_logs_pkey PRIMARY KEY (id);


--
-- Name: workout_session_ejercicios workout_session_ejercicios_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workout_session_ejercicios
    ADD CONSTRAINT workout_session_ejercicios_pkey PRIMARY KEY (id);


--
-- Name: workout_sessions workout_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workout_sessions
    ADD CONSTRAINT workout_sessions_pkey PRIMARY KEY (id);


--
-- Name: idx_cache_expires; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_cache_expires ON public.app_cache_alimentos USING btree (expires_at);


--
-- Name: idx_cache_food; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_cache_food ON public.app_cache_alimentos USING btree (food_normalized);


--
-- Name: idx_cache_food_user; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_cache_food_user ON public.app_cache_alimentos USING btree (food_normalized, user_id);


--
-- Name: idx_cache_plato_expires; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_cache_plato_expires ON public.app_cache_platos USING btree (expires_at);


--
-- Name: idx_cache_plato_norm; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_cache_plato_norm ON public.app_cache_platos USING btree (plato_normalized);


--
-- Name: idx_cache_plato_user; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_cache_plato_user ON public.app_cache_platos USING btree (plato_normalized, user_id);


--
-- Name: idx_cache_rutinas_expires; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_cache_rutinas_expires ON public.app_cache_rutinas USING btree (expires_at);


--
-- Name: idx_cache_rutinas_key; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_cache_rutinas_key ON public.app_cache_rutinas USING btree (cache_key);


--
-- Name: idx_cache_rutinas_user; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_cache_rutinas_user ON public.app_cache_rutinas USING btree (user_id);


--
-- Name: idx_comida_registros_client_fecha; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_comida_registros_client_fecha ON public.comida_registros USING btree (client_id, fecha);


--
-- Name: idx_comida_registros_fecha; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_comida_registros_fecha ON public.comida_registros USING btree (fecha);


--
-- Name: idx_pr_code; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_pr_code ON public.password_resets USING btree (reset_code);


--
-- Name: idx_pr_email; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_pr_email ON public.password_resets USING btree (email);


--
-- Name: idx_rut_ej_ejercicio; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_rut_ej_ejercicio ON public.rutinas_ejercicios USING btree (ejercicio_id);


--
-- Name: idx_rut_ej_orden; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_rut_ej_orden ON public.rutinas_ejercicios USING btree (rutina_id, orden);


--
-- Name: idx_rut_ej_rutina; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_rut_ej_rutina ON public.rutinas_ejercicios USING btree (rutina_id);


--
-- Name: idx_rutinas_grupo; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_rutinas_grupo ON public.rutinas USING btree (grupo_muscular);


--
-- Name: idx_rutinas_perfil; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_rutinas_perfil ON public.rutinas USING btree (perfil_tipo);


--
-- Name: idx_sin_resolver_estado; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_sin_resolver_estado ON public.alimentos_sin_resolver USING btree (estado);


--
-- Name: idx_sin_resolver_nombre; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_sin_resolver_nombre ON public.alimentos_sin_resolver USING btree (nombre_normalizado);


--
-- Name: idx_sin_resolver_user; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_sin_resolver_user ON public.alimentos_sin_resolver USING btree (user_id);


--
-- Name: idx_ws_client; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ws_client ON public.workout_sessions USING btree (client_id);


--
-- Name: idx_ws_fecha; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ws_fecha ON public.workout_sessions USING btree (client_id, fecha);


--
-- Name: idx_ws_rutina; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_ws_rutina ON public.workout_sessions USING btree (rutina_id);


--
-- Name: idx_wse_ejercicio; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_wse_ejercicio ON public.workout_session_ejercicios USING btree (ejercicio_id);


--
-- Name: idx_wse_orden; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_wse_orden ON public.workout_session_ejercicios USING btree (session_id, orden);


--
-- Name: idx_wse_session; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX idx_wse_session ON public.workout_session_ejercicios USING btree (session_id);


--
-- Name: ix_alertas_salud_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_alertas_salud_id ON public.alertas_salud USING btree (id);


--
-- Name: ix_alimento_alias_normalizado; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_alimento_alias_normalizado ON public.alimento_alias USING btree (alias_normalizado);


--
-- Name: ix_alimentos_nombre_normalizado; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_alimentos_nombre_normalizado ON public.alimentos USING btree (nombre_normalizado);


--
-- Name: ix_auditoria_admin_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_auditoria_admin_id ON public.auditoria_admin USING btree (id);


--
-- Name: ix_clients_email; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_clients_email ON public.clients USING btree (email);


--
-- Name: ix_clients_flutter_uid; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_clients_flutter_uid ON public.clients USING btree (flutter_uid);


--
-- Name: ix_clients_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_clients_id ON public.clients USING btree (id);


--
-- Name: ix_comida_registros_client_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_comida_registros_client_id ON public.comida_registros USING btree (client_id);


--
-- Name: ix_comida_registros_fecha; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_comida_registros_fecha ON public.comida_registros USING btree (fecha);


--
-- Name: ix_comida_registros_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_comida_registros_id ON public.comida_registros USING btree (id);


--
-- Name: ix_ejercicios_nombre_normalizado; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_ejercicios_nombre_normalizado ON public.ejercicios USING btree (nombre_normalizado);


--
-- Name: ix_historial_client; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_historial_client ON public.historial_recomendaciones USING btree (client_id);


--
-- Name: ix_historial_imc_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_historial_imc_id ON public.historial_imc USING btree (id);


--
-- Name: ix_historial_peso_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_historial_peso_id ON public.historial_peso USING btree (id);


--
-- Name: ix_historial_plato; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_historial_plato ON public.historial_recomendaciones USING btree (plato_id);


--
-- Name: ix_metas_usuario_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_metas_usuario_id ON public.metas_usuario USING btree (id);


--
-- Name: ix_planes_diarios_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_planes_diarios_id ON public.planes_diarios USING btree (id);


--
-- Name: ix_planes_nutricionales_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_planes_nutricionales_id ON public.planes_nutricionales USING btree (id);


--
-- Name: ix_plato_ing_plato; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_plato_ing_plato ON public.plato_ingredientes USING btree (plato_id);


--
-- Name: ix_platos_nombre_norm; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_platos_nombre_norm ON public.platos USING btree (nombre_normalizado);


--
-- Name: ix_preferencias_alimentos_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_preferencias_alimentos_id ON public.preferencias_alimentos USING btree (id);


--
-- Name: ix_preferencias_ejercicios_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_preferencias_ejercicios_id ON public.preferencias_ejercicios USING btree (id);


--
-- Name: ix_progreso_calorias_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_progreso_calorias_id ON public.progreso_calorias USING btree (id);


--
-- Name: ix_receta_ingredientes_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_receta_ingredientes_id ON public.receta_ingredientes USING btree (id);


--
-- Name: ix_recetas_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_recetas_id ON public.recetas USING btree (id);


--
-- Name: ix_recetas_nombre_normalizado; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_recetas_nombre_normalizado ON public.recetas USING btree (nombre_normalizado);


--
-- Name: ix_roles_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_roles_id ON public.roles USING btree (id);


--
-- Name: ix_sugerencias_guardadas_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_sugerencias_guardadas_id ON public.sugerencias_guardadas USING btree (id);


--
-- Name: ix_users_email; Type: INDEX; Schema: public; Owner: postgres
--

CREATE UNIQUE INDEX ix_users_email ON public.users USING btree (email);


--
-- Name: ix_users_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_users_id ON public.users USING btree (id);


--
-- Name: ix_workout_logs_client_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_workout_logs_client_id ON public.workout_logs USING btree (client_id);


--
-- Name: ix_workout_logs_created_at; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_workout_logs_created_at ON public.workout_logs USING btree (created_at);


--
-- Name: ix_workout_logs_id; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_workout_logs_id ON public.workout_logs USING btree (id);


--
-- Name: alertas_salud alertas_salud_atendido_por_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alertas_salud
    ADD CONSTRAINT alertas_salud_atendido_por_id_fkey FOREIGN KEY (atendido_por_id) REFERENCES public.users(id);


--
-- Name: alertas_salud alertas_salud_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alertas_salud
    ADD CONSTRAINT alertas_salud_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id);


--
-- Name: alimento_alias alimento_alias_alimento_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimento_alias
    ADD CONSTRAINT alimento_alias_alimento_id_fkey FOREIGN KEY (alimento_id) REFERENCES public.alimentos(id) ON DELETE CASCADE;


--
-- Name: alimento_unidades alimento_unidades_alimento_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimento_unidades
    ADD CONSTRAINT alimento_unidades_alimento_id_fkey FOREIGN KEY (alimento_id) REFERENCES public.alimentos(id) ON DELETE CASCADE;


--
-- Name: alimentos_sin_resolver alimentos_sin_resolver_reporter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimentos_sin_resolver
    ADD CONSTRAINT alimentos_sin_resolver_reporter_id_fkey FOREIGN KEY (reporter_id) REFERENCES public.users(id) ON DELETE SET NULL;


--
-- Name: alimentos_sin_resolver alimentos_sin_resolver_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.alimentos_sin_resolver
    ADD CONSTRAINT alimentos_sin_resolver_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.clients(id) ON DELETE SET NULL;


--
-- Name: app_cache_alimentos app_cache_alimentos_alimento_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_alimentos
    ADD CONSTRAINT app_cache_alimentos_alimento_id_fkey FOREIGN KEY (alimento_id) REFERENCES public.alimentos(id) ON DELETE CASCADE;


--
-- Name: app_cache_alimentos app_cache_alimentos_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_alimentos
    ADD CONSTRAINT app_cache_alimentos_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.clients(id) ON DELETE SET NULL;


--
-- Name: app_cache_platos app_cache_platos_plato_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_platos
    ADD CONSTRAINT app_cache_platos_plato_id_fkey FOREIGN KEY (plato_id) REFERENCES public.platos(id) ON DELETE CASCADE;


--
-- Name: app_cache_platos app_cache_platos_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_platos
    ADD CONSTRAINT app_cache_platos_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.clients(id) ON DELETE SET NULL;


--
-- Name: app_cache_rutinas app_cache_rutinas_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.app_cache_rutinas
    ADD CONSTRAINT app_cache_rutinas_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.clients(id) ON DELETE CASCADE;


--
-- Name: auditoria_admin auditoria_admin_admin_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.auditoria_admin
    ADD CONSTRAINT auditoria_admin_admin_id_fkey FOREIGN KEY (admin_id) REFERENCES public.users(id);


--
-- Name: clients clients_assigned_coach_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.clients
    ADD CONSTRAINT clients_assigned_coach_id_fkey FOREIGN KEY (assigned_coach_id) REFERENCES public.users(id);


--
-- Name: clients clients_assigned_nutri_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.clients
    ADD CONSTRAINT clients_assigned_nutri_id_fkey FOREIGN KEY (assigned_nutri_id) REFERENCES public.users(id);


--
-- Name: comida_registros comida_registros_alimento_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.comida_registros
    ADD CONSTRAINT comida_registros_alimento_id_fkey FOREIGN KEY (alimento_id) REFERENCES public.alimentos(id);


--
-- Name: comida_registros comida_registros_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.comida_registros
    ADD CONSTRAINT comida_registros_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id);


--
-- Name: comida_registros comida_registros_plato_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.comida_registros
    ADD CONSTRAINT comida_registros_plato_id_fkey FOREIGN KEY (plato_id) REFERENCES public.platos(id);


--
-- Name: ejercicio_alias ejercicio_alias_ejercicio_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ejercicio_alias
    ADD CONSTRAINT ejercicio_alias_ejercicio_id_fkey FOREIGN KEY (ejercicio_id) REFERENCES public.ejercicios(id) ON DELETE CASCADE;


--
-- Name: ejercicio_musculo ejercicio_musculo_ejercicio_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ejercicio_musculo
    ADD CONSTRAINT ejercicio_musculo_ejercicio_id_fkey FOREIGN KEY (ejercicio_id) REFERENCES public.ejercicios(id) ON DELETE CASCADE;


--
-- Name: historial_imc historial_imc_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historial_imc
    ADD CONSTRAINT historial_imc_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id);


--
-- Name: historial_peso historial_peso_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historial_peso
    ADD CONSTRAINT historial_peso_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id);


--
-- Name: historial_recomendaciones historial_recomendaciones_plato_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.historial_recomendaciones
    ADD CONSTRAINT historial_recomendaciones_plato_id_fkey FOREIGN KEY (plato_id) REFERENCES public.platos(id) ON DELETE SET NULL;


--
-- Name: metas_usuario metas_usuario_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.metas_usuario
    ADD CONSTRAINT metas_usuario_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id);


--
-- Name: planes_diarios planes_diarios_plan_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.planes_diarios
    ADD CONSTRAINT planes_diarios_plan_id_fkey FOREIGN KEY (plan_id) REFERENCES public.planes_nutricionales(id);


--
-- Name: planes_nutricionales planes_nutricionales_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.planes_nutricionales
    ADD CONSTRAINT planes_nutricionales_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id);


--
-- Name: planes_nutricionales planes_nutricionales_nutricionista_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.planes_nutricionales
    ADD CONSTRAINT planes_nutricionales_nutricionista_id_fkey FOREIGN KEY (nutricionista_id) REFERENCES public.users(id);


--
-- Name: planes_nutricionales planes_nutricionales_validated_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.planes_nutricionales
    ADD CONSTRAINT planes_nutricionales_validated_by_id_fkey FOREIGN KEY (validated_by_id) REFERENCES public.users(id);


--
-- Name: plato_ingredientes plato_ingredientes_alimento_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.plato_ingredientes
    ADD CONSTRAINT plato_ingredientes_alimento_id_fkey FOREIGN KEY (alimento_id) REFERENCES public.alimentos(id) ON DELETE RESTRICT;


--
-- Name: plato_ingredientes plato_ingredientes_plato_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.plato_ingredientes
    ADD CONSTRAINT plato_ingredientes_plato_id_fkey FOREIGN KEY (plato_id) REFERENCES public.platos(id) ON DELETE CASCADE;


--
-- Name: preferencias_alimentos preferencias_alimentos_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.preferencias_alimentos
    ADD CONSTRAINT preferencias_alimentos_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id);


--
-- Name: preferencias_ejercicios preferencias_ejercicios_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.preferencias_ejercicios
    ADD CONSTRAINT preferencias_ejercicios_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id);


--
-- Name: progreso_calorias progreso_calorias_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.progreso_calorias
    ADD CONSTRAINT progreso_calorias_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id);


--
-- Name: receta_ingredientes receta_ingredientes_alimento_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.receta_ingredientes
    ADD CONSTRAINT receta_ingredientes_alimento_id_fkey FOREIGN KEY (alimento_id) REFERENCES public.alimentos(id) ON DELETE CASCADE;


--
-- Name: receta_ingredientes receta_ingredientes_receta_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.receta_ingredientes
    ADD CONSTRAINT receta_ingredientes_receta_id_fkey FOREIGN KEY (receta_id) REFERENCES public.recetas(id) ON DELETE CASCADE;


--
-- Name: rutinas_ejercicios rutinas_ejercicios_ejercicio_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rutinas_ejercicios
    ADD CONSTRAINT rutinas_ejercicios_ejercicio_id_fkey FOREIGN KEY (ejercicio_id) REFERENCES public.ejercicios(id) ON DELETE CASCADE;


--
-- Name: rutinas_ejercicios rutinas_ejercicios_rutina_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.rutinas_ejercicios
    ADD CONSTRAINT rutinas_ejercicios_rutina_id_fkey FOREIGN KEY (rutina_id) REFERENCES public.rutinas(id) ON DELETE CASCADE;


--
-- Name: sugerencias_guardadas sugerencias_guardadas_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sugerencias_guardadas
    ADD CONSTRAINT sugerencias_guardadas_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id);


--
-- Name: users users_role_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.users
    ADD CONSTRAINT users_role_id_fkey FOREIGN KEY (role_id) REFERENCES public.roles(id);


--
-- Name: workout_logs workout_logs_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workout_logs
    ADD CONSTRAINT workout_logs_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id);


--
-- Name: workout_session_ejercicios workout_session_ejercicios_ejercicio_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workout_session_ejercicios
    ADD CONSTRAINT workout_session_ejercicios_ejercicio_id_fkey FOREIGN KEY (ejercicio_id) REFERENCES public.ejercicios(id) ON DELETE CASCADE;


--
-- Name: workout_session_ejercicios workout_session_ejercicios_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workout_session_ejercicios
    ADD CONSTRAINT workout_session_ejercicios_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.workout_sessions(id) ON DELETE CASCADE;


--
-- Name: workout_sessions workout_sessions_client_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workout_sessions
    ADD CONSTRAINT workout_sessions_client_id_fkey FOREIGN KEY (client_id) REFERENCES public.clients(id) ON DELETE CASCADE;


--
-- Name: workout_sessions workout_sessions_rutina_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.workout_sessions
    ADD CONSTRAINT workout_sessions_rutina_id_fkey FOREIGN KEY (rutina_id) REFERENCES public.rutinas(id) ON DELETE SET NULL;


--
-- PostgreSQL database dump complete
--

\unrestrict 4wATf19WTADOrDnh7Ja0HwkPij6cBenfsVzWvtZtqXyaM2DIAg2BhNnRX0D8Ke4

