-- Agrega campo es_favorito a preferencias_alimentos
ALTER TABLE preferencias_alimentos
    ADD COLUMN IF NOT EXISTS es_favorito SMALLINT NOT NULL DEFAULT 0;
