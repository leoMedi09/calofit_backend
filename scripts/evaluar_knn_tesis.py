"""
Evaluación reproducible del recomendador KNN para la tesis — similitud coseno
top-3, ILD (Intra-list Diversity) y cobertura del catálogo, con semilla fija
en TODOS los puntos de aleatoriedad del proceso (generación de escenarios Y
desempate interno del muestreo).

Compara dos versiones históricas del recomendador, congeladas en este archivo
tal como existían en sus commits originales. NO se modifica ni se importa
app/services/ml_service.py para la lógica de ranking — solo se reutiliza el
mismo recomendador_knn.pkl de producción (catálogo INS/CENAN 2017 +
OpenFoodFacts, 1082 alimentos), en modo lectura:

  --version sin_mmr   Commit e55f94a — muestreo ponderado Efraimidis-Spirakis.
                       Es la versión que usaron los usuarios piloto en la
                       validación de junio 2026 (ver tag git "tesis-validacion").
  --version con_mmr    Commit a2a772a — MMR (λ=0.5) + softmax (T=0.20). Es la
                       que reemplazó a la anterior en producción (ver tag git
                       "tesis-mmr"). Aquí está congelada con la MISMA fórmula
                       que ml_service.py, cambiando solo su semilla interna
                       (que en producción depende del minuto del reloj) por
                       un valor constante, para que el resultado sea
                       reproducible. No es una re-implementación: es una copia
                       literal de la lógica de obtener_recomendaciones() tal
                       como existía en a2a772a.

Aviso de mantenimiento: este script es una fotografía fija de esos dos
commits. Si ml_service.py cambia más adelante (nuevos filtros, otro boost,
etc.), "con_mmr" dejará de reflejar la producción actual y habrá que congelar
una nueva versión aquí — es intencional, para que los números de la tesis no
cambien solos con cada futuro commit.

Uso:
    python scripts/evaluar_knn_tesis.py --version sin_mmr --base reales
    python scripts/evaluar_knn_tesis.py --version con_mmr --base reales
    python scripts/evaluar_knn_tesis.py --version sin_mmr --base simulado --n 100

Resultados que este script reproduce exactamente con semilla=42 (ver README
en scripts/README_evaluar_knn_tesis.md):

    sin_mmr  reales   (524 días, <=2026-05-29)  -> similitud 93.0%  ILD 0.0808  cobertura 42.1% (456 alimentos)
    con_mmr  reales   (524 días, <=2026-05-29)  -> similitud 91.6%  ILD 0.1084  cobertura 42.3% (458 alimentos)
    sin_mmr  simulado (n=100, seed=42)          -> similitud 83.6%  ILD 0.1699  cobertura 15.6% (169 alimentos)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import date, datetime

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.alimentos_ux_filters import es_alimento_bloqueado_ia, nombre_coincide_exclusion

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "models", "ai_models")
RECOMENDADOR_PKL = os.path.join(MODELS_DIR, "recomendador_knn.pkl")
FEATURES_KNN = ["calorias_100g", "proteina_100g", "carbohindratos_100g", "grasas_100g"]
TOTAL_CATALOGO = 1082
CORTE_524_DIAS = date(2026, 5, 29)
SEED_POR_DEFECTO = 42

_OMEGA3_ESPECIES = frozenset(
    {"caballa", "lisa", "mero", "ojo de uva", "tollo", "cabrilla", "anchoveta", "trucha", "salmon", "sardina"}
)


def _cargar_catalogo():
    paquete = joblib.load(RECOMENDADOR_PKL)
    return paquete["modelo_knn"], paquete["scaler"], paquete["df_alimentos"]


class RecomendadorSinMMR:
    """Copia congelada de RecomendadorAlimentosKNN.obtener_recomendaciones()
    tal como existía en el commit e55f94a (muestreo ponderado
    Efraimidis-Spirakis, sin MMR). La única diferencia con el original: la
    semilla, que en e55f94a dependía de get_peru_date(), aquí es constante.
    """

    def __init__(self, seed: int = SEED_POR_DEFECTO):
        self._knn, self._scaler, self._df = _cargar_catalogo()
        self._seed = seed

    def obtener_recomendaciones(
        self, calorias_faltantes, prote_faltante, carbo_faltante, grasa_faltante, n_recomendaciones=3, contexto=None
    ):
        calorias_faltantes = max(50, min(calorias_faltantes, 900))
        prote_faltante = max(0, prote_faltante)
        carbo_faltante = max(0, carbo_faltante)
        grasa_faltante = max(0, grasa_faltante)
        vector = [calorias_faltantes, prote_faltante, carbo_faltante, grasa_faltante]

        vector_scaled = self._scaler.transform([vector])
        n_pool = min(max(n_recomendaciones * 20, 60), len(self._df))
        distancias, idx = self._knn.kneighbors(vector_scaled, n_neighbors=max(1, n_pool))

        candidatos = []
        for i, row_idx in enumerate(idx[0]):
            row = self._df.iloc[row_idx]
            nombre = str(row["alimento"])
            if es_alimento_bloqueado_ia(nombre):
                continue
            similitud = round((1 - distancias[0][i]) * 100, 1)
            candidatos.append(
                {
                    "alimento": nombre,
                    "calorias_100g": row["calorias_100g"],
                    "proteina_100g": row["proteina_100g"],
                    "carbohindratos_100g": row["carbohindratos_100g"],
                    "grasas_100g": row["grasas_100g"],
                    "similitud": similitud,
                }
            )
        if not candidatos:
            return []

        _ctx = (contexto or "").lower()
        _omega_activo = any(kw in _ctx for kw in ("omega", "marino", "pescado", "mariscos")) or grasa_faltante > 3.0
        if _omega_activo:
            for c in candidatos:
                if any(esp in c["alimento"].lower() for esp in _OMEGA3_ESPECIES):
                    c["similitud"] = min(99.9, round(c["similitud"] * 1.25, 1))

        rng = random.Random(self._seed)
        keys = [rng.random() ** (1.0 / max(c["similitud"], 0.1)) for c in candidatos]
        candidatos = [c for _, c in sorted(zip(keys, candidatos), reverse=True)]

        vistos, resultados = set(), []
        for c in candidatos:
            key = c["alimento"].lower().strip()
            if key in vistos:
                continue
            vistos.add(key)
            resultados.append(c)
            if len(resultados) >= n_recomendaciones:
                break
        return resultados

    @property
    def df_alimentos(self):
        return self._df

    @property
    def scaler(self):
        return self._scaler


class RecomendadorConMMR:
    """Copia congelada de RecomendadorAlimentosKNN.obtener_recomendaciones()
    tal como existe en el commit a2a772a (MMR λ=0.5 + softmax T=0.20),
    la versión en producción. Única diferencia con el original: la semilla
    interna, que en producción depende del minuto del reloj
    (int(timestamp() // 60)) y aquí es constante, para poder reproducir el
    resultado exacto en cualquier momento.
    """

    _MMR_LAMBDA = 0.5
    _MMR_TEMPERATURE = 0.20

    def __init__(self, seed: int = SEED_POR_DEFECTO):
        self._knn, self._scaler, self._df = _cargar_catalogo()
        self._seed = seed

    def obtener_recomendaciones(
        self, calorias_faltantes, prote_faltante, carbo_faltante, grasa_faltante, n_recomendaciones=3, contexto=None
    ):
        calorias_faltantes = max(50, min(calorias_faltantes, 900))
        prote_faltante = max(0, prote_faltante)
        carbo_faltante = max(0, carbo_faltante)
        grasa_faltante = max(0, grasa_faltante)
        vector = [calorias_faltantes, prote_faltante, carbo_faltante, grasa_faltante]

        vector_scaled = self._scaler.transform([vector])
        n_pool = min(max(n_recomendaciones * 20, 60), len(self._df))
        distancias, idx = self._knn.kneighbors(vector_scaled, n_neighbors=max(1, n_pool))

        candidatos = []
        for i, row_idx in enumerate(idx[0]):
            row = self._df.iloc[row_idx]
            nombre = str(row["alimento"])
            if es_alimento_bloqueado_ia(nombre):
                continue
            similitud = round((1 - distancias[0][i]) * 100, 1)
            vec = self._scaler.transform([[row["calorias_100g"], row["proteina_100g"], row["carbohindratos_100g"], row["grasas_100g"]]])[0]
            candidatos.append(
                {
                    "alimento": nombre,
                    "calorias_100g": row["calorias_100g"],
                    "proteina_100g": row["proteina_100g"],
                    "carbohindratos_100g": row["carbohindratos_100g"],
                    "grasas_100g": row["grasas_100g"],
                    "similitud": similitud,
                    "_vec": vec,
                }
            )
        if not candidatos:
            return []

        _ctx = (contexto or "").lower()
        _omega_activo = any(kw in _ctx for kw in ("omega", "marino", "pescado", "mariscos")) or grasa_faltante > 3.0
        if _omega_activo:
            for c in candidatos:
                if any(esp in c["alimento"].lower() for esp in _OMEGA3_ESPECIES):
                    c["similitud"] = min(99.9, round(c["similitud"] * 1.25, 1))

        rng = np.random.default_rng(self._seed)
        vistos, resultados, restantes = set(), [], candidatos.copy()
        while restantes and len(resultados) < n_recomendaciones:
            scores, cands = [], []
            for c in restantes:
                key = c["alimento"].lower().strip()
                if key in vistos:
                    continue
                rel = c["similitud"] / 100.0
                if not resultados:
                    div_penalty = 0.0
                else:
                    div_penalty = max(
                        float(
                            np.dot(c["_vec"], s["_vec"]) / (np.linalg.norm(c["_vec"]) * np.linalg.norm(s["_vec"]) + 1e-9)
                        )
                        for s in resultados
                    )
                scores.append(self._MMR_LAMBDA * rel - (1 - self._MMR_LAMBDA) * div_penalty)
                cands.append(c)
            if not cands:
                break
            scores_arr = np.array(scores)
            logits = (scores_arr - scores_arr.max()) / self._MMR_TEMPERATURE
            probs = np.exp(logits)
            probs /= probs.sum()
            elegido = cands[rng.choice(len(cands), p=probs)]
            resultados.append(elegido)
            vistos.add(elegido["alimento"].lower().strip())
            restantes.remove(elegido)
        return resultados

    @property
    def df_alimentos(self):
        return self._df

    @property
    def scaler(self):
        return self._scaler


def _ild_de_lista(recos, df, scaler):
    if len(recos) < 2:
        return None
    item_vecs = []
    for r in recos:
        fila = df[df["alimento"] == r["alimento"]]
        if not fila.empty:
            item_vecs.append(scaler.transform(fila[FEATURES_KNN].values)[0])
    if len(item_vecs) < 2:
        return None
    dmat = cosine_distances(item_vecs)
    n = len(item_vecs)
    pares = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean([dmat[i][j] for i, j in pares]))


def _vectores_simulados(n: int, seed: int):
    rng = np.random.default_rng(seed)
    return [[rng.uniform(50, 900), rng.uniform(0, 60), rng.uniform(0, 120), rng.uniform(0, 40)] for _ in range(n)]


def evaluar_base_simulada(recomendador, n: int, seed: int):
    similitudes, ild_scores, vistos = [], [], set()
    for v in _vectores_simulados(n, seed):
        recos = recomendador.obtener_recomendaciones(*v, n_recomendaciones=3)
        for r in recos:
            vistos.add(r["alimento"].lower().strip())
            similitudes.append(r["similitud"])
        valor_ild = _ild_de_lista(recos, recomendador.df_alimentos, recomendador.scaler)
        if valor_ild is not None:
            ild_scores.append(valor_ild)
    return _resumen(similitudes, ild_scores, vistos)


def evaluar_base_real(recomendador, fecha_max: date):
    from app.core.database import SessionLocal
    from app.models.client import Client
    from app.models.historial import ProgresoCalorias
    from app.services.asistente.asistente_plan import obtener_plan_hoy

    db = SessionLocal()
    similitudes, ild_scores, vistos = [], [], set()
    dias = 0
    try:
        for cid in range(55, 70):
            cliente = db.query(Client).filter(Client.id == cid).first()
            if not cliente:
                continue
            edad = 25
            if cliente.birth_date:
                edad = datetime.now().year - cliente.birth_date.year
            try:
                _, plan, _ = obtener_plan_hoy(cliente, edad, db)
            except Exception:
                continue
            for r in db.query(ProgresoCalorias).filter(ProgresoCalorias.client_id == cid).all():
                if r.fecha > fecha_max:
                    continue
                rest_kcal = max(plan["calorias_dia"] - (r.calorias_consumidas or 0), 50)
                rest_prot = max(plan["proteinas_g"] - (r.proteinas_consumidas or 0), 0)
                rest_carb = max(plan["carbohidratos_g"] - (r.carbohidratos_consumidos or 0), 0)
                rest_gras = max(plan["grasas_g"] - (r.grasas_consumidas or 0), 0)
                recos = recomendador.obtener_recomendaciones(rest_kcal, rest_prot, rest_carb, rest_gras, n_recomendaciones=3)
                if not recos:
                    continue
                dias += 1
                for rc in recos:
                    vistos.add(rc["alimento"].lower().strip())
                    similitudes.append(rc["similitud"])
                valor_ild = _ild_de_lista(recos, recomendador.df_alimentos, recomendador.scaler)
                if valor_ild is not None:
                    ild_scores.append(valor_ild)
    finally:
        db.close()
    resumen = _resumen(similitudes, ild_scores, vistos)
    resumen["dias"] = dias
    return resumen


def _resumen(similitudes, ild_scores, vistos):
    sims = np.array(similitudes)
    return {
        "similitud": float(sims.mean()) if len(sims) else 0.0,
        "ild": float(np.mean(ild_scores)) if ild_scores else 0.0,
        "cobertura": len(vistos) / TOTAL_CATALOGO,
        "alimentos": len(vistos),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", choices=["sin_mmr", "con_mmr"], required=True)
    parser.add_argument("--base", choices=["reales", "simulado"], required=True)
    parser.add_argument("--n", type=int, default=100, help="N de escenarios simulados (solo con --base simulado)")
    parser.add_argument("--seed", type=int, default=SEED_POR_DEFECTO)
    args = parser.parse_args()

    Recomendador = RecomendadorSinMMR if args.version == "sin_mmr" else RecomendadorConMMR
    recomendador = Recomendador(seed=args.seed)

    if args.base == "reales":
        r = evaluar_base_real(recomendador, CORTE_524_DIAS)
        base_desc = f"{r['dias']} días reales (clientes 55-69, <=2026-05-29)"
    else:
        r = evaluar_base_simulada(recomendador, n=args.n, seed=args.seed)
        base_desc = f"{args.n} escenarios simulados"

    print(f"Versión   : {args.version}")
    print(f"Base      : {base_desc}")
    print(f"Semilla   : {args.seed}")
    print(f"Similitud top-3 promedio : {r['similitud']:.1f}%")
    print(f"ILD                      : {r['ild']:.4f}")
    print(f"Cobertura               : {r['cobertura']:.1%} ({r['alimentos']} / {TOTAL_CATALOGO} alimentos)")


if __name__ == "__main__":
    main()
