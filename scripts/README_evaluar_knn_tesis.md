# `evaluar_knn_tesis.py` — evaluación reproducible del KNN para la tesis

Calcula similitud coseno top-3, ILD (Intra-list Diversity) y cobertura del
catálogo para el recomendador de alimentos, con semilla fija en todos los
puntos de aleatoriedad (generación de escenarios y desempate interno del
muestreo). No modifica `app/services/ml_service.py` ni ningún `.pkl`; carga
el mismo `recomendador_knn.pkl` de producción en modo lectura.

## Por qué existe un script aparte

`ml_service.py` (producción) tiene dos versiones históricas relevantes para
la tesis:

- **`sin_mmr`** — commit `e55f94a`, tag `tesis-validacion`. Es la versión que
  usaron los usuarios piloto en la validación de junio 2026.
- **`con_mmr`** — commit `a2a772a`, tag `tesis-mmr`. Es la que reemplazó a la
  anterior en producción (MMR λ=0.5 + softmax T=0.20).

Ninguna de las dos es reproducible tal cual en producción: su semilla interna
depende de la fecha (`sin_mmr`) o del minuto del reloj (`con_mmr`), así que
correr la misma evaluación dos veces da números distintos. Este script
congela ambas lógicas con una semilla constante, solo para evaluación.

## Comandos exactos para reproducir la tabla de la tesis

```bash
python scripts/evaluar_knn_tesis.py --version sin_mmr --base reales
python scripts/evaluar_knn_tesis.py --version con_mmr --base reales
python scripts/evaluar_knn_tesis.py --version sin_mmr --base simulado --n 100
```

| Versión | Base | Semilla | Similitud top-3 | ILD | Cobertura |
|---|---|---|---|---|---|
| sin_mmr | 524 días reales (clientes 55-69, ≤2026-05-29) | 42 | 93.0% | 0.0808 | 42.1% (456 / 1082 alimentos) |
| con_mmr | 524 días reales (clientes 55-69, ≤2026-05-29) | 42 | 91.6% | 0.1084 | 42.3% (458 / 1082 alimentos) |
| sin_mmr | 100 escenarios simulados | 42 | 83.6% | 0.1699 | 15.6% (169 / 1082 alimentos) |

## Parámetros

| Flag | Valores | Descripción |
|---|---|---|
| `--version` | `sin_mmr` \| `con_mmr` | Qué lógica de ranking usar |
| `--base` | `reales` \| `simulado` | 524 días reales de clientes 55-69 (≤2026-05-29), o escenarios de déficit simulados |
| `--n` | entero, default 100 | N° de escenarios, solo aplica con `--base simulado` |
| `--seed` | entero, default 42 | Semilla única para generar los escenarios y para el desempate interno del muestreo |

## Nota sobre el ILD = 0.3842 documentado antes

El valor anterior (`ARQUITECTURA_SISTEMA.md`, previo a esta actualización) no
es reproducible: se generó antes de que la semilla interna del recomendador
se fijara, así que dependía de la fecha/minuto en que se corrió el script
original. La tabla de esta evaluación reemplaza a ese valor como la
referencia estable de aquí en adelante.
