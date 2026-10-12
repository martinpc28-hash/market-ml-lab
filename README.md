# Market ML Lab

Dos modelos de machine learning con su ciclo de vida completo en Google Cloud: entrenamiento con seguimiento en MLflow, Model Registry con champion/challenger, API de inferencia y reentrenamiento programado.

| | Modelo 1: VIX y SPY | Modelo 2: cartera de ETFs |
|---|---|---|
| **Qué predice** | Nivel del VIX y rentabilidad del SPY a **1 día, 1 semana y 1 mes**, con intervalo del 80 % | Qué ETFs comprar. Compiten 4 estrategias (2 con ML, una de ellas alimentada por el modelo 1) y solo gana la que supera a la más simple |
| **Tipo de inferencia** | Online: cada llamada calcula las features de hoy | Batch: un job diario genera la cartera y la API la sirve |
| **Modelos** | 18 × `HistGradientBoostingRegressor` (6 objetivos × mediana, p10 y p90) | Regresor de ranking + clasificador de meta-labeling, sobre un panel ETF × día |
| **Métrica de promoción** | `val_skill_mean`: mejora media sobre un baseline ingenuo | `strategy_sharpe`: Sharpe de la estrategia que pasó el filtro (ML o momentum) |

La web tiene una pestaña por modelo, otra que explica cómo se usa el ML y otra con el estado del Model Registry.

**Demo en vivo:** [web y API](https://marketml-api-rs3ndcpqda-uc.a.run.app) · [Swagger](https://marketml-api-rs3ndcpqda-uc.a.run.app/docs) · [MLflow](https://mlflow-ui-rs3ndcpqda-uc.a.run.app). La primera carga tarda unos 10 segundos porque Cloud Run escala a cero.

## Arquitectura

```mermaid
flowchart LR
    subgraph Programado
        SCH[Cloud Scheduler<br/>L-V 19:00 NY] -->|OIDC| FN[Cloud Function<br/>retrain-trigger]
        FN -->|jobs.run| JOB[Cloud Run Job<br/>marketml-train]
    end

    CBOE[(Cboe<br/>VIX, VIX3M<br/>FRED de respaldo)] --> JOB
    YF[(Yahoo Finance<br/>SPY y ETFs)] --> JOB
    JOB -->|runs, métricas,<br/>modelos, registry| GCS[(Cloud Storage<br/>mlflow.db + artefactos)]
    JOB -->|cartera batch| GCS

    subgraph Servicio
        API[Cloud Run<br/>FastAPI + web]
        UI[Cloud Run<br/>MLflow UI solo lectura]
    end

    GCS -->|models:/...@champion| API
    CBOE -->|datos de hoy| API
    YF -->|datos de hoy| API
    GCS --> UI
    USER((Usuario)) --> API
    USER --> UI
```

### Ciclo del reentrenamiento diario

1. **Cloud Scheduler** llama a la **Cloud Function** con un token OIDC. La función no es pública.
2. La función lanza el **Cloud Run Job**.
3. El job descarga `mlflow.db` de GCS, baja los precios y entrena los dos modelos con validación walk-forward. Registra en MLflow parámetros, métricas, gráficos y el modelo.
4. **Champion/challenger:** cada modelo nuevo se registra con el alias `challenger`. Solo pasa a `champion` si su métrica de validación no empeora la del campeón más allá de una tolerancia.
5. El job carga el modelo `etf-allocator@champion` y genera la cartera del día (inferencia batch) en `gs://.../predictions/allocation/latest.json`.
6. Sube `mlflow.db` con `if_generation_match`: si alguien la modificó entretanto, la subida falla en lugar de pisar los cambios.
7. La **API** comprueba cada 5 minutos si la base cambió y recarga el modelo `vix-spy-forecaster@champion`.

### Por qué SQLite en GCS y no Cloud SQL

MLflow necesita una base de datos para el registry. Cloud SQL costaría unos 10 USD al mes con la instancia siempre encendida. Aquí solo hay **un escritor**, el job (con `parallelism=1`), así que basta un fichero SQLite en GCS con control de concurrencia optimista. La API y la interfaz de MLflow solo leen. El bucket tiene versionado activado, así que cualquier versión anterior de la base se puede recuperar. Si hubiera varios equipos entrenando a la vez, el siguiente paso sería pasar a Cloud SQL, cambiando solo el `--backend-store-uri`.

## Cómo se usa el ML

### Modelo 1: VIX y SPY a 1 día, 1 semana y 1 mes

- **Datos:** VIX y VIX3M descargados directamente de **Cboe**, que es quien calcula el índice (CSV público con el cierre del mismo día). El VIX3M de Cboe empieza en septiembre de 2009, así que la historia anterior se completa con FRED (`VXVCLS`), que republica los mismos cierres de Cboe; en los días que se solapan la diferencia es 0. Así el entrenamiento conserva la crisis de 2008. Si Cboe falla se usa FRED (con un día de retraso), y si FRED también falla, Yahoo. El SPY ajustado por dividendos viene de Yahoo Finance, porque ni Cboe ni FRED publican ETFs. Cada run de MLflow guarda la fuente de cada serie en las etiquetas `data_source.*`.
- **Objetivo:** para cada horizonte h ∈ {1, 5, 21} días hábiles se predice `log(VIX[t+h]/VIX[t])` y `log(SPY[t+h]/SPY[t])`. Se predice el cambio y no el nivel, porque el cambio es estacionario y el nivel no.
- **Variables (18):** nivel y cambios del VIX, z-score del VIX, estructura temporal VIX/VIX3M, volatilidad realizada a 5, 21 y 63 días, prima de volatilidad (VIX menos volatilidad realizada), momentum, drawdown y distancia a las medias de 50 y 200 días del SPY. Todas usan solo información disponible al cierre del día t; `tests/test_features.py` lo verifica cambiando el futuro y comprobando que las variables no se alteran.
- **Incertidumbre:** además de la mediana, dos modelos de **regresión cuantílica** estiman los percentiles 10 y 90, que forman un intervalo del 80 %.
- **Validación walk-forward:** 5 cortes temporales. Siempre se entrena con el pasado y se evalúa con el futuro, dejando un hueco de h días para que los objetivos no se solapen.
- **Baseline:** para el VIX, "no cambia"; para el SPY, "la rentabilidad mediana histórica". El *skill* indica cuánto menos se equivoca el modelo.

Resultados de validación (fuera de muestra, 2007-2026, modelo campeón v3):

| Objetivo | Skill sobre el baseline | Acierto de dirección | Cobertura del intervalo del 80 % |
|---|---|---|---|
| VIX a 1 día | +2,2 % | 57 % | 78 % |
| VIX a 1 semana | +4,6 % | 60 % | 78 % |
| VIX a 1 mes | **+9,3 %** | **65 %** | 76 % |
| SPY a 1 día | -0,1 % | 55 % (baseline 55 %) | 85 % |
| SPY a 1 semana | 0,0 % | 62 % (baseline 62 %) | 86 % |
| SPY a 1 mes | +1,6 % | 69 % (baseline 69 %) | 82 % |

**Lectura honesta:** el VIX tiene una parte predecible porque vuelve a su media, y el modelo la captura. La variable más importante a 1 mes es la estructura temporal VIX/VIX3M (43 % de la importancia). La rentabilidad del SPY es casi impredecible, como cabe esperar en un mercado eficiente. Aun así, el modelo del SPY aporta algo útil: un intervalo de incertidumbre que se ensancha cuando el VIX está alto, como se ve en el simulador de la web.

> Ajuste de hiperparámetros: con árboles más profundos (`max_depth=3`, `min_samples_leaf=80`) el skill del SPY a 1 mes era -18,7 %, porque el modelo sobreajustaba el ruido. Con una regularización fuerte (`max_depth=2`, `min_samples_leaf=500`, `l2=10`) sube a +0,7 % y la cobertura de los intervalos pasa del 67 % al 83 %.

### Modelo 2: cartera de ETFs. Compiten cuatro estrategias

- **Universo:** SPY, QQQ, IWM, EFA, EEM (acciones); TLT, IEF, LQD, HYG (bonos); GLD, DBC (materias primas); VNQ (REITs).
- **Cartera:** top 4 equiponderado, con rebalanceo cada 21 días hábiles.

| # | Estrategia | Cómo elige |
|---|---|---|
| 1 | **Equiponderada** | Los 12 ETFs a partes iguales (equivale, en media, a elegir al azar) |
| 2 | **Momentum 12-1** | Los 4 ETFs que más han subido en 12 meses, sin contar el último |
| 3 | **Momentum + ML (meta-labeling)** | El momentum propone 6 candidatos. Un clasificador estima la probabilidad de que cada uno rinda por encima de la mediana, es decir, de que no sea un falso positivo, y se quedan los 4 más probables. **Usa como variables las predicciones del modelo 1**: cambio previsto del VIX, escenario alto del VIX y escenario malo del SPY |
| 4 | **Ranking ML** | Un regresor predice el puesto (percentil 0-1) de cada ETF entre los 12 y se compran los 4 mejores |

**Filtro antes de producción (regla de parsimonia).** Las cuatro estrategias se evalúan en el mismo backtest walk-forward. Se parte de la más simple, y una más compleja solo la sustituye si **mejora su Sharpe en más de 0,05**; empatar no basta. Al ranking ML se le exige además que sus 4 favoritos rindan más que sus 4 últimos. Se registra la estrategia ganadora, y la web y la API explican por qué ganó. Los tests de `tests/test_features.py` comprueban el filtro.

**Resultado actual: en producción está la regla de momentum.**

| Estrategia (2011-2026, walk-forward) | Rentabilidad anual | Sharpe | Máx. caída |
|---|---|---|---|
| **Momentum 12-1** (en producción) | 11,3 % | **1,02** | -16,4 % |
| Momentum + ML (meta-labeling con señales del VIX) | 10,4 % | 0,86 | -22,9 % |
| Equiponderada | 7,8 % | 0,84 | -19,9 % |
| Ranking ML | 9,9 % | 0,71 | -26,7 % |
| SPY (referencia) | 15,5 % | 1,05 | -23,1 % |

**Historia de este modelo.** Es lo más útil del proyecto:

1. **Primera versión: objetivo = rentabilidad relativa a 1 mes.** Rendía menos que elegir 4 ETFs al azar (Sharpe 0,43). Sus favoritos rendían menos que sus últimos. El sistema champion/challenger lo promovió igualmente, porque la métrica de promoción, el IC medio, era engañosa: la sostenían unos pocos días de 2011. **Lección:** la métrica de promoción tiene que comparar con alternativas simples.
2. **Objetivo corregido: predecir el puesto y no la rentabilidad.** El modelo pasa a ordenar: IC de 0,12, positivo en 14 de 16 años, y sus 4 favoritos baten a sus 4 últimos en un 6,7 % anual. Pero elige activos más volátiles y su Sharpe queda por debajo del momentum ([experiments/allocator_variants.py](experiments/allocator_variants.py)).
3. **Meta-labeling: momentum + ML para filtrar falsos positivos.** Sin las señales del VIX, el clasificador apenas distingue (AUC de 0,545) y el Sharpe baja a 0,79-0,83 ([experiments/meta_labeling.py](experiments/meta_labeling.py)).
4. **Uso del modelo del VIX en el de cartera** ([experiments/vix_into_allocation.py](experiments/vix_into_allocation.py)). Para no filtrar información futura, se usan las predicciones del modelo 1 hechas fuera de muestra.
   - **Como variables del meta-labeling: ayuda.** El Sharpe sube a 0,86 y la precisión del 53,7 % al 55,1 %. No basta para batir al momentum.
   - **Como freno de riesgo (pasarse a defensivo si el modelo prevé estrés): empeora todas las carteras.** El Sharpe del momentum baja de 0,94 a entre 0,57 y 0,84, según la variante. La razón es lo que sabe el modelo del VIX: que vuelve a su media. Cuando prevé estrés suele ser justo después de un pico, el momento en que las acciones rebotan, así que salir del mercado entonces sale caro. Por eso no se implementó.
5. **Conclusión:** con estas variables, el ML no supera a una regla de momentum bien conocida, y el sistema lo detecta solo en cada reentrenamiento. Si algún día una estrategia con ML la supera por el margen exigido, pasará a producción automáticamente.

## API

URL base: `https://<marketml-api>.run.app`. La documentación interactiva está en `/docs`.

```bash
# Predicción de VIX y SPY a 1 día, 1 semana y 1 mes (inferencia online)
curl https://<marketml-api>.run.app/api/forecast
```

```json
{
  "as_of": "2026-10-09",
  "current": {"vix": 14.84, "spy": 778.57},
  "forecast": [
    {"horizon": "1d", "days": 1,  "vix": {"pred": 14.89, "p10": 14.09, "p90": 15.97}, "spy_return_pct": {"pred": 0.053, "p10": -0.659, "p90": 0.714}},
    {"horizon": "1w", "days": 5,  "vix": {"pred": 15.16, "p10": 13.65, "p90": 17.73}, "spy_return_pct": {"pred": 0.239, "p10": -1.803, "p90": 1.799}},
    {"horizon": "1m", "days": 21, "vix": {"pred": 15.51, "p10": 13.19, "p90": 20.40}, "spy_return_pct": {"pred": 1.125, "p10": -3.394, "p90": 3.964}}
  ],
  "model": {"name": "vix-spy-forecaster", "version": 1, "trained_until": "2026-10-09"}
}
```

```bash
# Escenario: ¿qué pasa si el VIX salta a 35 con la curva en backwardation?
curl -X POST https://<marketml-api>.run.app/api/forecast/scenario \
  -H "content-type: application/json" \
  -d '{"overrides": {"vix": 35, "term_structure": 1.1}}'

# Cartera recomendada (inferencia batch)
curl https://<marketml-api>.run.app/api/allocation
```

| Endpoint | Descripción |
|---|---|
| `GET /api/forecast` | Predicción online con el modelo campeón |
| `POST /api/forecast/scenario` | La misma predicción, cambiando variables a mano |
| `GET /api/forecast/explain` | Métricas, importancia de variables y parámetros |
| `GET /api/forecast/backtest?target=vix_1m` | Predicción frente a realidad fuera de muestra |
| `GET /api/allocation` | Cartera del día (batch) |
| `GET /api/allocation/backtest` | Curva de capital y métricas del backtest |
| `GET /api/models` | Versiones, alias y decisiones champion/challenger |

## Estructura

```
src/marketml/
  config.py       configuración por variables de entorno
  data.py         Cboe (VIX, VIX3M) con FRED de respaldo + Yahoo Finance (SPY, ETFs)
  features.py     variables sin lookahead
  forecaster.py   modelo 1 + pyfunc de MLflow
  allocator.py    modelo 2 + backtest + filtro frente a momentum + inferencia batch
  store.py        sincronización de mlflow.db con GCS y champion/challenger
  train.py        punto de entrada del Cloud Run Job
api/              FastAPI + web (static/index.html)
mlflow_server/    imagen de la interfaz de MLflow
functions/        Cloud Function que lanza el job
deploy/deploy.sh  despliegue completo e idempotente
experiments/      experimentos: variantes del ranking, meta-labeling y modelo del VIX en la cartera
tests/            tests de no-lookahead y de salida del modelo
```

## Ejecutar en local

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt
PYTHONPATH=src python -m marketml.train                      # entrena y registra en ./.marketml
PYTHONPATH="src;." uvicorn api.main:app --port 8080          # en Linux/macOS: PYTHONPATH=src:.
PYTHONPATH=src pytest -q tests
mlflow ui --backend-store-uri sqlite:///.marketml/mlflow.db  # para ver los runs
```

## Desplegar en Google Cloud

```bash
PROJECT_ID=mi-proyecto ./deploy/deploy.sh
```

El script crea el bucket (con versionado), el repositorio de Artifact Registry y 5 cuentas de servicio con mínimo privilegio: el trainer escribe en el bucket, y la API y MLflow solo leen. También construye las imágenes con Cloud Build, despliega el job, la API, la interfaz de MLflow, la Cloud Function y Cloud Scheduler, y lanza el primer entrenamiento.

Coste aproximado: menos de 1-2 USD al mes. Todo escala a cero y el job tarda unos 3 minutos al día.

```bash
# Forzar un reentrenamiento ahora
gcloud scheduler jobs run marketml-retrain --location us-central1
```

---

*Proyecto educativo de MLOps. No es asesoramiento financiero.*
