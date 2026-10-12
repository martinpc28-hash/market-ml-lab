# Una sola imagen para la API (Cloud Run service) y el entrenamiento (Cloud Run Job).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    MARKETML_HOME=/tmp/marketml \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/

RUN useradd --create-home app
USER app

# Por defecto arranca la API; el Job sobrescribe el comando con: python -m marketml.train
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
