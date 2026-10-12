# Interfaz de MLflow de solo lectura: descarga la base de GCS al arrancar.
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
RUN pip install --no-cache-dir mlflow==3.17.0 google-cloud-storage==3.17.0
COPY start.sh /start.sh
RUN useradd --create-home app
USER app
CMD ["sh", "/start.sh"]
