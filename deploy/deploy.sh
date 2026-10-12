#!/usr/bin/env bash
# Despliegue completo en Google Cloud. Idempotente: se puede volver a ejecutar.
#
#   PROJECT_ID=mi-proyecto ./deploy/deploy.sh
#
# Requisitos: gcloud autenticado y un proyecto con facturación activa.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT_ID="${PROJECT_ID:?Define PROJECT_ID}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-${PROJECT_ID}-mlops}"
REPO="marketml"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
APP_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/app:${TAG}"
MLFLOW_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/mlflow:${TAG}"
JOB="marketml-train"
API="marketml-api"
MLFLOW="mlflow-ui"
FUNCTION="retrain-trigger"
SCHEDULE="${SCHEDULE:-0 19 * * 1-5}"   # lunes a viernes 19:00, tras el cierre de NY

g() { gcloud --project "$PROJECT_ID" --quiet "$@"; }
sa() { echo "$1@${PROJECT_ID}.iam.gserviceaccount.com"; }
step() { echo; echo "==> $*"; }

step "Activando APIs"
g services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  cloudscheduler.googleapis.com cloudfunctions.googleapis.com storage.googleapis.com \
  iam.googleapis.com logging.googleapis.com

step "Bucket gs://${BUCKET}"
g storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1 || \
  g storage buckets create "gs://${BUCKET}" --location "$REGION" --uniform-bucket-level-access
g storage buckets update "gs://${BUCKET}" --versioning   # protege la base de MLflow

step "Artifact Registry"
g artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  g artifacts repositories create "$REPO" --repository-format docker --location "$REGION"

step "Cuentas de servicio (mínimo privilegio)"
for s in sa-trainer sa-api sa-mlflow sa-retrain-fn sa-scheduler; do
  g iam service-accounts describe "$(sa $s)" >/dev/null 2>&1 || \
    g iam service-accounts create "$s" --display-name "marketml $s"
done
sleep 10  # propagación de IAM
g storage buckets add-iam-policy-binding "gs://${BUCKET}" --member "serviceAccount:$(sa sa-trainer)" --role roles/storage.objectAdmin >/dev/null
for s in sa-api sa-mlflow; do
  g storage buckets add-iam-policy-binding "gs://${BUCKET}" --member "serviceAccount:$(sa $s)" --role roles/storage.objectViewer >/dev/null
done

step "Construyendo imágenes con Cloud Build"
g builds submit --tag "$APP_IMAGE" .
g builds submit --tag "$MLFLOW_IMAGE" mlflow_server

step "Cloud Run Job de entrenamiento"
g run jobs deploy "$JOB" --region "$REGION" --image "$APP_IMAGE" \
  --command python --args=-m,marketml.train \
  --service-account "$(sa sa-trainer)" --set-env-vars "GCS_BUCKET=${BUCKET}" \
  --cpu 2 --memory 4Gi --task-timeout 30m --max-retries 1 --parallelism 1 --tasks 1

if ! g storage objects describe "gs://${BUCKET}/mlflow/mlflow.db" >/dev/null 2>&1; then
  step "Primer entrenamiento (unos 3-5 minutos)"
  g run jobs execute "$JOB" --region "$REGION" --wait
fi

step "Interfaz de MLflow"
g run deploy "$MLFLOW" --region "$REGION" --image "$MLFLOW_IMAGE" \
  --service-account "$(sa sa-mlflow)" --set-env-vars "GCS_BUCKET=${BUCKET}" \
  --memory 1Gi --min-instances 0 --max-instances 1 --allow-unauthenticated
MLFLOW_URL=$(g run services describe "$MLFLOW" --region "$REGION" --format 'value(status.url)')

step "API de inferencia"
g run deploy "$API" --region "$REGION" --image "$APP_IMAGE" \
  --service-account "$(sa sa-api)" \
  --set-env-vars "GCS_BUCKET=${BUCKET},MLFLOW_UI_URL=${MLFLOW_URL}" \
  --memory 2Gi --cpu 1 --min-instances 0 --max-instances 3 --allow-unauthenticated
API_URL=$(g run services describe "$API" --region "$REGION" --format 'value(status.url)')

step "Cloud Function que lanza el reentrenamiento"
g functions deploy "$FUNCTION" --gen2 --region "$REGION" --runtime python312 \
  --source functions/retrain_trigger --entry-point trigger --trigger-http --no-allow-unauthenticated \
  --service-account "$(sa sa-retrain-fn)" \
  --set-env-vars "PROJECT_ID=${PROJECT_ID},REGION=${REGION},JOB_NAME=${JOB}"
g run jobs add-iam-policy-binding "$JOB" --region "$REGION" \
  --member "serviceAccount:$(sa sa-retrain-fn)" --role roles/run.jobsExecutorWithOverrides >/dev/null
g functions add-invoker-policy-binding "$FUNCTION" --region "$REGION" \
  --member "serviceAccount:$(sa sa-scheduler)" >/dev/null
FN_URL=$(g functions describe "$FUNCTION" --region "$REGION" --gen2 --format 'value(serviceConfig.uri)')

step "Cloud Scheduler: ${SCHEDULE} (America/New_York)"
SCHED_ARGS=(--location "$REGION" --schedule "$SCHEDULE" --time-zone "America/New_York"
  --uri "$FN_URL" --http-method POST --oidc-service-account-email "$(sa sa-scheduler)"
  --oidc-token-audience "$FN_URL" --attempt-deadline 60s)
if g scheduler jobs describe marketml-retrain --location "$REGION" >/dev/null 2>&1; then
  g scheduler jobs update http marketml-retrain "${SCHED_ARGS[@]}"
else
  g scheduler jobs create http marketml-retrain "${SCHED_ARGS[@]}"
fi

echo
echo "Listo"
echo "  Web y API:  ${API_URL}"
echo "  Swagger:    ${API_URL}/docs"
echo "  MLflow UI:  ${MLFLOW_URL}"
echo "  Reentrenar ahora: gcloud scheduler jobs run marketml-retrain --location ${REGION} --project ${PROJECT_ID}"
