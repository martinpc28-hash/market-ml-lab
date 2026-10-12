"""Cloud Function (gen2) que lanza el Cloud Run Job de reentrenamiento.

Cloud Scheduler la invoca con un token OIDC; la función no es pública.
"""
import os

import functions_framework
import google.auth
from google.auth.transport.requests import AuthorizedSession

PROJECT = os.environ["PROJECT_ID"]
REGION = os.environ["REGION"]
JOB = os.environ["JOB_NAME"]


@functions_framework.http
def trigger(request):
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    url = f"https://run.googleapis.com/v2/projects/{PROJECT}/locations/{REGION}/jobs/{JOB}:run"
    body = {"overrides": {"containerOverrides": [{"env": [{"name": "TRIGGER", "value": "scheduler"}]}]}}
    r = AuthorizedSession(creds).post(url, json=body, timeout=30)
    if r.status_code >= 300:
        print(f"Error lanzando el job: {r.status_code} {r.text}")
        return {"error": r.text}, r.status_code
    op = r.json().get("name")
    print(f"Job {JOB} lanzado: {op}")
    return {"started": JOB, "operation": op}, 200
