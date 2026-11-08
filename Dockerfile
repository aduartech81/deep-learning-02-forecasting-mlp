# API de inferencia: MLP multisalida por horizonte (7, 14, 21, 28 días)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3
WORKDIR /srv

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app/ ./app/

EXPOSE 8000
# Render y otros servicios inyectan PORT; en local se usa 8000
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
