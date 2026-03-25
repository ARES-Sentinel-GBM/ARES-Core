FROM python:3.11-slim

LABEL maintainer="ARES-Sentinel GBM Pipeline v2.1"
LABEL description="GBM Computational Pipeline — Nanodrone PK & Target Analysis"

WORKDIR /app

# System deps per matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p output

# Healthcheck: verifica che il modulo principale sia importabile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from modules.nanodrone_sim import NanodronePKSimulator; print('OK')" || exit 1

CMD ["python", "main.py"]
