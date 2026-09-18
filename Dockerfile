# ─── Базовый образ ────────────────────────────────────────────────────────────
# slim достаточен: TensorFlow тянет свои бинарники через pip.
# python:3.11 — последняя версия, официально поддерживаемая TF 2.16+.
FROM python:3.11-slim

# ─── Системные зависимости ────────────────────────────────────────────────────
# libgomp1  — OpenMP, нужен TensorFlow/NumPy для многопоточности.
# curl      — удобно для health-check внутри контейнера (опционально).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Рабочая директория ───────────────────────────────────────────────────────
WORKDIR /app

# ─── Зависимости Python ───────────────────────────────────────────────────────
# Копируем отдельным слоем — Docker кэширует его до изменения requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Исходный код приложения ──────────────────────────────────────────────────
COPY smoking_pipeline.py .
COPY main.py             .
COPY templates/          templates/

# ─── Артефакты обученной модели ───────────────────────────────────────────────
# Папка должна содержать: model.keras  preprocessing.joblib  config.json
COPY smoking_nn_pipeline/ smoking_nn_pipeline/

# ─── Порт ─────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ─── Health-check ─────────────────────────────────────────────────────────────
# Docker (и оркестраторы) используют его, чтобы знать, готов ли контейнер.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# ─── Запуск ───────────────────────────────────────────────────────────────────
# --workers 1: TensorFlow не thread-safe между форками — один воркер.
# --host 0.0.0.0: слушаем на всех интерфейсах контейнера.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
