# Adım 1: Bağımlılıkları kurmak için bir temel imaj kullan
FROM python:3.9-slim as builder

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt

# Adım 2: Asıl uygulama imajını oluştur
FROM python:3.9-slim

WORKDIR /usr/src/app

# Builder aşamasından tekerlek dosyalarını kopyala
COPY --from=builder /usr/src/app/wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache /wheels/*

# Uygulama kodunu kopyala
COPY ./app ./app

# Uygulamayı çalıştır
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]