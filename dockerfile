# Базовый образ с Python 3.9 (легковесный)
FROM python:3.9-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями ПЕРВЫМ (для кэширования слоев)
COPY requirements.txt .
COPY app.py .
COPY Modeling ./Modeling
COPY templates ./templates

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt