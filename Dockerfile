# Python image
FROM python:3.10-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Gereken dosyaları kopyala
COPY . /app

# Bağımlılıkları yükle
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama başlat
CMD ["python", "app.py"]