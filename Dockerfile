
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Libraries
RUN pip install --no-cache-dir -r requirements.txt

# Descargamos los paquetes de NLTK que Pengügle necesita para procesar el texto
RUN python -m nltk.downloader punkt stopwords

COPY . .

# Port
EXPOSE 5000

# Start Web Server
CMD ["python", "app.py"]