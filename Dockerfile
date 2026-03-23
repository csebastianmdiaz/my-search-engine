# Using a lightweight Python image to keep the container size small
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Libraries
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download specific NLTK datasets (tokenizers and stopwords) needed for the text processing pipeline
RUN python -m nltk.downloader punkt stopwords

COPY . .

# Port
EXPOSE 5000

# Start Web Server
CMD ["python", "app.py"]