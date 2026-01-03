FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.json .
COPY main.py .

# On retire la ligne ENV PORT=8000 pour laisser l'hébergeur décider
ENV MODEL_PATH=model.json

# On expose le port 8080 qui semble être celui par défaut de votre hébergeur
EXPOSE 8080

CMD ["python", "main.py"]
