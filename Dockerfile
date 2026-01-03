FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.json .
COPY main.py .

# On n'impose pas de port via ENV, on laisse l'hébergeur décider.
# Cette commande CMD utilise directement la variable $PORT injectée par l'hébergeur.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
