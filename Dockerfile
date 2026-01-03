FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.json .
COPY main.py .

# On supprime EXPOSE pour éviter les conflits, l'hébergeur gérera le port via la variable $PORT
# On utilise ["sh", "-c", ...] pour garantir que la variable ${PORT} est bien lue
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
