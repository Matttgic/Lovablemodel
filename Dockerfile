FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.json .
COPY main.py .

# 'exec' stabilise le processus
# '--workers 1' réduit la consommation de mémoire pour éviter les crashs
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
