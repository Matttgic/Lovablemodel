FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.json .
COPY main.py .

ENV PORT=8000
ENV MODEL_PATH=model.json

EXPOSE 8000

CMD ["python", "main.py"]
