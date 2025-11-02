FROM python:3.10-slim-buster
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP app.py
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY modelo_diamante_v3.pkl .
COPY scaler_diamante.pkl .
EXPOSE 5000
CMD ["python", "app.py"]
```
