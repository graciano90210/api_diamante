1. Base: Imagen oficial de Python (versión ligera)

FROM python:3.10-slim-buster

2. Variables de entorno

ENV PYTHONUNBUFFERED 1
ENV FLASK_APP app.py

3. Directorio de trabajo

WORKDIR /app

4. Instalación de dependencias: Copia requirements.txt e instala

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

5. Copia de archivos clave (API y modelos .pkl)

COPY app.py .
COPY modelo_diamante_v3.pkl .
COPY scaler_diamante.pkl .

6. Exponer puerto de Flask

EXPOSE 5000

7. Comando de inicio

CMD ["python", "app.py"]