import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, Field  # <--- Importaciones de Pydantic
from typing import Optional  # Aunque ya no lo usaremos en CreditData
from pydantic import ValidationError  # Para manejar errores específicos


# --- ESQUEMA DE VALIDACIÓN DE DATOS (PYDANTIC) ---
# Ahora los campos NO son 'Optional' (obligatorios).
class CreditData(BaseModel):
    # Nota: Quitamos 'Optional[int]' y 'Field(None, ...)'
    Status_Existing_Account: int = Field(..., ge=0, le=4, description="Estado de la cuenta existente (0-4)")
    Duration_in_Month: int = Field(..., ge=4, le=72, description="Duración del crédito en meses")
    Credit_History: int = Field(..., ge=0, le=4, description="Historial crediticio (0-4)")
    Purpose: int = Field(..., ge=0, le=10, description="Propósito del crédito (0-10)")
    Credit_Amount: int = Field(..., ge=250, le=18420, description="Monto del crédito")
    Savings_Account_Bonds: int = Field(..., ge=0, le=4, description="Cuenta de ahorros/bonos (0-4)")
    Present_Employment_Since: int = Field(..., ge=0, le=4, description="Empleo actual desde (0-4)")
    Installment_Rate_Percentage: int = Field(..., ge=1, le=4, description="Tasa de cuota (%)")
    Personal_Status_Sex: int = Field(..., ge=0, le=3, description="Estado personal y sexo (0-3)")
    Other_Debtors_Guarantors: int = Field(..., ge=0, le=2, description="Otros deudores/garantes (0-2)")
    Present_Residence_Since: int = Field(..., ge=1, le=4, description="Residencia actual desde (1-4)")
    Property: int = Field(..., ge=0, le=3, description="Tipo de propiedad (0-3)")
    Age_in_Years: int = Field(..., ge=19, le=75, description="Edad del cliente")
    Other_Installment_Plans: int = Field(..., ge=0, le=2, description="Otros planes de cuotas (0-2)")
    Housing: int = Field(..., ge=0, le=2, description="Situación de vivienda (0-2)")
    Number_of_Existing_Credits: int = Field(..., ge=1, le=4, description="Número de créditos existentes (1-4)")
    Job: int = Field(..., ge=0, le=3, description="Tipo de trabajo (0-3)")
    Number_of_Dependents: int = Field(..., ge=1, le=2, description="Número de dependientes (1-2)")
    Telephone: int = Field(..., ge=0, le=1, description="Tiene teléfono (0/1)")
    Foreign_Worker: int = Field(..., ge=0, le=1, description="Trabajador extranjero (0/1)")

    class Config:
        # Prohíbe campos adicionales no definidos, asegura que solo se envíen los 20.
        extra = 'forbid'


# 1. Creamos la aplicación Flask
app = Flask(__name__)

# 2. Cargamos nuestros archivos .pkl (¡una sola vez al inicio!)
print("Cargando el modelo...")
model = joblib.load('modelo_diamante_v3.pkl')
print("Cargando el escalador...")
scaler = joblib.load('scaler_diamante.pkl')
print("¡Modelo y escalador listos!")

# 3. Definimos las 20 columnas que nuestro modelo espera
# ¡El orden debe ser EXACTO!
columnas_modelo = [
    'Status_Existing_Account', 'Duration_in_Month', 'Credit_History', 'Purpose',
    'Credit_Amount', 'Savings_Account_Bonds', 'Present_Employment_Since',
    'Installment_Rate_Percentage', 'Personal_Status_Sex', 'Other_Debtors_Guarantors',
    'Present_Residence_Since', 'Property', 'Age_in_Years', 'Other_Installment_Plans',
    'Housing', 'Number_of_Existing_Credits', 'Job', 'Number_of_Dependents',
    'Telephone', 'Foreign_Worker'
]


# 4. Creamos la "ruta" o "endpoint" de predicción
@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # A. Recibimos el JSON
        data_json = request.get_json(force=True)

        # B. VALIDACIÓN: Usar Pydantic para validar los datos.
        validated_data = CreditData(**data_json)

        # C. Convertir el objeto Pydantic validado a un diccionario de Python
        datos_dict = validated_data.model_dump()

    except ValidationError as e:
        # Manejamos errores específicos de Pydantic: campos faltantes, tipos incorrectos, etc.
        return jsonify({
            'error': 'Error de validación en los datos de entrada. Faltan campos requeridos.',
            'detalles': e.errors()
        }), 400

    except Exception as e:
        # Manejo de errores generales de JSON o request
        return jsonify({'error': 'Error en el formato de la solicitud JSON.', 'detalles': str(e)}), 400

    try:
        # D. Convertimos el diccionario validado a un DataFrame de Pandas
        df_cliente = pd.DataFrame([datos_dict])
        df_cliente = df_cliente[columnas_modelo]  # Asegura el orden

        # E. ¡Usamos el ESCALADOR!
        datos_escalados = scaler.transform(df_cliente)

        # F. ¡Hacemos la PREDICCIÓN!
        prediccion = model.predict(datos_escalados)
        # Probabilidad de ser 1 (Alto Riesgo)
        prediccion_proba = model.predict_proba(datos_escalados)[0][1]

        # G. Interpretamos la respuesta
        resultado = int(prediccion[0])
        probabilidad = float(f'{prediccion_proba:.4f}')

        if resultado == 0:
            status = 'Aprobado'
        else:
            status = 'Rechazado (Alto Riesgo)'

        # H. Devolvemos la respuesta
        return jsonify({
            'prediccion_numerica': resultado,
            'status_credito': status,
            'probabilidad_rechazo': probabilidad
        })

    except Exception as e:
        # Si algo falla DENTRO de la lógica del modelo/escalador
        return jsonify({'error': 'Error interno del modelo durante la predicción.', 'detalles': str(e)}), 500


# Esto es para correr el servidor (ej: en PyCharm)
if __name__ == '__main__':
    # host='0.0.0.0' hace que sea accesible desde fuera de la máquina
    app.run(host='0.0.0.0', port=5000)
