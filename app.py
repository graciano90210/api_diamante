import pandas as pd
import joblib
from flask import Flask, request, jsonify

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
        # A. Recibimos los datos del cliente (en formato JSON)
        datos_cliente = request.get_json()

        # B. Convertimos el JSON a un DataFrame de Pandas
        #    ¡Usamos 'columnas_modelo' para asegurar el orden!
        df_cliente = pd.DataFrame(datos_cliente, index=[0])
        df_cliente = df_cliente[columnas_modelo]  # Reordena las columnas

        # C. ¡Usamos el ESCALADOR! (El paso olvidado por muchos)
        datos_escalados = scaler.transform(df_cliente)

        # D. ¡Hacemos la PREDICCIÓN!
        prediccion = model.predict(datos_escalados)

        # E. Interpretamos la respuesta (0 o 1)
        resultado = int(prediccion[0])  # Convertimos de numpy a int

        if resultado == 0:
            status = 'Aprobado'
        else:
            status = 'Rechazado (Alto Riesgo)'

        # F. Devolvemos la respuesta
        return jsonify({
            'prediccion_numerica': resultado,
            'status_credito': status
        })

    except Exception as e:
        # Si algo falla (ej: faltan datos), devolvemos un error
        return jsonify({'error': str(e)}), 400


# Esto es para correr el servidor (ej: en PyCharm)
if __name__ == '__main__':
    # host='0.0.0.0' hace que sea accesible desde fuera de la máquina
    app.run(host='0.0.0.0', port=5000)

