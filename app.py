from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# Load the model and preprocessor with custom objects
model = tf.keras.models.load_model('best_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
preprocessor = joblib.load('preprocessor.joblib')

# Define options for dropdowns (based on dataset)
ciudades = [
    'Lima', 'Arequipa', 'Trujillo', 'Chiclayo', 'Piura', 'Cusco', 'Iquitos',
    'Huancayo', 'Pucallpa', 'Tacna', 'Ayacucho', 'Chimbote', 'Ica', 'Juliaca', 'Tarapoto'
]
categorias = [
    'Documentos', 'Ropa', 'Electrónicos', 'Alimentos', 'Muebles',
    'Libros', 'Medicamentos', 'Repuestos', 'Herramientas', 'Otros'
]
tipos_servicio = ['Estándar', 'Express', 'Económico']
meses = list(range(1, 13))
dias_semana = [
    'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'
]

@app.route('/')
def home():
    return render_template('index.html', 
                          ciudades=ciudades,
                          categorias=categorias,
                          tipos_servicio=tipos_servicio,
                          meses=meses,
                          dias_semana=dias_semana)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        peso = float(data['peso'])
        ciudad_origen = data['ciudad_origen']
        ciudad_destino = data['ciudad_destino']
        categoria = data['categoria']
        tipo_servicio = data['tipo_servicio']
        mes = int(data['mes'])
        dia_semana = dias_semana.index(data['dia_semana'])  # Convert to 0-6
        fragil = data['fragil'] == 'True'

        # Create input DataFrame
        input_data = pd.DataFrame({
            'Peso_Kg': [peso],
            'Ciudad_Origen': [ciudad_origen],
            'Ciudad_Destino': [ciudad_destino],
            'Categoria': [categoria],
            'Tipo_Servicio': [tipo_servicio],
            'Mes': [mes],
            'DiaSemana': [dia_semana],
            'Fragil': [fragil]
        })

        # Preprocess input
        input_processed = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(input_processed)[0][0]

        return jsonify({'prediction': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)