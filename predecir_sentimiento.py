from flask import Flask, render_template, request, session, jsonify
import tensorflow as tf
import numpy as np
from deep_translator import GoogleTranslator
import io
import matplotlib
matplotlib.use('Agg')  # Evita problemas con el backend de Tkinter

import matplotlib.pyplot as plt
import re

tf.config.set_visible_devices([], 'GPU')


app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_ultrasecreta'  # Cambia esto

# Cargar modelo
modelo = tf.keras.models.load_model("modelo_sentimiento")



# Mapeo de etiquetas
etiquetas = {0: "Negativo", 1: "Neutral", 2: "Positivo"}




def predecir_sentimiento(model, texto):
    
    entrada = [texto]
    probabilidades = model.predict(entrada, verbose=0)
    confianza = np.max(probabilidades)  # Obtener la confianza de la predicción
    indice_pred = np.argmax(probabilidades, axis=1)[0]

    return etiquetas[indice_pred], texto

def generar_grafico_pie(conteo):
    labels = list(conteo.keys())
    sizes = list(conteo.values())
    
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%',
           colors=['#FFC107' , '#F44336', '#4CAF50'])
    ax.axis('equal')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return img.read().hex()

@app.route('/')
def portada():
    return render_template('portada.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            texto = request.form.get('texto', '').strip()
            if not texto:
                return jsonify({'error': 'Texto vacío'}), 400
            
            resultado, traduccion = predecir_sentimiento(modelo, texto)
            nuevo_mensaje = {
                'usuario': texto,
                
                'resultado': resultado
            }
            
            session.setdefault('historial', []).append(nuevo_mensaje)
            session.modified = True
            return jsonify(nuevo_mensaje)
        
        return jsonify({'error': 'Solicitud inválida'}), 400
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(session.get('historial', []))
    
    return render_template('chat.html')

@app.route('/analizar-archivo', methods=['POST'])
def analizar_archivo():
    if 'archivo' not in request.files:
        return jsonify({'error': 'No se ha subido ningún archivo'}), 400

    archivo = request.files['archivo']
    if archivo.filename == '':
        return jsonify({'error': 'Archivo no seleccionado'}), 400

    if archivo and archivo.filename.endswith('.txt'):
        try:
            contenido = archivo.read().decode('utf-8')
            lineas = [linea.strip() for linea in contenido.split('\n') if linea.strip()]

            if not lineas:
                return jsonify({'error': 'El archivo está vacío'}), 400

            batch_size = 32
            resultados = []
            for i in range(0, len(lineas), batch_size):
                batch = lineas[i:i+batch_size]
                predicciones = modelo.predict(batch, verbose=0)
                resultados.extend(np.argmax(predicciones, axis=1))

            conteo = {'Positivo': 0, 'Neutral': 0, 'Negativo': 0}
            for res in resultados:
                sentimiento = etiquetas[res]
                conteo[sentimiento] += 1

            grafico = generar_grafico_pie(conteo)

            return jsonify({
                'conteo': conteo,
                'total': len(lineas),
                'grafico': grafico
            })

        except Exception as e:
            return jsonify({'error': f'Error interno: {str(e)}'}), 500

    return jsonify({'error': 'Formato de archivo no válido'}), 400

@app.route('/file')
def file_upload():
    return render_template('file.html')


@app.route('/guardar_mensaje', methods=['POST'])
def guardar_mensaje():
    data = request.get_json()
    texto = data['texto']

    ruta_archivo = 'datos_entrenamiento2.txt'  # ruta relativa desde app.py

    try:
        with open(ruta_archivo, 'a', encoding='utf-8') as f:
            f.write(texto + '\n')
        return jsonify({'estado': 'Mensaje guardado correctamente'})
    except Exception as e:
        return jsonify({'estado': f'Error al guardar: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000) 