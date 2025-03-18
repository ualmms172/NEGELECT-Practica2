import tensorflow as tf
import numpy as np
tf.config.set_visible_devices([], 'GPU')


# Cargar el modelo previamente entrenado
modelo = tf.keras.models.load_model("modelo_sentimiento")

# Mapeo de etiquetas a valores numéricos
etiquetas = {"Negative": 0, "Neutral": 1, "Positive": 2}
etiquetas_inversas = {v: k for k, v in etiquetas.items()}



def predecir_sentimiento(texto):
    
    entrada = [texto]
    
    # Obtener probabilidades de cada clase
    probabilidades = modelo.predict(entrada)[0]  # Extraer el primer (y único) resultado
    indice_pred = np.argmax(probabilidades)  # Índice de la clase con mayor probabilidad
    probabilidad_pred = probabilidades[indice_pred]  # Probabilidad de la clase predicha
      
    return indice_pred, probabilidad_pred  # Devuelve clase predicha y su confianza

def evaluar_modelo(archivo_test):
    """ Evalúa el modelo con un archivo test en formato review:valoracion """
    total = 0
    aciertos = 0

    with open(archivo_test, 'r', encoding='utf-8') as file:
        for line in file:
            if ':' in line:
                review, valoracion = line.strip().rsplit(':', 1)
                review = review.strip()
                valoracion = valoracion.strip()

                # Convertir valoración de texto a número
                if valoracion in etiquetas:
                    valoracion_real = etiquetas[valoracion]
                    valoracion_predicha, probabilidad_predicha = predecir_sentimiento(review)

                    print(f"📌 Texto: {review}")
                    print(f"🔹 Valoración real: {valoracion}")
                    print(f"🔹 Predicción: {etiquetas_inversas[valoracion_predicha]} ({probabilidad_predicha:.2%} de confianza)")
                    print("-" * 60)

                    if valoracion_real == valoracion_predicha:
                        aciertos += 1
                    total += 1

    if total == 0:
        print("⚠ No se encontraron datos en el archivo.")
        return
    
    precision = (aciertos / total) * 100
    print(f"\n📌 Evaluación del modelo:")
    print(f"✅ Total de pruebas: {total}")
    print(f"✅ Aciertos: {aciertos}")
    print(f"⚠ Errores: {total - aciertos}")
    print(f"🎯 Precisión del modelo en test: {precision:.2f}%")

# Ejecutar la evaluación con un archivo de prueba
archivo_test = "prueba.txt"  # Reemplázalo con el archivo real
evaluar_modelo(archivo_test)