import random

def barajar_archivo(entrada: str, salida: str):
    try:
        # Lee las líneas del archivo de entrada
        with open(entrada, 'r', encoding='utf-8') as archivo_entrada:
            filas = archivo_entrada.readlines()
        
        # Baraja las líneas aleatoriamente
        random.shuffle(filas)
        
        # Guarda las líneas barajadas en el archivo de salida
        with open(salida, 'w', encoding='utf-8') as archivo_salida:
            archivo_salida.writelines(filas)
        
        print(f"Las filas se han barajado y guardado en '{salida}'")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{entrada}'")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Uso de la función con rutas relativas
barajar_archivo("cleaned_conjuntoDatosBarajadoEliminadosNeutros.txt", "conjuntoDatosBarajado.txt")
