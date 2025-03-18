from collections import Counter

def process_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Eliminar espacios en blanco y líneas vacías
        lines = [line.strip() for line in lines if line.strip()]
        
        # Contar duplicados
        total_lines = len(lines)
        unique_lines = list(set(lines))
        duplicate_count = total_lines - len(unique_lines)
        
        # Contar muestras por clase
        class_counts = Counter(line.split(':')[-1].strip() for line in unique_lines)
        
        # Guardar archivo sin duplicados
        output_file = "cleaned_" + filename
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write('\n'.join(unique_lines))
        
        print("Conteo de clases:")
        for key, value in class_counts.items():
            print(f"{key}: {value}")
        
        print(f"\nNúmero de duplicados eliminados: {duplicate_count}")
        print(f"Archivo sin duplicados guardado como: {output_file}")
    
    except FileNotFoundError:
        print("Error: Archivo no encontrado.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

# Uso: cambiar 'reviews.txt' por el nombre de tu archivo
process_file('conjuntoDatosBarajadoEliminadosNeutros.txt')
