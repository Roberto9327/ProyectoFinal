import pandas as pd

# Función para cargar y preparar los datos
def cargar_datos(file_path):
    data = pd.read_csv(file_path)
    return data

# Función para limpiar y preparar los datos
def preparar_datos(data):
    # Eliminar filas con NaN
    data = data.dropna()
    
    # Más procesamiento si es necesario
    return data

# Guardar los datos procesados en la sesión de Streamlit
def procesar_datos(file_path):
    data = cargar_datos(file_path)
    data_limpia = preparar_datos(data)
    return data_limpia

# Llamar a la función
if __name__ == "__main__":
    file_path = 'data/dataset.csv'  # Asegúrate de que la ruta sea correcta
    data_limpia = procesar_datos(file_path)
    print("Datos procesados listos.")

