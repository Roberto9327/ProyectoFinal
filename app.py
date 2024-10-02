import streamlit as st
from pages.cluster_jerarquico import cluster_jerarquico
from pages.tsne import tsne_visualization
from pages.data_processing import procesar_datos
from pages.modelos import mostrar_modelos


# Función principal para la navegación
def main():
    st.sidebar.title("Navegación")
    
    # Opciones de la barra lateral
    opciones = ["Entender los Datos", "Cluster Jerárquico", "t-SNE", "Modelos Implementados"]
    eleccion = st.sidebar.radio("Ir a:", opciones)
    
    # Cargar la página según la selección
    if eleccion == "Entender los Datos":
        st.title("Entender, Limpiar y Preparar los Datos")
        file_path = 'data/dataset.csv'  # Ruta del archivo CSV
        st.write("Cargando los datos desde:", file_path)
        data = procesar_datos(file_path)
        st.write("Datos procesados:")
        st.write(data)
    elif eleccion == "Cluster Jerárquico":
        if 'data' in st.session_state:
            cluster_jerarquico(st.session_state['data'])
        else:
            st.write("No se han cargado datos.")
    elif eleccion == "t-SNE":
        if 'data' in st.session_state:
            tsne_visualization(st.session_state['data'])
        else:
            st.write("No se han cargado datos.")
    elif eleccion == "Modelos Implementados":
        mostrar_modelos()

# Llamar a la función principal
if __name__ == "__main__":
    main()
