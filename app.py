import streamlit as st
from data_processing import cargar_datos, limpiar_y_preparar_datos
from pca import procesar_pca  # Asegúrate de que el archivo pca.py esté en la misma carpeta o ajusta la ruta

st.title('Análisis de Datos')

st.sidebar.title('Opciones')

# Añadir PCA a las opciones
opciones = ['Cargar datos', 'Clustering Jerárquico', 't-SNE', 'PCA']
opcion = st.sidebar.selectbox('Seleccione una opción', opciones)

if opcion == 'Cargar datos':
    st.write("Sube tu archivo CSV o XLSX para iniciar el análisis.")
    archivo = st.file_uploader('Seleccione un archivo', type=['csv', 'xlsx'])
    
    if archivo:
        st.session_state.df = cargar_datos(archivo)
        if st.session_state.df is not None:
            st.success('Datos cargados exitosamente.')
else:
    if 'df' in st.session_state:
        df = limpiar_y_preparar_datos(st.session_state.df)  # Limpiar datos antes de procesar
        if opcion == 'Clustering Jerárquico':
            from Cluster_Jerarquico import procesar_cluster
            procesar_cluster(df)
        elif opcion == 't-SNE':
            from t_SNE import procesar_tsne
            procesar_tsne(df)
        elif opcion == 'PCA':  # Añadir la opción de PCA
            procesar_pca(df)  # Llamar a la función desde pca.py para procesar y visualizar el PCA
    else:
        st.warning('No hay datos cargados. Por favor, carga un archivo primero.')
