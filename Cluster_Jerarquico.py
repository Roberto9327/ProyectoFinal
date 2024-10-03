import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler, LabelEncoder

def limpiar_y_preparar_datos(df):
    """Función para limpiar y preparar los datos."""
    
    # Mostrar información de las columnas antes de la conversión
    st.write("### Información de las columnas antes de modificar:")
    st.write(df.dtypes)  # Mostrar el tipo de dato de cada columna
    
    # Mostrar los primeros datos antes de la modificación
    st.write("### Datos originales:")
    st.dataframe(df.head())

    # Verificar si hay valores faltantes
    st.write("### Verificación de datos faltantes:")
    st.write(df.isnull().sum())  # Muestra cuántos valores faltantes hay por columna

    # Convertir las columnas categóricas a numéricas
    df = convertir_columnas_categoricas(df)
    
    # Mostrar datos después de la conversión de categóricas
    st.write("### Datos después de convertir variables categóricas a numéricas:")
    st.dataframe(df.head())

    return df  # Retornamos el DataFrame con las variables categóricas convertidas

def convertir_columnas_categoricas(df):
    """Función para convertir variables categóricas en variables numéricas."""
    le = LabelEncoder()

    # Convertir las columnas categóricas a numéricas usando LabelEncoder
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df

def normalizar_datos(df):
    """Función para normalizar las columnas numéricas."""
    
    # Seleccionar solo las columnas numéricas para la normalización
    scaler = StandardScaler()
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    df[columnas_numericas] = scaler.fit_transform(df[columnas_numericas])

    return df

def procesar_cluster(df):
    """Función para procesar el clustering jerárquico."""
    st.subheader('Clustering Jerárquico')
    
    # Llamamos a la función de limpieza y preparación de datos (sin normalización)
    df_numerico = limpiar_y_preparar_datos(df)

    # Mostramos los datos convertidos a numéricos
    st.write("### Datos listos para clustering :")
    st.dataframe(df_numerico)

    # Seleccionar columnas para clustering
    lista_columnas = df_numerico.columns
    columnas = st.sidebar.multiselect('Seleccione las columnas a utilizar', lista_columnas)

    if columnas:
        X = df_numerico[columnas]
        st.write("### Primeras filas de las columnas seleccionadas:")
        st.write(X.head())

        # Seleccionar el tipo de enlace
        enlace = st.sidebar.selectbox('Seleccione el tipo de enlace', ['ward', 'complete', 'average', 'single'])
        
        # Calcular la matriz de enlace
        Z = linkage(X, enlace)

        # Graficar el dendrograma
        fig = plt.figure(figsize=(6, 6))
        corte = st.sidebar.slider('Seleccione el valor de corte', 0, 10, 3)
        dendrogram(Z)
        plt.axhline(y=corte, color='r', linestyle='--')
        st.pyplot(fig)

        # Asignar clusters
        k = st.sidebar.slider('Seleccione el número de clusters', 2, 10, 2)
        clusters = fcluster(Z, k, criterion='maxclust')
        df_numerico['Cluster'] = clusters

        st.write("### Datos con clusters asignados:")
        st.write(df_numerico.head())

        # Graficar los clusters
        fig = plt.figure(figsize=(6, 6))
        sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=clusters, palette='tab10', legend='full')
        st.pyplot(fig)
    else:
        st.warning('Seleccione al menos una columna')

