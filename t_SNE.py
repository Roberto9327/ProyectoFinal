import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from data_processing import limpiar_y_preparar_datos

def procesar_tsne(df):
    """Función para procesar t-SNE."""
    st.title('t-SNE')
    
    # Limpiar y preparar los datos
    df_limpio = limpiar_y_preparar_datos(df)

    st.write("Datos limpios:")
    st.dataframe(df_limpio)

    # Convertir todas las columnas categóricas a numéricas antes de aplicar t-SNE
    df_limpio = convertir_columnas_categoricas(df_limpio)

    # Mostrar los datos después de la conversión
    st.write("Datos convertidos a numéricos:")
    st.dataframe(df_limpio)

    # Parámetros para t-SNE
    n_components = st.sidebar.slider('Número de componentes', 2, 3, 2)
    perplexity = st.sidebar.slider('Perplexity', 5, 50, 30)
    n_sample = st.sidebar.slider('Número de muestras', 100, len(df_limpio), min(500, len(df_limpio)))

    # Seleccionar las características y la columna objetivo ('Total Charges')
    x = np.asarray(df_limpio.drop(columns=['Total Charges']))[:n_sample, :]
    y = np.asarray(df_limpio['Total Charges'])[:n_sample].ravel()

    # Ejecutar t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    x2 = tsne.fit_transform(x)

    # Visualización
    if n_components == 2:
        fig = plt.figure(figsize=(6, 6))
        sns.scatterplot(x=x2[:, 0], y=x2[:, 1], hue=y, palette='viridis', s=50)
        st.pyplot(fig)
    else:
        fig_3d = plt.figure(figsize=(6, 6))
        ax = fig_3d.add_subplot(111, projection='3d')
        ax.scatter(x2[:, 0], x2[:, 1], x2[:, 2], c=y, cmap='viridis', s=50)
        st.pyplot(fig_3d)

def convertir_columnas_categoricas(df):
    """Función para convertir variables categóricas en variables numéricas."""
    le = LabelEncoder()

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df
