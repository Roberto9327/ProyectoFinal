import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def procesar_pca(df):
    """Función para procesar el PCA."""
    st.title('Análisis de Componentes Principales (PCA)')

    # Seleccionar las columnas numéricas que deseas usar para PCA
    numerical_columns = ['Tenure in Months', 'Avg Monthly Long Distance Charges', 
                         'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges', 'Total Revenue']
    
    if not all(col in df.columns for col in numerical_columns):
        st.error("Faltan columnas necesarias para el análisis de PCA.")
        return
    
    df_numerical = df[numerical_columns]

    # Estandarizar los datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_numerical)

    # Aplicar PCA para reducir a 2 componentes principales
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

    # Crear un DataFrame con los componentes principales
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

    # Mostrar los resultados numéricos del PCA
    st.subheader("Datos numéricos después del PCA")
    st.dataframe(pca_df)  # Muestra los datos de PCA en un DataFrame de Streamlit

    # Visualización de los resultados del PCA
    st.subheader("Resultados de PCA (2 Componentes Principales)")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, s=100)
    plt.title('Análisis de Componentes Principales (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% varianza explicada)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% varianza explicada)')
    st.pyplot(plt)
