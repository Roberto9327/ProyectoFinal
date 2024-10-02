import streamlit as st
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Función para procesar y visualizar t-SNE
def tsne_visualization(data, n_components=2):
    st.title("t-SNE")

    # Ejecutar t-SNE
    tsne = TSNE(n_components=n_components)
    tsne_results = tsne.fit_transform(data)

    # Visualización en 2D o 3D
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        plt.title("Visualización t-SNE en 2D")
        st.pyplot(plt)
    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2])
        ax.set_title("Visualización t-SNE en 3D")
        st.pyplot(plt)

# Llamar a la función
if 'data' in st.session_state:
    tsne_visualization(st.session_state['data'])
else:
    st.write("No se han cargado datos.")
