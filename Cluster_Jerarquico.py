import streamlit as st
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Función para procesar y visualizar Clustering Jerárquico
def cluster_jerarquico(data):
    st.title("Cluster Jerárquico")

    # Vinculación utilizando el método 'ward'
    Z = linkage(data, 'ward')
    
    # Dendrograma
    plt.figure(figsize=(10, 7))
    plt.title("Dendrograma - Clustering Jerárquico")
    dendrogram(Z)
    st.pyplot(plt)

    # Modelo de clustering jerárquico
    model = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    labels = model.fit_predict(data)

    # Mostrar los clusters generados
    st.write("Clusters asignados:", labels)

# Llamar a la función
if 'data' in st.session_state:
    cluster_jerarquico(st.session_state['data'])
else:
    st.write("No se han cargado datos.")
