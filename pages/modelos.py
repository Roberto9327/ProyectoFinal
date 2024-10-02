import streamlit as st

# Página para mostrar todos los modelos implementados
def mostrar_modelos():
    st.title("Lista de Modelos Implementados")

    modelos = {
        "Cluster Jerárquico": "Modelo de agrupamiento jerárquico con dendrograma.",
        "t-SNE": "Modelo de reducción de dimensionalidad con t-SNE.",
        "Word2Vec": "Modelo Word2Vec entrenado con textos de 'Canción de Hielo y Fuego'.",
        "Naive Bayes": "Modelo de clasificación Naive Bayes utilizando CountVectorizer.",
        "TF-IDF": "Modelo de análisis TF-IDF usando datos de películas.",
        "Tokenización y Lemmatización": "Procesamiento de texto con NLTK y SpaCy."
    }

    for modelo, descripcion in modelos.items():
        st.subheader(modelo)
        st.write(descripcion)

# Llamar a la función
if __name__ == "__main__":
    mostrar_modelos()
