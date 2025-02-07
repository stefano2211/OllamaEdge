from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


def process_file(filepath: str) -> list:
    """
    Procesa un archivo PDF y devuelve los documentos extraídos.

    Args:
        filepath (str): La ruta del archivo PDF a procesar.

    Returns:
        list: Lista de documentos extraídos del archivo.
    """
    loader = PyPDFLoader(file_path=filepath)
    data_csv = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.split_documents(data_csv)
    return docs

def create_vectorstore(filepath: str):
    """
    Crea un vectorstore a partir de un archivo PDF.

    Args:
        filepath (str): La ruta del archivo PDF a procesar.

    Returns:
        vectorstore: El vectorstore creado a partir de los documentos extraídos.
    """
    # Autenticación en Hugging Face

    docs = process_file(filepath)
    embed_model = OllamaEmbeddings(model="llama3.1:8b")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="./db/chroma_db_dir",
        collection_name="weather_data"
    )
    return vectorstore