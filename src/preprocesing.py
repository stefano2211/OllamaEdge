from langchain.text_splitter import RecursiveCharacterTextSplitter
from read import get_data_cripto


class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}



def process_and_store_data(api_key, simbolos, vectorstore):
    all_docs = []  # Inicializa una lista para almacenar todos los documentos
    for simbolo in simbolos:
        data = get_data_cripto(api_key, simbolo)
        if data:  # Verifica que los datos no estén vacíos
            # Crea un documento a partir de los datos
            content = f"Símbolo: {data['simbolo']}, Precio: {data['precio']}, Volumen: {data['volumen']}, Market Cap: {data['market_cap']}, Total Supply: {data['total_supply']}"
            all_docs.append(Document(content))  # Agrega el documento a la lista

    if all_docs:  # Solo procesa si hay datos
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        split_docs = text_splitter.split_documents(all_docs)  # Divide los documentos en fragmentos
        vectorstore.add_documents(split_docs)  # Agrega los nuevos documentos al vectorstore

