from langchain.text_splitter import RecursiveCharacterTextSplitter
from read import get_data_cripto


class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}



async def process_and_store_data(api_key, simbolos, vectorstore):
    all_docs = []  # Initialize a list to store all documents
    for simbolo in simbolos:
        data = await get_data_cripto(api_key, simbolo)  # Ensure get_data_cripto is also async
        if data:  # Check that data is not empty
            # Create a document from the data
            content = f"Símbolo: {data['simbolo']}, Precio: {data['precio']}, Volumen: {data['volumen']}, Market Cap: {data['market_cap']}, Total Supply: {data['total_supply']}"
            all_docs.append(Document(content))  # Add the document to the list

    if all_docs:  # Only process if there is data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        split_docs = text_splitter.split_documents(all_docs)  # Split the documents into chunks
        vectorstore.add_documents(split_docs)  # Add the new documents to the vectorstore

