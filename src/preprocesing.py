import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser



def load_data():
    ruta = "./data/"
    all_docs = []  # Initialize a list to hold all documents
    for archivo in os.listdir(ruta):
        if archivo.endswith(".csv"):
            file_path = os.path.join(ruta, archivo)  # Construct the full file path
            loader = CSVLoader(file_path=file_path)  # Use the full file path
            data_csv = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
            docs = text_splitter.split_documents(data_csv)
            all_docs.extend(docs)  # Add the split documents to the list
    return all_docs

def crear_vectorstore(docs):
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="chroma_db_dir",
        collection_name="stanford_report_data"
    )
    return vectorstore