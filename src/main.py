from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from preprocesing import process_and_store_data
import time



def main(api_key, simbolos):
    print("Inicializando el almacenamiento de datos...")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./db/chroma_db_dir", collection_name="crypto_data", embedding_function=embed_model)

    while True:
        print("Obteniendo datos de criptomonedas...")
        process_and_store_data(api_key, simbolos, vectorstore)
        print("Datos procesados y almacenados. Esperando un minuto para la próxima actualización...")
        time.sleep(60)  # Espera un minuto antes de la siguiente solicitud

if __name__ == "__main__":
    api_key = '4f8debb2-d650-4186-96ad-a5d73d0576ef'
    simbolos = ['BTC']  # Cambia a una lista
    main(api_key, simbolos)



    