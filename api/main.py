from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import uvicorn
from fastapi.responses import JSONResponse
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
import os
from datetime import datetime
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter



app = FastAPI()

async def get_data_cripto(api_key, simbolo):
    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    parameters = {
        'symbol': simbolo,
        'convert': 'USD'
    }
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': api_key,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, params=parameters)
        if response.status_code == 200:
            datos = response.json()
            if simbolo in datos['data']:  # Check that the symbol is in the data
                resultado = {
                    'simbolo': simbolo,
                    'precio': datos['data'][simbolo]['quote']['USD']['price'],
                    'volumen': datos['data'][simbolo]['quote']['USD']['volume_24h'],
                    'market_cap': datos['data'][simbolo]['quote']['USD']['market_cap'],
                    'total_supply': datos['data'][simbolo]['total_supply']
                }
                return resultado
            else:
                print(f"Símbolo {simbolo} no encontrado en los datos.")
                return None
        else:
            print(f"Error en la solicitud: {response.status_code}.")
            return None

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


def process_file(filepath):
    loader = PyPDFLoader(file_path=filepath)  # Usar el archivo específico
    data_csv = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.split_documents(data_csv)
    return docs

def create_vectorstore(filepath):
    docs = process_file(filepath)
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="./db/chroma_db_dir",
        collection_name="crypto_data"
    )
    return vectorstore


async def fetch_and_store_btc_data(api_key, symbols):
    print("Inicializando el almacenamiento de datos...")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./db/chroma_db_dir", collection_name="crypto_data", embedding_function=embed_model)

    while True:
        try:
            print("Obteniendo datos de criptomonedas...")
            await process_and_store_data(api_key, symbols, vectorstore)  # Await the call
            print("Datos procesados y almacenados. Esperando un minuto para la próxima actualización...")
            await asyncio.sleep(60)
        except Exception as e:
            print(f"Error al obtener o almacenar datos: {e}")
            await asyncio.sleep(60)

def chat(msg):
    llm = Ollama(model="llama2:7b")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(embedding_function=embed_model,
                     persist_directory="./db/chroma_db_dir",
                     collection_name="crypto_data")
    
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3})

    custom_prompt_template = """Usa la siguiente información almacenada para responder a la pregunta del usuario sobre btc y los archivos que esten en el retrival.
    Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

    Contexto: {context}
    Pregunta: {question}

    Solo devuelve la respuesta útil a continuación y nada más y responde siempre en español
    Respuesta útil:
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    
    qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=retriever,
                                 return_source_documents=True,
                                 chain_type_kwargs={"prompt": prompt})
    
    response = qa.invoke({"query": msg})
    return response['result']

@app.on_event("startup")
async def startup_event():
    api_key = ""  # Reemplaza con tu clave API real
    symbols = ["BTC"]  # Reemplaza con tus símbolos reales
    task = asyncio.create_task(fetch_and_store_btc_data(api_key, symbols))

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Verificar si el archivo subido es un PDF
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="¡Por favor, sube un archivo PDF!")

    # Sanitizar el nombre del archivo
    safe_filename = os.path.basename(file.filename)
    file_location = f"./data/{safe_filename}"

    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(file_location), exist_ok=True)

    try:
        # Escribir el archivo en la ubicación especificada
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al guardar el archivo: {str(e)}")

    # Obtener la fecha y hora actual
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Llamar a create_vectorstore con el file_location
    create_vectorstore(file_location)

    return {
        "pdf_name": safe_filename,
        "Content-Type": file.content_type,
        "file_location": file_location,
        "file_size": f"{file.size / 1_048_576:.2f} MB",
        "upload_time": upload_time,  # Fecha y hora de la subida
    }
    
@app.post("/chat/")
async def quick_response(msg: str):
    result = chat(msg)
    return result

#hablame del Proof-of-Work de Bitcoin: A Peer-to-Peer Electronic Cash System

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)