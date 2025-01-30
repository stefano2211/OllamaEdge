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
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel


app = FastAPI()

async def get_data_cripto(api_key: str, simbolo: str) -> dict:
    """
    Obtiene datos de criptomonedas desde la API de CoinMarketCap.

    Args:
        api_key (str): La clave API para autenticar la solicitud.
        simbolo (str): El símbolo de la criptomoneda a consultar.

    Returns:
        dict: Un diccionario con los datos de la criptomoneda, o None si no se encuentra.
    """
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
            if simbolo in datos['data']:
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
    def __init__(self, content: str, metadata: dict = None):
        """
        Inicializa un documento con contenido y metadatos.

        Args:
            content (str): El contenido del documento.
            metadata (dict, optional): Metadatos asociados al documento. Por defecto es None.
        """
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}

async def process_and_store_data(api_key: str, simbolos: list, vectorstore) -> None:
    """
    Procesa y almacena datos de criptomonedas en un vectorstore.

    Args:
        api_key (str): La clave API para autenticar la solicitud.
        simbolos (list): Lista de símbolos de criptomonedas a consultar.
        vectorstore: El vectorstore donde se almacenarán los documentos.
    """
    all_docs = []
    for simbolo in simbolos:
        data = await get_data_cripto(api_key, simbolo)
        if data:
            content = f"Símbolo: {data['simbolo']}, Precio: {data['precio']}, Volumen: {data['volumen']}, Market Cap: {data['market_cap']}, Total Supply: {data['total_supply']}"
            all_docs.append(Document(content))

    if all_docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        split_docs = text_splitter.split_documents(all_docs)
        vectorstore.add_documents(split_docs)

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
    docs = process_file(filepath)
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embed_model,
        persist_directory="./db/chroma_db_dir",
        collection_name="crypto_data"
    )
    return vectorstore

async def fetch_and_store_btc_data(api_key: str, symbols: list) -> None:
    """
    Obtiene y almacena datos de Bitcoin de forma continua.

    Args:
        api_key (str): La clave API para autenticar la solicitud.
        symbols (list): Lista de símbolos de criptomonedas a consultar.
    """
    print("Inicializando el almacenamiento de datos...")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./db/chroma_db_dir", collection_name="crypto_data", embedding_function=embed_model)

    while True:
        try:
            print("Obteniendo datos de criptomonedas...")
            await process_and_store_data(api_key, symbols, vectorstore)
            print("Datos procesados y almacenados. Esperando un minuto para la próxima actualización...")
            await asyncio.sleep(60)
        except Exception as e:
            print(f"Error al obtener o almacenar datos: {e}")
            await asyncio.sleep(60)

def chat(msg: str) -> str:
    """
    Genera una respuesta a un mensaje utilizando un modelo de lenguaje y un vectorstore.

    Args:
        msg (str): El mensaje del usuario.

    Returns:
        str: La respuesta generada por el modelo.
    """
    llm = Ollama(model="deepseek-r1:8b")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(embedding_function=embed_model,
                     persist_directory="./db/chroma_db_dir",
                     collection_name="crypto_data")
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})


    custom_prompt_template = """Quiero que seas bastante flexible a la hora de responder prenguntas si te llegan a preguntar algo relacionado a los
    que tienes en el retrival respondes pero si no responde con una respuesta que no tenga relación con el retrival

    Contexto: {context}
    Pregunta: {question}

    Responde siempre en español
    Respuesta:
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
    """
    Evento de inicio de la aplicación. Inicia la tarea de obtención y almacenamiento de datos de Bitcoin.
    """
    api_key = "4f8debb2-d650-4186-96ad-a5d73d0576ef"  # Reemplaza con tu clave API real
    symbols = ["BTC"]  # Reemplaza con tus símbolos reales
    task = asyncio.create_task(fetch_and_store_btc_data(api_key, symbols))

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint para subir un archivo PDF.

    Args:
        file (UploadFile): El archivo PDF a subir.

    Returns:
        dict: Información sobre el archivo subido.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="¡Por favor, sube un archivo PDF!")

    safe_filename = os.path.basename(file.filename)
    file_location = f"./data/{safe_filename}"

    os.makedirs(os.path.dirname(file_location), exist_ok=True)

    try:
        with open(file_location, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al guardar el archivo: {str(e)}")

    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    create_vectorstore(file_location)

    return {
        "pdf_name": safe_filename,
        "Content-Type": file.content_type,
        "file_location": file_location,
        "file_size": f"{file.size / 1_048_576:.2f} MB",
        "upload_time": upload_time,
    }


class ChatMessage(BaseModel):
    msg: str

    
@app.post("/chat/")
async def quick_response(message: ChatMessage):
    """
    Endpoint para generar una respuesta a un mensaje.

    Args:
        message (ChatMessage): El mensaje del usuario.

    Returns:
        dict: La respuesta generada.
    """
    # Aquí va tu lógica para generar la respuesta
    response = chat(message.msg)  # Asegúrate de que esta función esté definida
    return {"response": response}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)