from fastapi import FastAPI, File, UploadFile, HTTPException
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import uvicorn
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
import os
from datetime import datetime
import httpx
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel
from typing import Literal


app = FastAPI()

weather_service_status = "apagado"

async def get_weather_data(location: str = "Madrid") -> dict:
    """
    Obtiene datos del clima desde la API de wttr.in.

    Args:
        location (str): Ubicación para la cual se obtendrá el clima. Por defecto es "Madrid".

    Returns:
        dict: Un diccionario con los datos del clima.
    """
    url = f"https://wttr.in/{location}?format=j1"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            weather_data = response.json()
            return weather_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener datos del clima: {e}")

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

def preprocess_weather_data(weather_data: dict) -> str:
    """
    Preprocesa los datos del clima para convertirlos en un formato de texto.

    Args:
        weather_data (dict): Datos del clima en formato JSON.

    Returns:
        str: Texto preprocesado con los datos del clima.
    """
    current_condition = weather_data["current_condition"][0]
    weather_text = (
        f"Ubicación: {weather_data['nearest_area'][0]['areaName'][0]['value']}\n"
        f"Temperatura: {current_condition['temp_C']}°C\n"
        f"Condición: {current_condition['weatherDesc'][0]['value']}\n"
        f"Humedad: {current_condition['humidity']}%\n"
        f"Viento: {current_condition['windspeedKmph']} km/h\n"
    )
    return weather_text

async def process_and_store_weather_data(vectorstore, location: str = "Madrid") -> None:
    """
    Procesa y almacena datos del clima en un vectorstore.

    Args:
        vectorstore: El vectorstore donde se almacenarán los documentos.
        location (str): Ubicación para la cual se obtendrá el clima. Por defecto es "Madrid".
    """
    weather_data = await get_weather_data(location)
    if weather_data:
        weather_text = preprocess_weather_data(weather_data)
        doc = Document(weather_text, metadata={"source": "wttr.in", "location": location})

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
        split_docs = text_splitter.split_documents([doc])
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
        collection_name="weather_data"
    )
    return vectorstore

def chat(msg: str, reset: bool = False) -> str:
    """
    Genera una respuesta a un mensaje utilizando un modelo de lenguaje y un vectorstore.
    Si reset es True, se borra el buffer de la conversación.

    Args:
        msg (str): El mensaje del usuario.
        reset (bool): Si es True, se resetea la memoria de la conversación.

    Returns:
        str: La respuesta generada por el modelo.
    """
    conversation_memory = ConversationBufferMemory(memory_key="history", input_key="question")

    # Resetear la memoria si el usuario lo solicita
    if reset:
        conversation_memory.clear()
        return "La conversación ha sido reseteada. ¿En qué puedo ayudarte?"

    llm = Ollama(model="llama3.1:8b")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(
        embedding_function=embed_model,
        persist_directory="./db/chroma_db_dir",
        collection_name="weather_data"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    custom_prompt_template = """Quiero que seas bastante flexible a la hora de responder preguntas. Si te preguntan algo relacionado con lo que tienes en el retrieval, responde basado en eso. Si no, responde con una respuesta general.

    Contexto: {context}
    Historial de conversación: {history}
    Pregunta: {question}

    Responde siempre en español.
    Respuesta:
    """
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'history', 'question']
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": conversation_memory,
        }
    )
    
    response = qa.invoke({"query": msg})
    return response['result']

class WeatherControl(BaseModel):
    status: Literal["prendido", "apagado"]
    location: str = "Madrid"  # Ubicación por defecto

@app.post("/weather/")
async def control_weather_service(control: WeatherControl):
    """
    Endpoint para controlar el servicio de obtención de datos del clima.

    Args:
        control (WeatherControl): Objeto con el estado del servicio ("prendido" o "apagado") y la ubicación.

    Returns:
        dict: Mensaje de confirmación del estado del servicio.
    """
    global weather_service_status
    weather_service_status = control.status

    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./db/chroma_db_dir", collection_name="weather_data", embedding_function=embed_model)

    if control.status == "prendido":
        return {"message": "El servicio está prendido. Último dato ya fue almacenado."}
    else:
        await process_and_store_weather_data(vectorstore, control.location)
        return {"message": "Datos del clima obtenidos y almacenados correctamente."}
    

    


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
    reset: bool = False  # Opción para resetear la conversación

@app.post("/chat/")
async def quick_response(message: ChatMessage):
    """
    Endpoint para generar una respuesta a un mensaje.
    Si reset es True, se borra el historial de la conversación.

    Args:
        message (ChatMessage): El mensaje del usuario.

    Returns:
        dict: La respuesta generada.
    """
    response = chat(message.msg, reset=message.reset)
    
    
    return {"response": response}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)