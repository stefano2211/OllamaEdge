from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chains import RetrievalQA


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