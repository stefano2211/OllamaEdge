from fastapi import FastAPI
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import uvicorn

app = FastAPI()

def chat(msg):
    llm = Ollama(model="llama2:7b")
    embed_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(embedding_function=embed_model,
                     persist_directory="./db/chroma_db_dir",
                     collection_name="stanford_report_data")
    
    retriever=vectorstore.as_retriever(search_kwargs={'k': 3})

    custom_prompt_template = """Usa la siguiente información para responder a la pregunta del usuario.
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


@app.post("/chat/")
async def quick_response(msg: str):
    result = chat(msg)
    return result

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)