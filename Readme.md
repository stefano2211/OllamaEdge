# Edge AI


## Description
This is a prototype of an edge cloud system specializing in AI, designed to empower businesses utilizing LangChain. It facilitates the creation of Retrieval Augmented Generation (RAG) AI models trained exclusively on a company's or organization's proprietary data.

## Tools used in this project
* [LangChain](https://www.langchain.com/): LangChain is a software framework that helps facilitate the integration of large language models (LLMs) into applications.
* [Ollama](https://ollama.com/): Ollama is an advanced AI tool that enables users to run large language models (LLMs) locally on their personal computers.
* [FastApi](https://fastapi.tiangolo.com/): astAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.
* [Llama2](https://www.llama.com/llama2/): LLM model



## Set up the environment


1. Create the virtual environment:
```bash
python3 -m venv venv
```
2. Activate the virtual environment:

- For Linux/MacOS:
```bash
source venv/bin/activate
```
- For Command Prompt:
```bash
.\venv\Scripts\activate
```
3. Install dependencies:
- To install all dependencies, run:
```bash
pip install -r requirements-dev.txt
```
- To install only production dependencies, run:
```bash
pip install -r requirements.txt
```
- To install a new package, run:
```bash
pip install <package-name>
```
## Install Ollama
```bash
Ollama run llama2:7b
```

## Run API 

```bash
uvicorn api.main:app
```

## Docker Install

```bash
docker build -t my-fastapi-app .
```

### Docker run

```bash
docker run -p 8000:8000 my-fastapi-app
```

## Contributions

Contributions are welcome. If you would like to contribute, please open an issue or a pull request.