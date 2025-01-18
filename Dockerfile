FROM python:3.11

WORKDIR /api

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .


CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","80"]
