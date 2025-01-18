FROM python:3.11

RUN mkdir -p /home/app

COPY . /home/app

RUN pip install --no-cache-dir -r requirements.txt


CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","80"]
