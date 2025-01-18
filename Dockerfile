FROM python:3.11

WORKDIR /api

COPY requirements.txt requirements.txt
COPY . /api
RUN pip install -r requirements.txt

COPY . .


