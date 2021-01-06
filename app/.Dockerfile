FROM python:3.7-slim
RUN mkdir /app
WORKDIR /app

ADD ./app
RUN pip install -r requirements.txt

ADD . .

EXPOSE 8080
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8080"]
