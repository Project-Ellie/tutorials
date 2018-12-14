FROM python:3.6-alpine

WORKDIR /home/microblog

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app app
COPY microblog.py microblog.py
ENV FLASK_APP microblog.py

EXPOSE 5000

ENTRYPOINT ["flask", "run", "--host=0.0.0.0"]