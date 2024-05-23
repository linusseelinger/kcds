FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install umbridge scipy numpy

COPY model.py /model.py

CMD python3 model.py