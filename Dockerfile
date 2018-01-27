FROM ubuntu:latest

# install Tensorflow and Cassandra related requirements
RUN apt-get update \
    && apt-get install -y \
        gcc \
        libev4 \
        libev-dev \
        python \
        python-dev \
        python-pip \
        python-tk

# install required Python modules
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

WORKDIR /opt
