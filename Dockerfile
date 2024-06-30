FROM pytorch/pytorch:latest

RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get -y install curl && \
    apt-get install libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade "pip==22.3"

WORKDIR /app

COPY ./requirements.txt $WORKDIR/

RUN pip install --no-cache-dir -r $WORKDIR/requirements.txt

ENV GIT_PYTHON_REFRESH=quiet

COPY . $WORKDIR
